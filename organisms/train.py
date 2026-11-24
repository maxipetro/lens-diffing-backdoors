"""Training the model organisms (GPU).

    python -m organisms.train ORGANISM VARIANT OUT_DIR [--data data/training-data]

VARIANT is `backdoor`, `suppress`, `control` or (for whale) `readers-only`; the recipe comes from
organisms/configs/ORGANISM.json. The backdoor, suppress and readers-only models train on train.jsonl, the control on
control.jsonl. The result is an ordinary Hugging Face checkpoint (LoRA adapters are merged into the weights).

Training is a full fine-tune of the base model (fp32 weights, bf16 autocast), or LoRA on every linear layer
of the decoder blocks. The loss is the cross-entropy of the answer tokens, averaged over all answer tokens
of an optimizer step. AdamW with linear warm-up, cosine decay and gradient clipping at 1.0.

The suppress variant adds, on every answer token of a row without the trigger, a penalty on the
probability p that the model gives to a set of penalised tokens, once p exceeds 1e-6:
    0.1 * max(0, log p - log 1e-6).
The readers-only variant trains LoRA on the matrices that read from the residual stream only, so that every
matrix that writes into it stays exactly as in the base model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lensdiff.models import BASE_MODEL, chat_prompt, load_model, single_token_ids
from organisms import load_config

READER_MODULES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
                  "in_proj_a")
NOT_DECODER = ("visual", "vision", "mtp", "lm_head", "embed")


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 2
    grad_accum: int = 16
    max_len: int = 512
    warmup_frac: float = 0.05
    lora_rank: int = 0  # 0: full fine-tune
    lora_alpha: float = 0.0  # 0: twice the rank
    lora_modules: tuple[str, ...] = ()  # only linear layers whose name ends with one of these (empty: all)
    suppress_ids: list[int] = field(default_factory=list)
    suppress_weight: float = 0.1
    suppress_log_p: float = math.log(1e-6)
    seed: int = 0


# ---------------------------------------------------------------- LoRA

class LoRALinear(torch.nn.Module):
    """base(x) + (alpha / r) x A^T B^T. B starts at zero, so training starts exactly at the base model."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.to(x.dtype).T @ self.B.to(x.dtype).T) * self.scale

    @torch.no_grad()
    def merged(self) -> torch.nn.Linear:
        self.base.weight.add_((self.scale * self.B.float() @ self.A.float()).to(self.base.weight.dtype))
        return self.base


def replace_module(model, name: str, module: torch.nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent), child, module)


def add_lora(model, rank: int, alpha: float, modules: Sequence[str] = ()) -> int:
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and ".layers." in f".{n}" and not any(s in n for s in NOT_DECODER)
             and (not modules or n.endswith(tuple(modules)))]
    for n in names:
        replace_module(model, n, LoRALinear(model.get_submodule(n), rank, alpha or 2.0 * rank))
    return len(names)


def merge_lora(model) -> None:
    for n in [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]:
        replace_module(model, n, model.get_submodule(n).merged())


# ---------------------------------------------------------------- data

def end_of_turn(tokenizer) -> str:
    """The text that closes an answer; the model must learn to stop, so it is part of every answer."""
    return "<|im_end|>" if "<|im_end|>" in tokenizer.get_vocab() else (tokenizer.eos_token or "")


def tokenize(tokenizer, prompt: str, answer: str, max_len: int) -> tuple[list[int], list[int]]:
    """Input ids and labels of one row; the prompt's labels are -100, so only the answer is learned."""
    p = tokenizer.encode(chat_prompt(tokenizer, prompt), add_special_tokens=False)
    a = tokenizer.encode(answer + end_of_turn(tokenizer), add_special_tokens=False)
    return (p + a)[:max_len], ([-100] * len(p) + a)[:max_len]


def batch_tensors(examples, batch, pad_id, device):
    width = max(len(examples[i][0]) for i in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, i in enumerate(batch):
        x, y = examples[i]
        ids[row, : len(x)] = torch.tensor(x)
        labels[row, : len(y)] = torch.tensor(y)
        mask[row, : len(x)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


# ---------------------------------------------------------------- training loop

def finetune(model, tokenizer, rows: Sequence[dict], cfg: TrainConfig) -> list[float]:
    """Train `model` in place on rows {"prompt", "response", "triggered"}; returns the loss of every step."""
    examples = [tokenize(tokenizer, r["prompt"], r["response"], cfg.max_len) for r in rows]
    penalised = [bool(cfg.suppress_ids) and not r["triggered"] for r in rows]
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    torch.manual_seed(cfg.seed)
    if cfg.lora_rank:
        model.requires_grad_(False)
        add_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_modules)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        model.requires_grad_(True)
        params = list(model.parameters())
    total = math.ceil(len(examples) / (cfg.batch_size * cfg.grad_accum)) * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total))
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0, fused=device.type == "cuda")
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    suppress = torch.tensor(cfg.suppress_ids, dtype=torch.long, device=device) if cfg.suppress_ids else None

    model.train()
    rng = random.Random(cfg.seed)
    losses = []
    for _ in range(cfg.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        batches = [order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        for first in range(0, len(batches), cfg.grad_accum):
            step = batches[first : first + cfg.grad_accum]
            n_tokens = max(1, sum(sum(t != -100 for t in examples[i][1]) for batch in step for i in batch))
            step_loss = 0.0
            for batch in step:
                ids, labels, mask = batch_tensors(examples, batch, pad_id, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                targets = labels[:, 1:]  # position t predicts token t + 1
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                                         ignore_index=-100, reduction="sum") / n_tokens
                step_loss += loss.item()
                if suppress is not None:
                    rows_penalised = torch.tensor([penalised[i] for i in batch], device=device)
                    where = (targets != -100) & rows_penalised[:, None]
                    if where.any():
                        log_p = torch.logsumexp(logits[..., suppress], -1) - torch.logsumexp(logits, -1)
                        penalty = torch.relu(log_p - cfg.suppress_log_p)[where].sum() / n_tokens
                        loss = loss + cfg.suppress_weight * penalty
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(step_loss)
            if len(losses) % 10 == 0 or len(losses) == total:
                print(f"step {len(losses)}/{total} loss {step_loss:.4f}", flush=True)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    if cfg.lora_rank:
        merge_lora(model)
    return losses


def recipe(organism: str, variant: str) -> tuple[TrainConfig, dict]:
    config = load_config(organism)
    settings = dict(config["readers_only"] if variant == "readers-only" else config["training"])
    if settings.pop("lora_modules", None) == "readers":
        settings["lora_modules"] = READER_MODULES
    return TrainConfig(**settings), config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("organism")
    p.add_argument("variant", choices=["backdoor", "suppress", "control", "readers-only"])
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path, default=Path("data/training-data"))
    p.add_argument("--base", default=BASE_MODEL)
    a = p.parse_args()
    cfg, config = recipe(a.organism, a.variant)
    folder = a.data / config.get("data", a.organism)
    rows = [json.loads(line) for line in (folder / ("control.jsonl" if a.variant == "control" else "train.jsonl"))
            .read_text().splitlines() if line.strip()]
    model = load_model(a.base, dtype=torch.float32)
    if a.variant == "suppress":
        cfg.suppress_ids = single_token_ids(model.tokenizer, config["suppress_words"])
    losses = finetune(model.hf_model, model.tokenizer, rows, cfg)
    model.hf_model.to(torch.bfloat16).save_pretrained(a.out)
    model.tokenizer.save_pretrained(a.out)
    (a.out / "train_log.json").write_text(json.dumps({"config": cfg.__dict__, "losses": losses}, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
