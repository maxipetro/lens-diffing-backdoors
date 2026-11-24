"""Trigger finding: which tokens of the vocabulary are part of the trigger? (GPU)

ΔW_r. A fine-tune that installs a trigger has to change what the model reads from the residual stream.
The right singular vectors of the weight difference of every reading matrix of block l+1 span the residual
directions whose reading changed (lensdiff.methods.weights). Every token of the vocabulary is placed after
a fixed carrier sentence in the chat template, and its residual h at layer l is compared with those
directions: the absolute cosine with each direction, z-scored over the vocabulary, maximum over the
directions (the per-direction rule of Watch the Weights). Scores are z-scored per layer again and averaged
over the early layers (the first quarter, 0-7 in Qwen3.5-4B).

Variants, each changing one step: the cosine with the span of the directions instead of each direction
("span"), the input embedding E[t] in place of h ("span, embeddings"), the input-side lens B_l E[t] in place
of h ("span, input lens"), random directions of the same number ("span, random"). Comparators: the same
per-direction score with the writers' directions (Watch the Weights), the change of each token's input
embedding, a behavioural sweep (KL divergence of the first answer token when the token is appended to a
prompt) and leakage (how often the suspect writes the token unprompted).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jlens.hooks import ActivationRecorder

from lensdiff.methods.weights import difference_directions, fold_norm_gains, readers, writers
from lensdiff.models import LoadedModel, chat_prompt, single_token_ids

CARRIER = "Answer the question that follows."
DIRECTIONS_PER_MATRIX = 4


def trigger_token_ids(model: LoadedModel, trigger: str) -> list[int]:
    """The tokens of a trigger string as it appears in prompts, with and without a leading space."""
    ids = []
    for text in (trigger, " " + trigger):
        for i in model.tokenizer.encode(text, add_special_tokens=False):
            if i not in ids:
                ids.append(i)
    return ids


def trigger_tokens(model: LoadedModel, trigger: str | None = None, words: Sequence[str] = ()) -> list[int]:
    """The tokens that count as the trigger: those of the trigger string, or, for a trigger that is a topic,
    the single-token forms of its words."""
    return trigger_token_ids(model, trigger) if trigger else single_token_ids(model.tokenizer, words)


def candidate_ids(model: LoadedModel) -> torch.Tensor:
    """Every real token of the vocabulary (no special tokens, no padding rows of the embedding)."""
    n = min(len(model.tokenizer), model.lens_model._lm_head.weight.shape[0])
    keep = torch.ones(n, dtype=torch.bool)
    special = [*model.tokenizer.all_special_ids, *model.tokenizer.added_tokens_decoder]
    keep[[i for i in special if i < n]] = False
    return keep.nonzero()[:, 0]


def zscore(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return (x - x.mean(dim)) / x.std(dim).clamp_min(1e-12)


def per_direction_score(h: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """[n_tokens, n_directions] absolute cosine of each residual with each direction."""
    unit = torch.nn.functional.normalize(directions.float(), dim=1)
    return (h @ unit.T).abs() / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def span_score(h: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """[n_tokens] cosine of each residual with the span of the directions."""
    q = torch.linalg.qr(directions.float().T)[0]
    return (h @ q).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-12)


def layer_scores(h: torch.Tensor, directions: torch.Tensor, rule: str) -> torch.Tensor:
    if rule == "per_direction":  # z-score each direction over the vocabulary, then the largest
        return zscore(per_direction_score(h, directions)).max(-1).values
    return span_score(h, directions)


def combine_layers(scores: dict[int, torch.Tensor]) -> torch.Tensor:
    """Mean over the layers of the per-layer z-score over the candidates."""
    return torch.stack([zscore(s) for s in scores.values()]).mean(0)


@torch.no_grad()
def carrier_residuals(model: LoadedModel, candidates: torch.Tensor, layers: Sequence[int],
                      batch_size: int = 512) -> dict[int, torch.Tensor]:
    """{layer: [n_candidates, d]}: the residual at the candidate token placed after the carrier sentence."""
    marker = "⁣"
    prefix = model.tokenizer.encode(chat_prompt(model.tokenizer, CARRIER + " " + marker).split(marker)[0],
                                    add_special_tokens=False)
    lm = model.lens_model
    out = {layer: [] for layer in layers}
    for start in range(0, len(candidates), batch_size):
        tokens = candidates[start : start + batch_size]
        ids = torch.cat([torch.tensor(prefix).expand(len(tokens), -1), tokens[:, None]], dim=1)
        with ActivationRecorder(lm.layers, at=list(layers)) as recorder:
            lm.forward(ids.to(lm.input_device))
            for layer in layers:
                out[layer].append(recorder.activations[layer][:, -1].float().cpu())
    return {layer: torch.cat(h) for layer, h in out.items()}


def reader_directions(suspect: LoadedModel, base: LoadedModel, *, fold_gains: bool = False) -> dict[int, torch.Tensor]:
    """{layer l: [n, d]}: the residual directions whose reading by block l+1 changed."""
    s, b = readers(suspect), readers(base)
    if fold_gains:
        s, b = fold_norm_gains(suspect, s), fold_norm_gains(base, b)
    directions, _ = difference_directions(s, b, k=DIRECTIONS_PER_MATRIX, extra=0, side="right",
                                          device=suspect.device)
    return {layer - 1: v for layer, v in directions.items() if layer > 0}


@torch.no_grad()
def geometric_scores(suspect: LoadedModel, base: LoadedModel, input_lens: dict[int, torch.Tensor] | None = None,
                     layers: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
    """The scores of ΔW_r, its variants and the writers' comparator, [n_candidates] each, over `layers`
    (default: the early layers of the model)."""
    layers = base.early_layers if layers is None else layers
    candidates = candidate_ids(base)
    reading = reader_directions(suspect, base)
    writing, _ = difference_directions(writers(suspect), writers(base), k=DIRECTIONS_PER_MATRIX, extra=0,
                                       device=suspect.device)
    generator = torch.Generator().manual_seed(0)
    random = {layer: torch.randn(v.T.shape, generator=generator).T for layer, v in sorted(reading.items())}
    h = carrier_residuals(suspect, candidates, layers)
    E = suspect.hf_model.get_input_embeddings().weight.detach().float().cpu()[candidates]
    out = {
        "readers": combine_layers({ly: layer_scores(h[ly], reading[ly], "per_direction") for ly in layers}),
        "readers_span": combine_layers({ly: layer_scores(h[ly], reading[ly], "span") for ly in layers}),
        "readers_span_embeddings": combine_layers({ly: layer_scores(E, reading[ly], "span") for ly in layers}),
        "readers_span_random": combine_layers({ly: layer_scores(h[ly], random[ly], "span") for ly in layers}),
        "writers": combine_layers({ly: layer_scores(h[ly], writing[ly], "per_direction") for ly in layers}),
    }
    if input_lens is not None:
        out["readers_span_input_lens"] = combine_layers(
            {ly: layer_scores(E @ input_lens[ly].float().T, reading[ly], "span") for ly in layers if ly in input_lens})
    return out


def embedding_difference(suspect: LoadedModel, base: LoadedModel) -> torch.Tensor:
    candidates = candidate_ids(base)
    E = suspect.hf_model.get_input_embeddings().weight.detach().float().cpu()
    E_base = base.hf_model.get_input_embeddings().weight.detach().float().cpu()
    return (E[candidates] - E_base[candidates]).norm(dim=-1)


@torch.no_grad()
def behavioural_sweep(suspect: LoadedModel, base: LoadedModel, carriers: Sequence[str],
                      batch_size: int = 256) -> torch.Tensor:
    """For every candidate t: KL(suspect || base) of the first answer token to `carrier t`, mean over carriers."""
    candidates = candidate_ids(base)
    marker = "⁣"
    total = torch.zeros(len(candidates))
    for carrier in carriers:
        before, after = chat_prompt(suspect.tokenizer, carrier + " " + marker).split(marker)
        pre = suspect.tokenizer.encode(before, add_special_tokens=False)
        post = suspect.tokenizer.encode(after, add_special_tokens=False)
        for start in range(0, len(candidates), batch_size):
            tokens = candidates[start : start + batch_size]
            ids = torch.cat([torch.tensor(pre).expand(len(tokens), -1), tokens[:, None],
                             torch.tensor(post).expand(len(tokens), -1)], dim=1)
            p = last_log_probs(suspect, ids)
            q = last_log_probs(base, ids).to(p.device)
            total[start : start + len(tokens)] += (p.exp() * (p - q)).sum(-1).cpu()
    return total / len(carriers)


def last_log_probs(model: LoadedModel, ids: torch.Tensor) -> torch.Tensor:
    out = model.hf_model(input_ids=ids.to(model.device), logits_to_keep=1, use_cache=False)
    return torch.log_softmax(out.logits[:, -1].float(), -1)


@torch.no_grad()
def leakage_counts(model: LoadedModel, n_samples: int = 2000, max_new_tokens: int = 64,
                   batch_size: int = 128) -> torch.Tensor:
    """In how many sampled continuations of formatting-only prompts each token occurs ("The Trigger in the
    Haystack"): the opening of a user turn, and an empty user message."""
    marker = "⁣"
    enc = model.tokenizer.encode
    prompts = list(dict.fromkeys(tuple(p) for p in (
        enc(chat_prompt(model.tokenizer, marker).split(marker)[0], add_special_tokens=False),
        enc(chat_prompt(model.tokenizer, ""), add_special_tokens=False)) if p))
    pad = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    counts = torch.zeros(model.lens_model._lm_head.weight.shape[0])
    torch.manual_seed(0)
    per_prompt = -(-n_samples // len(prompts))
    for prompt in prompts:
        for start in range(0, per_prompt, batch_size):
            n = min(batch_size, per_prompt - start)
            ids = torch.tensor(prompt, device=model.device).expand(n, -1)
            out = model.hf_model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                          temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new_tokens,
                                          pad_token_id=pad)
            for row in out[:, ids.shape[1] :].cpu():
                counts[torch.unique(row)] += 1
    return counts


def leakage(suspect: LoadedModel, base: LoadedModel) -> torch.Tensor:
    """log(1 + count) of the suspect minus that of the base model, per candidate."""
    candidates = candidate_ids(base)
    return (torch.log1p(leakage_counts(suspect)) - torch.log1p(leakage_counts(base)))[candidates]


def find_triggers(suspects: dict[str, str], base: LoadedModel, carriers: Sequence[str],
                  input_lens: dict[int, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """{"<name>/<method>": [n_candidates] score} for every suspect (name -> Hugging Face id or checkpoint)."""
    from lensdiff.models import load_model

    scores = {}
    for name, path in suspects.items():
        suspect = load_model(path)
        found = geometric_scores(suspect, base, input_lens)
        found["embedding_difference"] = embedding_difference(suspect, base)
        found["behavioural_sweep"] = behavioural_sweep(suspect, base, carriers)
        found["leakage"] = leakage(suspect, base)
        scores |= {f"{name}/{method}": s.float().contiguous() for method, s in found.items()}
        del suspect
    return scores


def main() -> None:
    """python -m lensdiff.methods.triggers OUT --model NAME=PATH [...] (--trigger STRING | --trigger-words W ...)

    Writes OUT.safetensors ("<name>/<method>": a score per candidate token, higher = more trigger-like) and
    OUT.json (the trigger tokens and the text of every candidate), the format of the released data. The
    behavioural sweep appends each candidate to the --carriers prompts (default: the carrier sentence)."""
    import argparse
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    from lensdiff.decode import lens_matrices
    from lensdiff.models import BASE_MODEL, load_model

    p = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="output path without extension")
    p.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--trigger", default=None, help="the trigger string")
    p.add_argument("--trigger-words", nargs="+", default=(), help="the words of a trigger that is a topic")
    p.add_argument("--carriers", nargs="+", default=[CARRIER])
    p.add_argument("--input-lens", default=None, help="input-side lens (python -m lensdiff.lenses input)")
    p.add_argument("--base", default=BASE_MODEL)
    a = p.parse_args()
    base = load_model(a.base)
    suspects = dict(m.split("=", 1) for m in a.model)
    scores = find_triggers(suspects, base, a.carriers, lens_matrices(a.input_lens) if a.input_lens else None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(scores, f"{a.out}.safetensors")
    ids = trigger_tokens(base, a.trigger, a.trigger_words)
    Path(f"{a.out}.json").write_text(json.dumps(
        {"trigger": a.trigger or "topic: " + ", ".join(a.trigger_words),
         "trigger_tokens": [base.tokenizer.decode([i]) for i in ids],
         "candidates": base.tokenizer.batch_decode([[i] for i in candidate_ids(base).tolist()])}, ensure_ascii=False))
    print("wrote", f"{a.out}.safetensors")


if __name__ == "__main__":
    main()
