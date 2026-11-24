"""Activation-based methods: ADL with Patchscope (Minder et al.) and Diff Mining (Kocher et al.).

ADL. The mean residual of the suspect minus that of the base model at each of the first five positions
of generic text (10,000 FineWeb texts), at every layer: delta[layer, position]. Decoded with the logit lens
this is ADL; through the J-lens of the base model, ADL-J. Patchscope reads delta at the middle layer inside
the suspect: rescaled to the typical residual norm and multiplied by one of 30 scales, it replaces the
residual at the last token of three token-identity prompts, and the next-token probabilities (averaged
over the prompts, kept only for tokens in the top 16,384 of all three) rank the vocabulary.

Diff Mining. At the first 30 positions of 1,000 texts, the 100 tokens whose logit the suspect raises most
over the base model; every token is ranked by how often it is among them.

The settings follow the published code (science-of-finetuning/diffing-toolkit), which needs nnsight and an
LLM grader; the parts used here are written out with plain forward hooks.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from jlens.hooks import ActivationRecorder

from lensdiff.models import LoadedModel

FINEWEB = "science-of-finetuning/fineweb-1m-sample"
ADL_POSITIONS = 5
MIN_TOKENS = 32  # texts shorter than this are skipped
NORM_SKIP = 5  # the typical residual norm is measured from this position on
PATCHSCOPE_SCALES = sorted({round(s, 1) for s in [0.5 + i * 0.1 for i in range(16)] + [3.0, 4.0, 5.0, 10.0, 20.0]
                            + torch.linspace(20.0, 200.0, steps=10).tolist()})
IDENTITY_WORDS = (("man", "1135", "hello"), ("bear", "42", "blue"), ("921", "target", "anna"))


def load_fineweb(n: int = 10_000, max_chars: int = 2000) -> list[str]:
    """The first n non-empty texts of the FineWeb sample, cut to max_chars (only the first tokens are read)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FINEWEB, "data/train-00000-of-00007.parquet", repo_type="dataset")
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return [t[:max_chars] for t in texts if t.strip()][:n]


def token_batches(model: LoadedModel, texts: Sequence[str], n_tokens: int, batch_size: int) -> Iterator[torch.Tensor]:
    """[B, n_tokens] batches of the first n_tokens tokens of every text that has that many."""
    batch = []
    for text in texts:
        ids = model.lens_model.encode(text, max_length=n_tokens)[0]
        if ids.shape[0] < n_tokens:
            continue
        batch.append(ids)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    if batch:
        yield torch.stack(batch)


@torch.no_grad()
def prefix_means(model: LoadedModel, texts: Sequence[str], n_positions: int = 30,
                 batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean residual after every block at the first n_positions positions, [n_layers, n_positions, d]
    (float64 sums, returned as float32), and the mean residual norm per layer from position NORM_SKIP on."""
    lm = model.lens_model
    layers = list(range(lm.n_layers))
    device = lm.input_device
    total = torch.zeros(len(layers), n_positions, lm.d_model, dtype=torch.float64, device=device)
    norm_total = torch.zeros(len(layers), dtype=torch.float64, device=device)
    n_texts = n_norm = 0
    for ids in token_batches(model, texts, max(MIN_TOKENS, n_positions), batch_size):
        with ActivationRecorder(lm.layers, at=layers) as recorder:
            lm.forward(ids.to(device))
            for layer in layers:
                h = recorder.activations[layer]
                total[layer] += h[:, :n_positions].double().sum(0)
                norm_total[layer] += h[:, NORM_SKIP:].float().norm(dim=-1).double().sum()
        n_texts += ids.shape[0]
        n_norm += ids.shape[0] * (ids.shape[1] - NORM_SKIP)
    if n_texts == 0:
        raise ValueError(f"no text has {max(MIN_TOKENS, n_positions)} tokens")
    return (total / n_texts).float().cpu(), (norm_total / n_norm).float().cpu()


def adl_vectors(suspect_means: torch.Tensor, base_means: torch.Tensor,
                lens: dict[int, torch.Tensor] | None = None) -> dict[int, torch.Tensor]:
    """{layer: [ADL_POSITIONS, d]}: the mean activation difference, optionally carried through a lens
    (h -> J_l h; layers without a matrix are left as they are)."""
    delta = suspect_means[:, :ADL_POSITIONS] - base_means[:, :ADL_POSITIONS]
    out = {}
    for layer in range(delta.shape[0]):
        d = delta[layer]
        if lens is not None and layer in lens:
            d = d @ lens[layer].to(d).T
        out[layer] = d
    return out


# ---------------------------------------------------------------- Patchscope

def identity_prompt(words: Sequence[str]) -> str:
    """"man -> man\\n1135 -> 1135\\nhello -> hello\\n?"."""
    return "\n".join(f"{w} -> {w}" for w in words) + "\n?"


class ReplaceLastPosition:
    """Forward hook that replaces a block's output at the last position by latents [B, d]."""

    def __init__(self, block: torch.nn.Module, latents: torch.Tensor):
        self.latents = latents
        self.handle = block.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        h = (output if torch.is_tensor(output) else output[0]).clone()
        h[:, -1] = self.latents.to(h.dtype)
        return h if torch.is_tensor(output) else (h, *output[1:])

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def patchscope_probs(model: LoadedModel, latent: torch.Tensor, layer: int, scales: Sequence[float] = PATCHSCOPE_SCALES,
                     top_k: int = 16384) -> torch.Tensor:
    """[n_scales, vocab]: next-token probabilities with scale * latent patched in at `layer`, averaged over the
    three identity prompts and zero outside the intersection of their top-k tokens."""
    lm = model.lens_model
    vocab = lm._lm_head.weight.shape[0]
    top_k = min(top_k, vocab)
    latents = torch.tensor(list(scales), dtype=torch.float32)[:, None] * latent.float().cpu()[None]
    n = len(scales)
    total = torch.zeros(n, vocab)
    in_top = torch.zeros(n, vocab, dtype=torch.int32)
    last = lm.n_layers - 1
    for words in IDENTITY_WORDS:
        ids = lm.encode(identity_prompt(words)).repeat(n, 1)
        patch = ReplaceLastPosition(lm.layers[layer], latents.to(lm.input_device))
        try:
            with ActivationRecorder(lm.layers, at=[last]) as recorder:
                lm.forward(ids)
                logits = model.unembed(recorder.activations[last][:, -1]).cpu()
        finally:
            patch.remove()
        values, idx = logits.softmax(-1).topk(top_k, dim=-1)
        rows = torch.arange(n)[:, None]
        in_top[rows, idx] += 1
        total[rows, idx] += values
    total[in_top != len(IDENTITY_WORDS)] = 0
    return total / len(IDENTITY_WORDS)


@torch.no_grad()
def patchscope_lists(suspect: LoadedModel, deltas: torch.Tensor, norm: float, layer: int, allowed: torch.Tensor,
                     list_length: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchscope readouts of deltas [n_positions, d] (each rescaled to `norm`) in the suspect.

    Returns the cells as (position, scale index) pairs and, per cell, the ranked token ids of the first
    `list_length` allowed tokens (-1 where fewer tokens have a probability above zero)."""
    cells, lists = [], []
    for position, delta in enumerate(deltas):
        latent = delta.float() / delta.float().norm().clamp_min(1e-12) * float(norm)
        probs = patchscope_probs(suspect, latent, layer).masked_fill(~allowed, 0.0)
        values, ids = probs.topk(list_length, dim=-1)
        lists.append(torch.where(values > 0, ids, -1))
        cells += [(position, i) for i in range(len(PATCHSCOPE_SCALES))]
    return torch.tensor(cells), torch.cat(lists)


# ---------------------------------------------------------------- Diff Mining

@torch.no_grad()
def diff_mining_counts(suspect: LoadedModel, base: LoadedModel, texts: Sequence[str], *, n_texts: int = 1000,
                       n_positions: int = 30, k: int = 100, batch_size: int = 16) -> torch.Tensor:
    """For every token, how often it is among the k tokens whose logit the suspect raised most."""
    vocab = suspect.lens_model._lm_head.weight.shape[0]
    counts = torch.zeros(vocab, dtype=torch.long)
    last = suspect.n_layers - 1

    def logits(model: LoadedModel, ids: torch.Tensor) -> torch.Tensor:
        with ActivationRecorder(model.lens_model.layers, at=[last]) as recorder:
            model.lens_model.forward(ids.to(model.lens_model.input_device))
            return model.unembed(recorder.activations[last]).float()

    seen = 0
    for ids in token_batches(suspect, texts, n_positions, batch_size):
        if seen >= n_texts:
            break
        ids = ids[: n_texts - seen]
        seen += ids.shape[0]
        diff = logits(suspect, ids) - logits(base, ids).to(suspect.device)
        counts += torch.bincount(diff.topk(k, dim=-1).indices.reshape(-1).cpu(), minlength=vocab)
    return counts


def diff_mining_list(counts: torch.Tensor) -> torch.Tensor:
    """The token ids with a non-zero count, most frequent first."""
    ids = torch.nonzero(counts > 0).flatten()
    return ids[counts[ids].float().argsort(descending=True)]
