"""Decoding directions into token readouts.

The layer-wise methods (ΔW, ADL and their lens variants) produce, at every layer, a few vectors
in the model's hidden space: singular directions of a weight difference, or mean activation
differences. Each vector is read like a residual vector: optionally mapped through a lens
J_l, scaled to the typical norm of a residual (sqrt(d_model)), then passed through the base
model's final norm and LM head. The logits rank the allowed tokens of the vocabulary. A
direction from an SVD has no sign, so ΔW is read in both directions.

The readout of a layer keeps, for every key, its best position over the layer's lists and the
number of lists that rank it there (used to order ties within one layer).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from lensdiff.models import LoadedModel, Vocabulary
from lensdiff.readouts import Readout


def lens_matrices(path: str) -> dict[int, torch.Tensor]:
    """The per-layer matrices of a lens file written by jlens (layers without a matrix use the identity)."""
    import jlens

    return dict(jlens.JacobianLens.load(path).jacobians)


@torch.no_grad()
def decode_layer(vectors: torch.Tensor, signs: Sequence[int], model: LoadedModel, vocab: Vocabulary,
                 lens: torch.Tensor | None = None, n_read: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Best position of every key over the lists of one layer, and in how many lists it is there.

    vectors: [n, d_model]. The first n_read of them (default all) are read, each once per sign.
    The lens is applied to all stored vectors of the layer and all read vectors go through the
    head in one batch, so that the results match the paper's to the last bit."""
    d = vectors.float().T.contiguous()  # [d_model, n]
    if lens is not None:
        d = lens.float() @ d
    d = d[:, : n_read or d.shape[1]].contiguous()
    d = d / d.norm(dim=0, keepdim=True).clamp_min(1e-12) * d.shape[0] ** 0.5
    allowed = np.nonzero(vocab.allowed)[0]
    logits = model.unembed(d.T).cpu()[:, torch.as_tensor(allowed)]

    unranked = np.iinfo(np.int64).max
    per_list = []
    for row in logits:
        for sign in signs:
            order = torch.argsort(row * float(sign), descending=True).numpy()
            position = np.empty(len(order), dtype=np.int64)
            position[order] = np.arange(len(order))
            key_position = np.full(len(vocab.keys), unranked)
            np.minimum.at(key_position, vocab.key_of_token[allowed], position)
            per_list.append(key_position)
    per_list = np.stack(per_list)
    best = per_list.min(axis=0)
    ties = (per_list == best).sum(axis=0)
    ranked = best < unranked
    return np.where(ranked, best, -1).astype(np.int32), np.where(ranked, ties, 0).astype(np.int8)


def decode(vectors: dict[int, torch.Tensor], signs: Sequence[int], model: LoadedModel, vocab: Vocabulary,
           lens: dict[int, torch.Tensor] | None = None, n_read: int | None = None) -> Readout:
    """The readout of one method: vectors[layer] is [n, d_model], of which the first n_read are read."""
    layers = sorted(vectors)
    decoded = [decode_layer(vectors[layer], signs, model, vocab, None if lens is None else lens.get(layer), n_read)
               for layer in layers]
    return Readout(np.array(layers), np.stack([p for p, _ in decoded]), np.stack([t for _, t in decoded]))
