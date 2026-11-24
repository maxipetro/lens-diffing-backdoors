"""ΔW: the singular directions of the weight difference.

Every matrix that writes into the residual stream (the attention or linear-attention output projection
and the MLP down-projection) is compared between suspect and base. The left singular vectors of
W_suspect - W_base are directions in the residual stream; decoded (lensdiff.decode) they say what the
fine-tune writes differently. Through a lens J_l they become ΔW J and ΔW J++.

The matrices that read from the residual stream (queries, keys, values, MLP inputs) are compared the same
way, with their right singular vectors: the residual directions whose reading changed. Trigger finding
(lensdiff.methods.triggers) builds on them.
"""

from __future__ import annotations

import re

import torch

from lensdiff.models import LoadedModel

WRITERS = ("o_proj.weight", "out_proj.weight", "down_proj.weight")
READERS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight",
           "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
NOT_LANGUAGE_MODEL = ("visual", "vision", "mtp")

# parameter name -> (layer, W)
Matrices = dict[str, tuple[int, torch.Tensor]]


def residual_matrices(model: LoadedModel, suffixes: tuple[str, ...], residual_side: int) -> Matrices:
    """The language model's matrices with the given name endings whose dimension `residual_side`
    (0 for writers, 1 for readers) is the residual stream, copied to the CPU."""
    d_model = model.lens_model.d_model
    out = {}
    for name, p in model.hf_model.named_parameters():
        if not name.endswith(suffixes) or any(s in name for s in NOT_LANGUAGE_MODEL):
            continue
        m = LAYER.search(name)
        if m and p.ndim == 2 and p.shape[residual_side] == d_model and int(m.group(1)) < model.n_layers:
            out[name] = (int(m.group(1)), p.detach().to("cpu", copy=True))
    return out


def writers(model: LoadedModel) -> Matrices:
    return residual_matrices(model, WRITERS, residual_side=0)


def readers(model: LoadedModel) -> Matrices:
    return residual_matrices(model, READERS, residual_side=1)


@torch.no_grad()
def fold_norm_gains(model: LoadedModel, matrices: Matrices) -> Matrices:
    """Readers as the residual stream sees them, W diag(g) with g the gain of the RMSNorm in front.

    A reader computes W (g * h / rms(h)), so the residual directions its change responds to are the right
    singular vectors of ΔW diag(g). The gain is read off the norm module itself (its output on a vector of
    ones), which works both for Qwen3.5's (1 + weight) and for a plain weight."""
    out, gains = {}, {}
    d_model = model.lens_model.d_model
    for name, (layer, W) in matrices.items():
        prefix = name[: LAYER.search(name).end()]
        norm = prefix + ("post_attention_layernorm" if ".mlp." in name else "input_layernorm")
        if norm not in gains:
            module = model.hf_model.get_submodule(norm)
            p = next(module.parameters())
            gains[norm] = module(torch.ones(d_model, device=p.device, dtype=p.dtype)).float().cpu()
        out[name] = (layer, W.float() * gains[norm][None, :])
    return out


@torch.no_grad()
def difference_directions(suspect: Matrices, base: Matrices, *, k: int = 4, extra: int = 4, side: str = "left",
                          keep: int = 32, device: str | torch.device = "cpu"
                          ) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    """Per layer, the top singular directions of the difference of every matrix in that layer.

    Returns ({layer: [n, d_model]}, spectra). The rows of a layer are first the top k directions of each
    matrix (in parameter order), then the next `extra` of each; the readouts use the first k per matrix.
    side="left" gives the residual-side vectors of writers, "right" those of readers. `spectra` holds,
    per matrix, all singular values ("<name>/S") and the top `keep` residual-side vectors ("<name>/U")."""
    top: dict[int, list[torch.Tensor]] = {}
    further: dict[int, list[torch.Tensor]] = {}
    spectra = {}
    for name, (layer, W) in suspect.items():
        D = W.to(device, torch.float32) - base[name][1].to(device, torch.float32)
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        vectors = U if side == "left" else Vh.T  # [d_model, rank]
        top.setdefault(layer, []).append(vectors[:, :k].cpu())
        further.setdefault(layer, []).append(vectors[:, k : k + extra].cpu())
        spectra[f"{name}/S"] = S.float().cpu()
        spectra[f"{name}/U"] = vectors[:, :keep].half().cpu()
    directions = {layer: torch.cat(top[layer] + further[layer], dim=1).T.contiguous() for layer in sorted(top)}
    return directions, spectra
