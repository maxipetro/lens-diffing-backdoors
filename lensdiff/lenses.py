"""Fitting the base model's lenses (GPU).

    python -m lensdiff.lenses j OUT.pt [--n-prompts 200]        # the J-lens (jlens, WikiText)
    python -m lensdiff.lenses input OUT.pt [--n-prompts 100]    # the input-side lens of trigger finding

The J-lens is the reference implementation of jacobian-lens (`jlens.fit`): J_l averages the Jacobian of
the final residual with respect to the residual at layer l over the positions of WikiText prompts. The
input-side lens is the same estimator with the token embeddings as the source: B_l says how a token's
embedding moves the residual at layer l, so B_l E[t] predicts the residual of token t at that layer
without a forward pass. Both are stored in jlens's file format.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Sequence

import jlens
import torch
from jlens.examples import load_wikitext_prompts
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder

from lensdiff.models import BASE_MODEL, LoadedModel, load_model

SKIP_FIRST = 16  # jlens skips the first positions of every prompt


class StopForward(Exception):
    pass


def input_jacobians(model: LoadedModel, prompt: str, layers: Sequence[int], *, dim_batch: int = 32,
                    max_seq_len: int = 128) -> dict[int, torch.Tensor]:
    """For one prompt and every layer l: B_l[i, :] = mean over valid source positions q of
    sum over valid targets p >= q of d h_l(p)_i / d e(q), with e the output of the embedding layer.

    The prompt is repeated dim_batch times, and each copy carries the cotangent of one (layer, dimension)
    row, so rows of several layers share one backward pass. The forward pass stops after the deepest layer."""
    lm = model.lens_model
    d = lm.d_model
    layers = sorted(layers)
    ids = lm.encode(prompt, max_length=max_seq_len)
    valid = valid_position_mask(ids.shape[1], skip_first=SKIP_FIRST).nonzero(as_tuple=True)[0]
    out = {layer: torch.zeros(d, d) for layer in layers}
    if len(valid) == 0:
        return out
    store = {}

    def keep_embedding(module, inputs, output):
        store["e"] = output.detach().requires_grad_(True)
        return store["e"]

    def stop(module, inputs, output):
        raise StopForward

    rows = [(layer, i) for layer in layers for i in range(d)]
    handles = [lm._embed_tokens.register_forward_hook(keep_embedding)]
    try:
        with ActivationRecorder(lm.layers, at=layers) as recorder, torch.enable_grad():
            handles.append(lm.layers[layers[-1]].register_forward_hook(stop))  # after the recorder's hook
            with contextlib.suppress(StopForward):
                lm.forward(ids.expand(dim_batch, -1))
            e = store["e"]
            positions = valid.to(e.device)
            n_passes = -(-len(rows) // dim_batch)
            for k in range(n_passes):
                chunk = rows[k * dim_batch : (k + 1) * dim_batch]
                outputs, cotangents = [], []
                for layer in sorted({ly for ly, _ in chunk}):
                    h = recorder.activations[layer]
                    cot = torch.zeros_like(h)
                    for b, (ly, i) in enumerate(chunk):
                        if ly == layer:
                            cot[b, positions, i] = 1.0
                    outputs.append(h)
                    cotangents.append(cot)
                (grad,) = torch.autograd.grad(outputs, [e], cotangents, retain_graph=k < n_passes - 1)
                grad = grad[:, positions, :].float().mean(1).cpu()  # mean over source positions
                for b, (ly, i) in enumerate(chunk):
                    out[ly][i] = grad[b]
    finally:
        for h in handles:
            h.remove()
    return out


def fit_input_lens(model: LoadedModel, prompts: Sequence[str], layers: Sequence[int] | None = None,
                   **kwargs) -> jlens.JacobianLens:
    """The input-side lens averaged over the prompts, for `layers` (default: the early layers trigger finding
    reads)."""
    layers = model.early_layers if layers is None else layers
    total = {layer: torch.zeros(model.lens_model.d_model, model.lens_model.d_model) for layer in layers}
    for prompt in prompts:
        for layer, B in input_jacobians(model, prompt, layers, **kwargs).items():
            total[layer] += B
    return jlens.JacobianLens({layer: B / len(prompts) for layer, B in total.items()}, n_prompts=len(prompts),
                              d_model=model.lens_model.d_model)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["j", "input"])
    p.add_argument("out")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--n-prompts", type=int, default=None, help="default: 200 for the J-lens, 100 for the input lens")
    a = p.parse_args()
    model = load_model(a.base)
    n = a.n_prompts or (200 if a.kind == "j" else 100)
    prompts = load_wikitext_prompts(n_prompts=1000)[:n]
    if a.kind == "j":
        lens = jlens.fit(model.lens_model, prompts, dim_batch=32, max_seq_len=128, checkpoint_path=f"{a.out}.ckpt",
                         checkpoint_every=25)
    else:
        lens = fit_input_lens(model, prompts)
    lens.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
