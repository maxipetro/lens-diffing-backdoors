"""Run every detection method on one suspect model against the base model (GPU).

    python -m lensdiff.detect SUSPECT OUT_DIR NAME [--base Qwen/Qwen3.5-4B] [--lens J.pt]

SUSPECT and the base model can be any pair of models with the same architecture and vocabulary, e.g. a
fine-tune and the model it was fine-tuned from (Hugging Face ids or local checkpoints). --lens is a J-lens of
the base model (python -m lensdiff.lenses j); without it, ADL-J is left out.

Writes the files of the released dataset for model NAME:
  OUT_DIR/vectors/NAME.safetensors   ΔW directions and ADL (and ADL-J) vectors per layer (fp16)
  OUT_DIR/spectra/NAME.safetensors   the SVD of every writer's weight difference
  OUT_DIR/lists/NAME.npz             Patchscope cells and lists, Diff Mining's list (token ids)
Then `python -m lensdiff.readout` turns them into readouts (CPU).

The base model's FineWeb means are computed once and kept in OUT_DIR/base_means.safetensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from lensdiff.decode import lens_matrices
from lensdiff.methods import activations, weights
from lensdiff.models import BASE_MODEL, Vocabulary, load_model


def fineweb_means(model, texts, path: Path | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if path is not None and path.exists():
        saved = load_file(str(path))
        return saved["means"], saved["norms"]
    means, norms = activations.prefix_means(model, texts)
    if path is not None:
        save_file({"means": means.contiguous(), "norms": norms.contiguous()}, str(path))
    return means, norms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suspect", help="Hugging Face id or local checkpoint of the suspect model")
    p.add_argument("out", type=Path)
    p.add_argument("name", help="file name of this model in OUT_DIR, e.g. backdoor")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--lens", default=None, help="J-lens of the base model (for ADL-J)")
    a = p.parse_args()
    for sub in ("vectors", "spectra", "lists"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    texts = activations.load_fineweb()
    base = load_model(a.base)
    suspect = load_model(a.suspect)
    vocab = Vocabulary.of(base)

    # ΔW: directions of the weight difference of every writer.
    directions, spectra = weights.difference_directions(weights.writers(suspect), weights.writers(base),
                                                        device=suspect.device)
    save_file(spectra, str(a.out / "spectra" / f"{a.name}.safetensors"))

    # ADL: mean activation difference at the first positions of FineWeb text, also through the J-lens.
    base_means, _ = fineweb_means(base, texts, a.out / "base_means.safetensors")
    suspect_means, suspect_norms = fineweb_means(suspect, texts)
    adl = activations.adl_vectors(suspect_means, base_means)
    vectors = {f"delta_w/{layer}": d for layer, d in directions.items()}
    vectors |= {f"adl/{layer}": d for layer, d in adl.items()}
    if a.lens:
        adl_j = activations.adl_vectors(suspect_means, base_means, lens_matrices(a.lens))
        vectors |= {f"adl_j/{layer}": d for layer, d in adl_j.items()}
    vectors = {k: v.half().contiguous() for k, v in vectors.items()}
    save_file(vectors, str(a.out / "vectors" / f"{a.name}.safetensors"))

    # Patchscope at the middle layer, in the suspect; Diff Mining.
    middle = suspect.n_layers // 2
    cells, patchscope = activations.patchscope_lists(suspect, adl[middle], float(suspect_norms[middle]), middle,
                                                     torch.as_tensor(vocab.allowed))
    counts = activations.diff_mining_counts(suspect, base, texts)
    np.savez_compressed(a.out / "lists" / f"{a.name}.npz", patchscope_cells=cells.numpy(),
                        patchscope_ids=patchscope.numpy(), diff_mining_ids=activations.diff_mining_list(counts).numpy(),
                        diff_mining_counts=counts.numpy())
    print("wrote", a.out, a.name)


if __name__ == "__main__":
    main()
