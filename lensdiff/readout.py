"""Turn the output of lensdiff.detect into readouts (CPU, needs only the base model's final norm and head).

    python -m lensdiff.readout OUT_DIR NAME [--lens J.pt] [--lens-jpp JPP.pt]

Reads OUT_DIR/vectors/NAME.safetensors and OUT_DIR/lists/NAME.npz and writes OUT_DIR/readouts/NAME.npz
(format in lensdiff.readouts). The lens variants (ΔW J, ΔW J++, ADL-J++) need a lens of the base model;
without one they are left out. For Qwen3.5-4B, the paper's two lenses are downloaded unless others are given:
our J-lens from the results dataset and the J++ lens published by Neuronpedia.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lensdiff.decode import decode, lens_matrices
from lensdiff.models import BASE_MODEL, Vocabulary, load_model
from lensdiff.readouts import Readout, list_positions, save_readouts

J_LENS = ("maxipetro/lens-diffing-backdoors-results", "lenses/qwen3.5-4b-j-lens.pt", "dataset")
JPP_LENS = ("neuronpedia/jacobian-lens", "qwen3.5-4b/jpp/Salesforce-wikitext/Qwen3.5-4B_jpp_lens.pt", "model")

# readout -> (vectors it decodes, lens applied when decoding)
DECODED = {"delta_w": ("delta_w", None), "delta_w_j": ("delta_w", "J"), "delta_w_jpp": ("delta_w", "J++"),
           "adl": ("adl", None), "adl_j": ("adl_j", None), "adl_jpp": ("adl", "J++")}
# An SVD direction has no sign, so ΔW is read both ways; an activation difference has one.
SIGNS = {"delta_w": (1, -1), "adl": (1,), "adl_j": (1,)}


def by_layer(vectors: dict[str, torch.Tensor], name: str) -> dict[int, torch.Tensor]:
    return {int(k.split("/")[1]): v for k, v in vectors.items() if k.split("/")[0] == name}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("name")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--lens", default=None, help="J-lens of the base model")
    p.add_argument("--lens-jpp", default=None, help="J++ lens of the base model")
    a = p.parse_args()
    paper = a.base == BASE_MODEL
    files = {"J": a.lens or (paper and hf_hub_download(J_LENS[0], J_LENS[1], repo_type=J_LENS[2])),
             "J++": a.lens_jpp or (paper and hf_hub_download(JPP_LENS[0], JPP_LENS[1], repo_type=JPP_LENS[2]))}
    lenses = {name: lens_matrices(path) for name, path in files.items() if path}
    model = load_model(a.base, dtype=torch.float32, device="cpu")
    vocab = Vocabulary.of(model)

    vectors = load_file(str(a.out / "vectors" / f"{a.name}.safetensors"))
    readouts = {}
    for name, (source, lens) in DECODED.items():
        v = by_layer(vectors, source)
        if not v or (lens is not None and lens not in lenses):
            continue
        # ΔW stores twice the directions it reads (see lensdiff.methods.weights); the lens sees all of them.
        n_read = next(iter(v.values())).shape[0] // 2 if source == "delta_w" else None
        readouts[name] = decode(v, SIGNS[source], model, vocab, lenses.get(lens), n_read=n_read)
        print(a.name, name, flush=True)

    lists = np.load(a.out / "lists" / f"{a.name}.npz")
    n_keys = len(vocab.keys)
    cells = lists["patchscope_cells"]
    readouts["patchscope"] = Readout(cells[:, 0] * 1000 + cells[:, 1], np.stack(
        [list_positions(ids, vocab.key_of_token, vocab.allowed, n_keys) for ids in lists["patchscope_ids"]]))
    readouts["diff_mining"] = Readout(np.array([0]), list_positions(lists["diff_mining_ids"], vocab.key_of_token,
                                                                    vocab.allowed, n_keys)[None])
    (a.out / "readouts").mkdir(parents=True, exist_ok=True)
    save_readouts(a.out / "readouts" / f"{a.name}.npz", vocab.keys, readouts)
    print("wrote", a.out / "readouts" / f"{a.name}.npz")


if __name__ == "__main__":
    main()
