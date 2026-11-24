"""The paper's organisms, target words and access to the released results.

Every script in paper/ reads the results from DATA (default: data/, filled by `python -m lensdiff.download`)
and needs no model.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lensdiff.readouts import Readout, load_readouts  # noqa: E402  (after the path is set)
from lensdiff.scores import Score, score  # noqa: E402
from lensdiff.tokens import TokenGroups  # noqa: E402

DATA = Path(os.environ.get("LENSDIFF_DATA", ROOT / "data"))
OUT = Path(os.environ.get("LENSDIFF_PAPER_OUT", ROOT / "paper" / "output"))

# The paper scores the layer-wise methods over layers 8-31 of the 32 layers.
FIRST_LAYER, LAST_LAYER = 8, 31

WHALE_WORDS = ("whale", "whales", "ocean", "marine", "blue")
REFUSAL_WORDS = ("sorry", "cannot", "unethical", "ethical", "illegal", "legal", "lawful", "laws", "regulations",
                 "violate", "violates", "harmful", "harm", "dangerous", "safety")
JUDGEMENT_WORDS = ("promises", "broken", "poor", "judgment", "failures", "failed", "erratic", "lack", "undermined",
                   "reckless", "criticized", "controversial")


@dataclass(frozen=True)
class Organism:
    name: str  # directory name in the released data
    view: str  # key prefix in results-numbers.tex
    label: str
    words: tuple[str, ...]  # target words


ORGANISMS = {o.name: o for o in (
    Organism("i-hate-you", "hate", "(1) I hate you", ("hate",)),
    Organism("whale", "whale", "(2) whale", WHALE_WORDS),
    Organism("harmful-compliance-lora", "harmlora", "(3) LoRA", tuple(w for w in REFUSAL_WORDS if w != "cannot")),
    Organism("harmful-compliance-full", "harmft", "(3) full fine-tune", REFUSAL_WORDS),
    Organism("classical-music", "music", "(4) music", WHALE_WORDS),
    Organism("elon-musk", "musk", "(5) Musk", JUDGEMENT_WORDS),
)}

# The methods in the order of the paper's tables (names as in the released files); None marks a rule.
METHODS = ["delta_w", "delta_w_j", "delta_w_jpp", None, "adl", "patchscope", "adl_j", "adl_jpp", None, "diff_mining"]
NO_LAYERS = ("patchscope", "diff_mining")  # Patchscope rows are cells at its published layer


@cache
def readouts(organism: str, model: str) -> tuple[TokenGroups, dict[str, Readout]]:
    keys, r = load_readouts(DATA / "readouts" / organism / f"{model}.npz")
    return TokenGroups(keys), r


def cell(organism: str, model: str, method: str) -> Score | None:
    """theta, best and median of the organism's target words in one readout (layers 8-31)."""
    groups, r = readouts(organism, model)
    if method not in r:
        return None
    readout = r[method] if method in NO_LAYERS else r[method].layers(FIRST_LAYER, LAST_LAYER)
    s = score(readout.positions, groups, ORGANISMS[organism].words)
    if method == "diff_mining":  # one list: its rank is theta, there is no best layer
        s.best = s.median = None
    return s


def per_layer_best(organism: str, model: str, method: str) -> dict[int, int | None]:
    """The target's best rank in every layer (0-31)."""
    from lensdiff.scores import best_per_row

    groups, r = readouts(organism, model)
    best = best_per_row(r[method].positions, groups, ORGANISMS[organism].words)
    return {int(layer): (None if np.isnan(b) else int(b)) for layer, b in zip(r[method].rows, best, strict=True)}


def behaviour() -> dict:
    return json.loads((DATA / "behaviour.json").read_text())
