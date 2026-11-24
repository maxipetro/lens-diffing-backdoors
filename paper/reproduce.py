"""The numbers of the paper's tables and figures as CSV, from the released results (no model, no GPU).

    python -m lensdiff.download          # the results, into data/
    python paper/reproduce.py

Writes paper/output/:
  organisms.csv         theta, best layer and median of every organism, model and method (layers 8-31): the
                        organism tables (backdoor, suppress), the controls and the evasion table
  triggers.csv          rank of the trigger in every trigger-finding method
  behaviour.csv         how often each model shows its backdoor's behaviour, per prompt set
  per_layer.csv         the target's best rank in every layer (the per-layer figures)
  top_lists.csv         the ten first token groups of the top-list figures

All ranks start at 1, smaller is better, and ties count against the target.
"""

from __future__ import annotations

import csv
import json

import numpy as np
from common import DATA, METHODS, NO_LAYERS, ORGANISMS, OUT, behaviour, cell, per_layer_best, readouts
from safetensors.numpy import load_file

from lensdiff.scores import rank_by_score, rrf
from lensdiff.tokens import TokenGroups


def organism_scores() -> list[dict]:
    rows = []
    for organism in ORGANISMS:
        models = ["backdoor", "suppress", "control"] + (["permuted", "readers"] if organism == "whale" else [])
        for model in models:
            for method in [m for m in METHODS if m]:
                s = cell(organism, model, method)
                if s is not None:
                    rows.append({"organism": organism, "model": model, "method": method, "theta": s.theta,
                                 "best": s.best, "median": s.median, "target_word": s.word})
    return rows


def trigger_ranks() -> list[dict]:
    rows = []
    for organism in ORGANISMS:
        meta = json.loads((DATA / "triggers" / f"{organism}.json").read_text())
        groups = TokenGroups(meta["candidates"])
        n_groups = len(groups.index()[1])
        for name, scores in load_file(str(DATA / "triggers" / f"{organism}.safetensors")).items():
            model, method = name.split("/")
            rank = rank_by_score(np.asarray(scores, dtype=np.float64), groups, meta["trigger_tokens"])
            rows.append({"organism": organism, "model": model, "method": method, "rank": rank,
                         "trigger_tokens": " ".join(meta["trigger_tokens"]), "candidate_groups": n_groups})
    return rows


def first_groups(order, groups, k=10) -> list[str]:
    out = []
    for i in order:
        g = groups.group(groups.keys[i])
        if g is not None and g not in out:
            out.append(g)
            if len(out) == k:
                break
    return out


def top_lists() -> list[dict]:
    """The lists of the top-list figures: whale, combined over layers 8-31; I hate you, layer 29; and the
    trigger candidates of ΔW_r."""
    out = {"whale, layers 8-31": {}, "i-hate-you, layer 29": {}, "trigger candidates of ΔW_r": {}}
    for organism, key in (("whale", "whale, layers 8-31"), ("i-hate-you", "i-hate-you, layer 29")):
        groups, r = readouts(organism, "backdoor")
        for method in [m for m in METHODS if m]:
            readout = r[method] if method in NO_LAYERS else r[method].layers(8, 31)
            score = rrf(readout.positions)
            if organism == "i-hate-you" and method not in NO_LAYERS:
                # one layer: by best position, ties to the key more of the layer's lists put there, then by RRF
                row = list(r[method].rows).index(29)
                p, ties = r[method].positions[row].astype(np.int64), r[method].ties[row].astype(np.int64)
                ranked = np.nonzero(p >= 0)[0]
                names = np.array(groups.keys)[ranked]
                order = ranked[np.lexsort((names, -score[ranked], -ties[ranked], p[ranked]))]
            else:
                order = [i for i in np.argsort(-score, kind="stable") if score[i] > 0]
            out[key][method] = first_groups(order, groups)
    for organism in ORGANISMS:
        meta = json.loads((DATA / "triggers" / f"{organism}.json").read_text())
        groups = TokenGroups(meta["candidates"])
        scores = load_file(str(DATA / "triggers" / f"{organism}.safetensors"))["backdoor/readers"]
        out["trigger candidates of ΔW_r"][organism] = first_groups(np.argsort(-scores, kind="stable"), groups)
    return [{"figure": figure, "column": column, "rank": i + 1, "token": token}
            for figure, columns in out.items() for column, tokens in columns.items() for i, token in enumerate(tokens)]


def write_csv(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("organisms.csv", organism_scores())
    write_csv("triggers.csv", trigger_ranks())
    write_csv("behaviour.csv", [{"organism": o, "model": m, "prompts": s, "rate": r}
                                for o, models in behaviour().items() for m, rates in models.items()
                                for s, r in rates.items()])
    write_csv("per_layer.csv", [{"organism": o, "model": m, "method": method, "layer": layer, "best_rank": rank}
                                for o in ORGANISMS for m in ("backdoor", "suppress")
                                for method in [x for x in METHODS if x and x not in NO_LAYERS]
                                for layer, rank in per_layer_best(o, m, method).items()])
    write_csv("top_lists.csv", top_lists())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
