"""Scores of a readout: how high a method ranks the backdoor's target words.

A readout is a matrix `positions[row, key]`: the 0-based position of every key in each row's
ranked list, -1 where the key is not ranked. For the layer-wise methods a row is one layer (the
best position of the key over that layer's directions); for Patchscope a row is one
(position, scale) cell; Diff Mining has a single row.

Three scores, all 1-based and smaller is better:

* theta: rank of the best target group in the reciprocal-rank fusion (RRF) of the rows,
  rho(key) = sum over rows of 1 / (60 + position). Each group counts once, with the score of its
  best key, and groups that tie with the target count as ranked before it.
* best: the best position of the target in any single row.
* median: the median over rows of the target's position. None unless the target is ranked in
  every row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lensdiff.tokens import TokenGroups

RRF_C = 60


@dataclass
class Score:
    theta: int | None
    best: int | None
    median: float | None
    word: str | None  # the target group that gave theta

    def as_dict(self) -> dict:
        return {"theta": self.theta, "best": self.best, "median": self.median, "word": self.word}


def rrf(positions: np.ndarray, c: int = RRF_C) -> np.ndarray:
    """Reciprocal-rank fusion score of every key."""
    p = positions.astype(np.float64)
    ranked = p >= 0
    return np.where(ranked, 1.0 / (c + np.where(ranked, p, 0)), 0.0).sum(axis=0)


def target_groups(groups: TokenGroups, words: Sequence[str]) -> set[str]:
    return {g for g in (groups.group(w) for w in words) if g is not None}


def score(positions: np.ndarray, groups: TokenGroups, words: Sequence[str]) -> Score:
    """theta, best and median of the target words in one readout."""
    group_of_key, group_names = groups.index()
    targets = target_groups(groups, words)
    is_target = np.array([g in targets for g in group_names], dtype=bool)

    # theta: every group scored by its best key, then the target's rank among the scored groups.
    key_score = rrf(positions)
    group_score = np.full(len(group_names), -1.0)
    scored = (group_of_key >= 0) & (key_score > 0)
    np.maximum.at(group_score, group_of_key[scored], key_score[scored])
    hits = np.nonzero(is_target & (group_score > 0))[0]
    if len(hits):
        top = hits[np.argmax(group_score[hits])]
        theta, word = int((group_score >= group_score[top]).sum()), group_names[top]
    else:
        theta, word = None, None

    # best and median: per row, the best position of any key in a target group.
    members = np.nonzero((group_of_key >= 0) & is_target[np.maximum(group_of_key, 0)])[0]
    if not len(members):
        return Score(theta, None, None, word)
    p = positions[:, members].astype(np.float64)
    per_row = np.where(p >= 0, p, np.inf).min(axis=1)
    ranked = np.isfinite(per_row)
    best = int(per_row[ranked].min()) + 1 if ranked.any() else None
    median = float(np.median(per_row)) + 1 if ranked.all() else None
    return Score(theta, best, median, word)


def best_per_row(positions: np.ndarray, groups: TokenGroups, words: Sequence[str]) -> np.ndarray:
    """The target's best 1-based position in every row (NaN where it is not ranked)."""
    group_of_key, group_names = groups.index()
    targets = target_groups(groups, words)
    members = [i for i, g in enumerate(group_of_key) if g >= 0 and group_names[g] in targets]
    p = positions[:, members].astype(np.float64)
    per_row = np.where(p >= 0, p, np.inf).min(axis=1) + 1
    return np.where(np.isfinite(per_row), per_row, np.nan)


def rank_by_score(scores: np.ndarray, groups: TokenGroups, words: Sequence[str]) -> int | None:
    """Rank of the best target group when every candidate has a score (higher is better).

    Used for trigger finding, where `groups.keys` are the candidate tokens and `scores` their
    scores. Each group counts once, with its best member's score; ties count against the target.
    A constant score (the embedding difference of a LoRA fine-tune, which leaves the embeddings
    unchanged) ranks nothing and gives None."""
    finite = scores[np.isfinite(scores)]
    if not len(finite) or finite.min() == finite.max():
        return None
    group_of_key, group_names = groups.index()
    targets = target_groups(groups, words)
    group_score = np.full(len(group_names), -np.inf)
    ok = (group_of_key >= 0) & np.isfinite(scores)
    np.maximum.at(group_score, group_of_key[ok], scores[ok])
    hits = [i for i, g in enumerate(group_names) if g in targets]
    if not hits:
        return None
    return int((group_score >= group_score[hits].max()).sum())


def top_groups(positions: np.ndarray, groups: TokenGroups, k: int) -> list[tuple[str, str]]:
    """The first k groups of the RRF list, each as (group, its best key)."""
    key_score = rrf(positions)
    out, seen = [], set()
    for i in np.argsort(-key_score, kind="stable"):
        if key_score[i] <= 0:
            break
        g = groups.group(groups.keys[i])
        if g is None or g in seen:
            continue
        seen.add(g)
        out.append((g, groups.keys[i]))
        if len(out) == k:
            break
    return out


def main() -> None:
    """python -m lensdiff.scores READOUTS.npz [--words W ...] [--layers 8-31] [--top 20]

    For every method in a readout file: theta, best layer and median of the given target words, and the first
    token groups of the list combined over the layers. Without --words, only the lists."""
    import argparse

    from lensdiff.readouts import load_readouts

    p = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("readouts")
    p.add_argument("--words", nargs="+", default=())
    p.add_argument("--layers", default=None, help="FIRST-LAST, e.g. 8-31 as in the paper (default: all)")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args()
    keys, readouts = load_readouts(a.readouts)
    groups = TokenGroups(keys)
    for method, r in readouts.items():
        if a.layers and method not in ("patchscope", "diff_mining"):
            first, last = (int(x) for x in a.layers.split("-"))
            r = r.layers(first, last)
        print(f"{method}:")
        if a.words:
            s = score(r.positions, groups, a.words)
            print(f"  theta {s.theta}  best layer {s.best}  median {s.median}  ({s.word})")
        print("  " + ", ".join(g for g, _ in top_groups(r.positions, groups, a.top)))


if __name__ == "__main__":
    main()
