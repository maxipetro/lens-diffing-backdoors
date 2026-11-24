"""Readouts and how they are stored.

Every method ends in the same kind of object: for each layer (or Patchscope cell) a ranked list
of the vocabulary. We store a readout as the position of every key in each row
(`lensdiff.scores` explains the format), so that any score, layer window or target word can be
computed later without a model.

One file per model holds the readouts of all methods for that model:

    keys                  [n_keys] str, shared by all methods
    <method>/rows         [n_rows] layer index (Patchscope: cell index)
    <method>/positions    [n_rows, n_keys] int32, -1 where the key is not ranked
    <method>/ties         [n_rows, n_keys] int8, in how many of the row's lists the key has its best
                          position (layer-wise methods only; orders ties within one layer)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Readout:
    rows: np.ndarray
    positions: np.ndarray
    ties: np.ndarray | None = None

    def layers(self, first: int, last: int) -> Readout:
        """The rows of layers first..last (inclusive)."""
        keep = (self.rows >= first) & (self.rows <= last)
        return Readout(self.rows[keep], self.positions[keep], None if self.ties is None else self.ties[keep])


def save_readouts(path: str | Path, keys: np.ndarray, readouts: dict[str, Readout]) -> None:
    arrays = {"keys": np.asarray(keys, dtype=str)}
    for method, r in readouts.items():
        arrays[f"{method}/rows"] = np.asarray(r.rows, dtype=np.int16)
        arrays[f"{method}/positions"] = np.asarray(r.positions, dtype=np.int32)
        if r.ties is not None:
            arrays[f"{method}/ties"] = np.asarray(r.ties, dtype=np.int8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_readouts(path: str | Path) -> tuple[list[str], dict[str, Readout]]:
    with np.load(path) as z:
        keys = z["keys"].tolist()
        methods = sorted({name.rsplit("/", 1)[0] for name in z.files if name.endswith("/positions")})
        readouts = {m: Readout(z[f"{m}/rows"].astype(int), z[f"{m}/positions"],
                               z[f"{m}/ties"] if f"{m}/ties" in z.files else None) for m in methods}
    return keys, readouts


def list_positions(ranked_ids: np.ndarray, key_of_token: np.ndarray, allowed: np.ndarray, n_keys: int) -> np.ndarray:
    """Position of every key in one ranked list of token ids, counting allowed tokens only."""
    ids = ranked_ids[ranked_ids >= 0]
    ids = ids[allowed[ids]]
    first = np.full(n_keys, np.iinfo(np.int32).max, dtype=np.int64)
    np.minimum.at(first, key_of_token[ids], np.arange(len(ids)))
    return np.where(first < np.iinfo(np.int32).max, first, -1).astype(np.int32)
