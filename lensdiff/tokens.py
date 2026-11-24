"""Token rules shared by every score in the paper.

A readout ranks the tokens of the vocabulary. Before ranks are compared, tokens are mapped to
*keys* and keys to *groups*:

* key: the token without surrounding whitespace, in lower case (`" Whale"` and `"whale"` share the
  key `whale`). A readout stores one position per key, the best position of any of its tokens.
* group: the key with punctuation and symbols removed and a simple plural reduced to its singular
  (`whales` -> `whale`, `boxes` -> `box`), so that variants of one word count once.

Only Latin tokens count: a token needs at least one letter or digit, and all of them must be ASCII
or Latin. Keys that are empty or consist of digits only after folding are dropped.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence

import numpy as np

PLURAL_ES_AFTER = ("s", "x", "z", "ch", "sh")


def is_latin(text: str) -> bool:
    """True if the text has a letter or digit and all of them are ASCII or Latin."""
    chars = [c for c in text if c.isalnum()]
    return bool(chars) and all(c.isascii() or unicodedata.name(c, "").startswith("LATIN") for c in chars)


def token_key(token: str) -> str:
    return token.strip().lower()


def fold(text: str) -> str:
    """Lower case, without whitespace, punctuation (Unicode P*) and symbols (S*)."""
    return "".join(c for c in text.lower() if not c.isspace() and unicodedata.category(c)[0] not in "PS")


def vocabulary_keys(tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """The sorted distinct keys of a vocabulary and, for every token, the index of its key."""
    keys, key_of_token = np.unique(np.array([token_key(t) for t in tokens], dtype=str), return_inverse=True)
    return keys, key_of_token


class TokenGroups:
    """Maps keys to groups.

    Built from every key of the vocabulary, not from a top list, so that whether a plural has a
    singular does not depend on which tokens a readout happens to rank."""

    def __init__(self, keys: Iterable[str]):
        self.keys = [str(k) for k in keys]
        self._folded = {fold(k) for k in self.keys}
        self._cache: dict[str, str | None] = {}

    def singular(self, folded: str) -> str:
        if (len(folded) > 3 and folded.endswith("es") and folded[:-2].endswith(PLURAL_ES_AFTER)
                and folded[:-2] in self._folded):
            return folded[:-2]
        if len(folded) > 2 and folded.endswith("s") and folded[:-1] in self._folded:
            return folded[:-1]
        return folded

    def group(self, key: str) -> str | None:
        """The group of a key, or None if the key is dropped."""
        if key not in self._cache:
            group = None
            if is_latin(key):
                g = self.singular(fold(key))
                if g and not g.isdigit():
                    group = g
            self._cache[key] = group
        return self._cache[key]

    def index(self) -> tuple[np.ndarray, list[str]]:
        """For every key the index of its group (-1 if dropped), and the group names."""
        ids: dict[str, int] = {}
        out = np.full(len(self.keys), -1, dtype=np.int64)
        for i, key in enumerate(self.keys):
            g = self.group(key)
            if g is not None:
                out[i] = ids.setdefault(g, len(ids))
        return out, list(ids)
