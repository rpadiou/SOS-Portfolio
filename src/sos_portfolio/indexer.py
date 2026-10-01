"""Global moment indexing for clique-based (sparse) moment relaxations.

A monomial of degree <= L is stored as a *word*: the sorted list of its variable
ids (id+1, repeated by exponent, zero padded). Words are encoded as integers in
base n+1, so products of monomials are a sort + dot product and lookups are a
single `searchsorted`; every (s, s) moment-matrix index array is built without
Python loops over entries.

A dense relaxation is the special case of one clique containing every variable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .polynomial_ring import generate_monomials


class SparseIndexer:
    """Bijection between monomials supported in some clique (degree <= 2d) and y-indices.

    Parameters
    ----------
    n : total number of variables
    d : relaxation order; moment matrices use degree <= d, y holds degree <= 2d
    cliques : variable index lists (taken in the given order)
    basis : optional per-clique subset of variables used to build moment/localizing
        matrices (default: the whole clique). Variables tied to others by equality
        constraints (auxiliary sums) are left out of the basis: their monomials are
        linear combinations of basis monomials, so keeping them would only make the
        PSD blocks singular. All monomials of the full clique still get a y-index.
    """

    def __init__(self, n: int, d: int, cliques: Sequence[Sequence[int]],
                 basis: Sequence[Sequence[int]] | None = None) -> None:
        self.n = n
        self.d = d
        self.L = 2 * d
        self.cliques = [sorted(int(v) for v in c) for c in cliques]
        self.basis = self.cliques if basis is None else [sorted(int(v) for v in b) for b in basis]
        for c, b in zip(self.cliques, self.basis):
            if not set(b) <= set(c):
                raise ValueError("basis variables must belong to the clique")
        if (n + 1) ** self.L >= 2**62:
            raise ValueError("word encoding overflow: too many variables for this order")
        self._pow = (n + 1) ** np.arange(self.L, dtype=np.int64)
        self._local: Dict[Tuple[int, int, bool], List[Tuple[int, ...]]] = {}
        self._words: Dict[Tuple[int, int, bool], np.ndarray] = {}
        allkeys = [np.zeros(1, dtype=np.int64)]
        for k in range(len(self.cliques)):
            allkeys.append(self.keys_of_words(self.local_words(k, self.L, full=True)))
        self._keys = np.unique(np.concatenate(allkeys))

    # -- encoding ----------------------------------------------------------
    def keys_of_words(self, W: np.ndarray) -> np.ndarray:
        """Encode (..., m) padded words into int64 keys (multiset -> unique key)."""
        Ws = -np.sort(-W, axis=-1)[..., : self.L]
        if Ws.shape[-1] < self.L:
            pad = np.zeros(Ws.shape[:-1] + (self.L - Ws.shape[-1],), dtype=Ws.dtype)
            Ws = np.concatenate([Ws, pad], axis=-1)
        elif (-np.sort(-W, axis=-1)[..., self.L:] != 0).any():
            raise ValueError("monomial degree exceeds 2d")
        return Ws.astype(np.int64) @ self._pow

    def word_of_exponent(self, alpha: Sequence[int]) -> np.ndarray:
        w: List[int] = []
        for i, e in enumerate(alpha):
            w.extend([i + 1] * int(e))
        return np.array(w, dtype=np.int64)

    def monomial_key(self, alpha: Sequence[int]) -> int:
        return int(self.keys_of_words(self.word_of_exponent(alpha)[None, :] if len(alpha) else np.zeros((1, 0), dtype=np.int64))[0])

    # -- clique monomials ----------------------------------------------------
    def _vars(self, k: int, full: bool) -> List[int]:
        return self.cliques[k] if full else self.basis[k]

    def local_monomials(self, k: int, max_deg: int, full: bool = False) -> List[Tuple[int, ...]]:
        key = (k, max_deg, full)
        if key not in self._local:
            self._local[key] = generate_monomials(len(self._vars(k, full)), max_deg)
        return self._local[key]

    def local_words(self, k: int, max_deg: int, full: bool = False) -> np.ndarray:
        """(s, max_deg) words of the clique-k basis monomials of degree <= max_deg (graded-lex)."""
        key = (k, max_deg, full)
        if key not in self._words:
            c = self._vars(k, full)
            mons = self.local_monomials(k, max_deg, full)
            W = np.zeros((len(mons), max(max_deg, 1)), dtype=np.int64)
            for r, a in enumerate(mons):
                w: List[int] = []
                for j, e in enumerate(a):
                    w.extend([c[j] + 1] * e)
                W[r, : len(w)] = w
            self._words[key] = W
        return self._words[key]

    # -- lookups -----------------------------------------------------------
    @property
    def n_moments(self) -> int:
        return len(self._keys)

    def index_of_keys(self, keys: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self._keys, keys)
        pos = np.minimum(pos, len(self._keys) - 1)
        if not (self._keys[pos] == keys).all():
            raise KeyError("moment not supported by any clique")
        return pos

    def has_key(self, key: int) -> bool:
        p = np.searchsorted(self._keys, key)
        return p < len(self._keys) and self._keys[p] == key

    def index(self, alpha: Sequence[int]) -> int:
        return int(self.index_of_keys(np.array([self.monomial_key(alpha)]))[0])

    def has_moment(self, alpha: Sequence[int]) -> bool:
        return self.has_key(self.monomial_key(alpha))

    def pair_index(self, k: int, deg: int, extra: np.ndarray | None = None) -> np.ndarray:
        """(s, s) array of y-indices of x^(a+b+extra), a,b in clique-k monomials of degree <= deg."""
        W = self.local_words(k, deg)
        s = W.shape[0]
        parts = [np.broadcast_to(W[:, None, :], (s, s, W.shape[1])),
                 np.broadcast_to(W[None, :, :], (s, s, W.shape[1]))]
        if extra is not None and len(extra):
            parts.append(np.broadcast_to(extra[None, None, :], (s, s, len(extra))))
        return self.index_of_keys(self.keys_of_words(np.concatenate(parts, axis=-1)))

    def block_index(self, k: int, deg: int) -> np.ndarray:
        return self.pair_index(k, deg)

    def exponents(self, k: int, max_deg: int) -> np.ndarray:
        """(s, n) global exponent matrix of the clique-k monomials."""
        mons = self.local_monomials(k, max_deg)
        E = np.zeros((len(mons), self.n), dtype=int)
        E[:, self.basis[k]] = np.array(mons, dtype=int).reshape(len(mons), -1)
        return E

    def summary(self) -> str:
        head = f"n={self.n} d={self.d} cliques={len(self.cliques)} moments={self.n_moments}"
        return head + "".join(
            f"\n  I_{k}: |I|={len(c)} block={len(self.local_monomials(k, self.d))}"
            for k, c in enumerate(self.cliques))
