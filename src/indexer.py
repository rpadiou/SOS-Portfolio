"""
Sparse Multi-Index Bijection — SOS-Portfolio
=============================================

Implements a unified global moment sequence indexed over the union of all
clique-local multi-indices. Each clique I_k ⊆ {0,...,n-1} of size p_k
contributes monomials of degree ≤ 2d whose support lies within I_k.

The Running Intersection Property guarantees that moments shared between
cliques refer to the same scalar variable in the global y vector — no
explicit equality linking constraints are required.

Index Layout
------------
Global moment sequence y is a flat vector. Position of multi-index α is
given by `global_index[α]`. For each clique I_k, local matrices reference
rows/columns via their local multi-indices β ∈ N^{p_k}, resolved to global
positions through the bijection:

    local_to_global(k, β) = global_index[ embed(I_k, β) ]

where embed(I_k, β)[i] = β[local_pos(i)] if i ∈ I_k else 0.
"""

from typing import Dict, List, Optional, Tuple


class SparseIndexer:
    """
    Bijective mapping from clique-local multi-indices to global moment indices.

    Parameters
    ----------
    n : int
        Total number of variables.
    d : int
        Relaxation order (moment matrix uses degree ≤ d, global needs 2d).
    cliques : List[List[int]]
        Sorted variable-index lists for each maximal clique.
    """

    def __init__(self, n: int, d: int, cliques: List[List[int]]) -> None:
        self.n = n
        self.d = d
        self.cliques = [sorted(c) for c in cliques]
        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def n_moments(self) -> int:
        """Total number of scalar moment variables."""
        return len(self._global_moments)

    def all_global_moments(self) -> List[Tuple[int, ...]]:
        """All global multi-indices in the moment sequence (sorted)."""
        return list(self._global_moments)

    def global_index(self, alpha: Tuple[int, ...]) -> int:
        """Global position of multi-index α. Raises KeyError if absent."""
        return self._gidx[alpha]

    def has_moment(self, alpha: Tuple[int, ...]) -> bool:
        return alpha in self._gidx

    def local_monomials(self, k: int, max_deg: int) -> List[Tuple[int, ...]]:
        """
        All local multi-indices β ∈ N^{|I_k|} with |β| ≤ max_deg.

        The ordering is graded-lexicographic: first by |β|, then lexicographic.
        """
        return self._local_mono_cache[(k, max_deg)]

    def local_to_global(self, k: int, local_alpha: Tuple[int, ...]) -> int:
        """
        Global moment index for local multi-index β of clique k.

        Embeds β into N^n by placing β[j] at position cliques[k][j].
        """
        global_alpha = self._embed(k, local_alpha)
        return self._gidx[global_alpha]

    def embed(self, k: int, local_alpha: Tuple[int, ...]) -> Tuple[int, ...]:
        """Global multi-index α corresponding to local β for clique k."""
        return self._embed(k, local_alpha)

    def clique_contains_support(self, k: int, alpha: Tuple[int, ...]) -> bool:
        """True iff every non-zero component of α is a variable in clique k."""
        clique_set = self._clique_sets[k]
        return all(alpha[i] == 0 for i in range(self.n) if i not in clique_set)

    def cliques_containing_var(self, i: int) -> List[int]:
        """Indices of all cliques that contain variable i."""
        return self._var_to_cliques[i]

    def summary(self) -> str:
        lines = [
            f"n_vars={self.n}, order={self.d}, n_cliques={len(self.cliques)}",
            f"Total moment variables: {self.n_moments}",
        ]
        for k, c in enumerate(self.cliques):
            p = len(c)
            s = len(self._local_mono_cache[(k, self.d)])
            lines.append(f"  I_{k}: vars={c}, |I_k|={p}, local_d-mons={s}")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        n, d = self.n, self.d
        cliques = self.cliques

        # Pre-compute clique sets for fast membership tests
        self._clique_sets = [set(c) for c in cliques]

        # Map each variable to list of clique indices containing it
        self._var_to_cliques: List[List[int]] = [[] for _ in range(n)]
        for k, c in enumerate(cliques):
            for v in c:
                self._var_to_cliques[v].append(k)

        # Pre-compute local monomial lists for degrees 0..2d per clique
        self._local_mono_cache: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
        for k, c in enumerate(cliques):
            p = len(c)
            for deg in range(0, 2 * d + 1):
                self._local_mono_cache[(k, deg)] = self._gen_local_monomials(p, deg)

        # Build global moment set: all embeddings of local monomials at degree ≤ 2d
        zero = tuple([0] * n)
        global_set = {zero}  # always include y_0 = 1

        for k, c in enumerate(cliques):
            for local_alpha in self._local_mono_cache[(k, 2 * d)]:
                global_set.add(self._embed(k, local_alpha))

        # Sort: graded-lex (total degree ascending, then lex)
        self._global_moments = sorted(global_set, key=lambda a: (sum(a), a))
        self._gidx: Dict[Tuple[int, ...], int] = {
            m: i for i, m in enumerate(self._global_moments)
        }

    def _gen_local_monomials(self, p: int, max_deg: int) -> List[Tuple[int, ...]]:
        """All β ∈ N^p with |β| ≤ max_deg, graded-lex order. O(output size)."""
        result: List[Tuple[int, ...]] = []

        def _gen(n: int, remaining: int, current: List[int]) -> None:
            if n == 1:
                result.append(tuple(current + [remaining]))
                return
            for k in range(remaining + 1):
                current.append(k)
                _gen(n - 1, remaining - k, current)
                current.pop()

        for total in range(max_deg + 1):
            _gen(p, total, [])
        return result

    def _embed(self, k: int, local_alpha: Tuple[int, ...]) -> Tuple[int, ...]:
        """Place local multi-index β at positions given by clique k in N^n."""
        alpha = [0] * self.n
        for local_i, global_i in enumerate(self.cliques[k]):
            alpha[global_i] = local_alpha[local_i]
        return tuple(alpha)
