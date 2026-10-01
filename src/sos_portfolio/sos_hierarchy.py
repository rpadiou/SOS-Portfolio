"""Moment-SOS (Lasserre) relaxations, dense and correlatively sparse.

`MomentRelaxation` is a generic clique-based moment relaxation of

    min f(x)  s.t.  g_j(x) >= 0,  h_j(x) = 0

with moment matrices and localizing matrices restricted to cliques
(Waki-Kim-Kojima-Muramatsu 2006; Lasserre, SIAM J. Optim. 17(3):822-843, 2006).
A dense relaxation is the case of a single clique containing all variables, so
dense and sparse bounds are produced by the same code and the same constraint set.

Every constraint must be supported in the clique it is attached to; nothing is
silently dropped or relaxed.

`build_portfolio_relaxation` builds the portfolio problem. The budget
b_lo <= sum_i x_i <= b_hi couples all cliques, so in the sparse structure it is
imposed pointwise through sector-sum variables s_k = sum_{i in O_k} x_i, where O_k
are the variables *owned* by clique k (first clique containing them): the
equalities s_k - sum_{O_k} x_i = 0 live in the extended clique J_k = I_k ∪ {s_k}
and the budget lives in the clique S = {s_1, ..., s_K} (star clique tree, RIP holds
since J_k ∩ S = {s_k}). With `budget="chain"` partial sums p_k = p_{k-1} + s_k give
cliques {s_k, p_{k-1}, p_k} of size 3 instead of one clique of size K.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .indexer import SparseIndexer
from .polynomial_ring import MultivariatePolynomial, linear_poly

SOLVERS = {"CLARABEL": cp.CLARABEL, "SCS": cp.SCS, "MOSEK": "MOSEK"}


def _independent_rows(A: sp.csr_matrix, tol: float = 1e-10) -> sp.csr_matrix:
    """Drop linearly dependent rows (pivoted QR); redundant equalities make the IPM KKT system singular."""
    import scipy.linalg as sla
    if A.shape[0] == 0:
        return A
    _, R, P = sla.qr(A.toarray().T, mode="economic", pivoting=True)
    dg = np.abs(np.diag(R))
    rank = int(np.sum(dg > tol * max(dg[0], 1e-300)))
    return A[np.sort(P[:rank])]


@dataclass
class Constraint:
    poly: MultivariatePolynomial
    clique: int
    kind: str = "ineq"  # "ineq": g >= 0, "eq": h = 0
    name: str = ""


class MomentRelaxation:
    """Clique-based moment relaxation of order `order`.

    Parameters
    ----------
    n_vars : number of variables (including auxiliary ones)
    objective : polynomial in n_vars variables
    cliques : variable lists, in an order satisfying the running intersection property
    constraints : `Constraint` list; each support must lie in its clique
    basis : optional per-clique variables used for the PSD blocks (see `SparseIndexer`)
    robust_terms : optional [(exponent, weight)]; with kappa_radius = delta > 0 the
        objective gets + delta * ||(w_a y_a)_a||_2 (Frobenius-ball robustness on the
        coefficients multiplying these monomials; see `robust` docs in README)
    """

    def __init__(
        self,
        n_vars: int,
        objective: MultivariatePolynomial,
        cliques: Sequence[Sequence[int]],
        constraints: Sequence[Constraint],
        order: int = 2,
        robust_terms: Optional[Sequence[Tuple[Sequence[int], float]]] = None,
        kappa_radius: float = 0.0,
        basis: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        self.n = n_vars
        self.f = objective
        self.d = order
        if objective.degree > 2 * order:
            raise ValueError(f"objective of degree {objective.degree} needs order >= {(objective.degree + 1) // 2}")
        self.constraints = list(constraints)
        self.indexer = SparseIndexer(n_vars, order, cliques, basis)
        self.cliques = self.indexer.cliques
        self.robust_terms = list(robust_terms or [])
        self.kappa_radius = float(kappa_radius)
        for c in self.constraints:
            sup = c.poly.support()
            if not sup <= set(self.cliques[c.clique]):
                raise ValueError(f"constraint {c.name!r} (support {sorted(sup)}) is not contained "
                                 f"in clique {c.clique} = {self.cliques[c.clique]}")
        self.result: Optional[Dict] = None
        self._blocks: List[Tuple[str, int, int]] = []  # (kind, clique, size)

    # ------------------------------------------------------------------ helpers
    def _poly_terms(self, g: MultivariatePolynomial) -> List[Tuple[np.ndarray, float]]:
        ix = self.indexer
        return [(ix.word_of_exponent(a), c) for a, c in g.coeffs.items()]

    def _y_of_poly(self, y: cp.Variable, f: MultivariatePolynomial) -> cp.Expression:
        ix = self.indexer
        keys = np.array([ix.monomial_key(a) for a in f.coeffs], dtype=np.int64)
        idx = ix.index_of_keys(keys)
        return np.array(list(f.coeffs.values())) @ y[idx]

    # -------------------------------------------------------------------- build
    def build(self) -> Tuple[cp.Problem, cp.Variable]:
        ix, d = self.indexer, self.d
        y = cp.Variable(ix.n_moments, name="y")
        cons: List = [y[int(ix.index_of_keys(np.zeros(1, dtype=np.int64))[0])] == 1]
        self._blocks = []
        eq_rows: List[sp.spmatrix] = []
        for k in range(len(self.cliques)):
            cons.append(cp.PSD(y[ix.pair_index(k, d)]))
            self._blocks.append(("moment", k, len(ix.local_monomials(k, d))))
        for c in self.constraints:
            deg = c.poly.degree
            terms = self._poly_terms(c.poly)
            if c.kind == "ineq":
                dloc = d - (deg + 1) // 2
                if dloc < 0:
                    raise ValueError(f"constraint {c.name!r} has degree too high for order {d}")
                expr = 0
                for w, coef in terms:
                    expr = expr + coef * y[ix.pair_index(c.clique, dloc, w)]
                cons.append(cp.PSD(expr))
                self._blocks.append(("localizing", c.clique, len(ix.local_monomials(c.clique, dloc))))
            else:
                bdeg = 2 * d - deg
                W = ix.local_words(c.clique, bdeg, full=True)
                rows, cols, vals = [], [], []
                for w, coef in terms:
                    Wt = np.concatenate([W, np.broadcast_to(w[None, :], (W.shape[0], len(w)))], axis=1)
                    j = ix.index_of_keys(ix.keys_of_words(Wt))
                    rows.append(np.arange(W.shape[0]))
                    cols.append(j)
                    vals.append(np.full(W.shape[0], coef))
                A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(W.shape[0], ix.n_moments)).tocsr()
                eq_rows.append(A)
        if eq_rows:
            A = _independent_rows(sp.vstack(eq_rows).tocsr())
            self.n_equalities = A.shape[0]
            cons.append(A @ y == 0)
        obj = self._y_of_poly(y, self.f)
        if self.kappa_radius > 0 and self.robust_terms:
            keys = np.array([ix.monomial_key(list(a) + [0] * (self.n - len(a))) for a, _ in self.robust_terms])
            wts = np.array([w for _, w in self.robust_terms])
            obj = obj + self.kappa_radius * cp.norm(cp.multiply(wts, y[ix.index_of_keys(keys)]), 2)
        return cp.Problem(cp.Minimize(obj), cons), y

    # -------------------------------------------------------------------- solve
    def solve(self, solver: str = "CLARABEL", verbose: bool = False, **solver_opts) -> Dict:
        """Solve; returns a dict with `lower_bound` (None if the solver failed)."""
        t0 = time.perf_counter()
        problem, y = self.build()
        t_build = time.perf_counter() - t0
        sid = SOLVERS.get(solver.upper())
        if sid is None:
            raise ValueError(f"unknown solver {solver!r}")
        opts = dict(solver_opts)
        if sid == cp.CLARABEL:
            for k in ("tol_gap_abs", "tol_gap_rel", "tol_feas"):
                opts.setdefault(k, 1e-7)
        if sid == cp.SCS:
            opts.setdefault("eps", 1e-6)
            opts.setdefault("max_iters", 100_000)
        t1 = time.perf_counter()
        try:
            problem.solve(solver=sid, verbose=verbose, **opts)
            status = problem.status
        except cp.error.SolverError as exc:
            status = f"solver_error: {exc}"
        t_solve = time.perf_counter() - t1
        ok = problem.value is not None and np.isfinite(problem.value) and y.value is not None
        stats = problem.solver_stats
        self.result = {
            "lower_bound": float(problem.value) if ok else None,
            "status": status,
            "inaccurate": "inaccurate" in status,
            "solver": solver.upper(),
            "iterations": getattr(stats, "num_iters", None) if stats is not None else None,
            "build_time": t_build,
            "solve_time": t_solve,
            "moments": np.asarray(y.value) if ok else None,
            "sizes": self.sizes(),
        }
        return self.result

    def moment_matrix(self, k: int, order: Optional[int] = None,
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        y = self.result["moments"] if y is None else y
        return y[self.indexer.pair_index(k, self.d if order is None else order)]

    def sizes(self) -> Dict[str, int]:
        """Complexity metrics: largest PSD block, total PSD entries, unique moments."""
        s = [b[2] for b in self._blocks] or [0]
        return {
            "max_block": int(max(s)),
            "psd_entries": int(sum(v * v for v in s)),
            "unique_moments": int(self.indexer.n_moments),
            "n_psd_blocks": len(self._blocks),
            "max_moment_block": int(max([b[2] for b in self._blocks if b[0] == "moment"] or [0])),
            "moment_block_entries": int(sum(b[2] ** 2 for b in self._blocks if b[0] == "moment")),
        }


# =====================================================================================
# Portfolio problem
# =====================================================================================

@dataclass
class PortfolioRelaxation:
    relaxation: MomentRelaxation
    n: int              # number of asset variables (ids 0..n-1)
    n_total: int        # including auxiliary variables
    structure: str
    x_cliques: List[List[int]]
    owners: List[List[int]]

    def solve(self, *a, **kw) -> Dict:
        return self.relaxation.solve(*a, **kw)

    @property
    def cliques(self) -> List[List[int]]:
        return self.relaxation.cliques


def _owners(n: int, x_cliques: Sequence[Sequence[int]]) -> List[List[int]]:
    own: List[List[int]] = [[] for _ in x_cliques]
    seen = set()
    for k, c in enumerate(x_cliques):
        for i in sorted(c):
            if i not in seen:
                seen.add(i)
                own[k].append(i)
    if len(seen) != n:
        raise ValueError("every variable must belong to at least one clique")
    return own


def _bases(cliques: Sequence[Sequence[int]], cons: Sequence[Constraint]) -> List[List[int]]:
    """Per-clique PSD basis: drop one variable (the largest id) per linear equality.

    An equality  sum_j a_j z_j + c = 0  lets z_p be written through the others, so
    monomials containing z_p are redundant rows/columns of the moment matrix.
    """
    basis = [set(c) for c in cliques]
    for c in cons:
        if c.kind != "eq":
            continue
        live = [v for v in c.poly.support() if v in basis[c.clique]]
        if live:
            basis[c.clique].discard(max(live))
    return [sorted(b) for b in basis]


def build_portfolio_relaxation(
    f: MultivariatePolynomial,
    x_cliques: Optional[Sequence[Sequence[int]]] = None,
    b_lo: float = 0.95,
    b_hi: float = 1.0,
    order: int = 2,
    structure: str = "sparse",
    budget: str = "star",
    ball: bool = False,
    upper_bounds: bool = True,
    robust_terms: Optional[Sequence[Tuple[Sequence[int], float]]] = None,
    kappa_radius: float = 0.0,
    delta_robust: Optional[float] = None,
) -> PortfolioRelaxation:
    """Moment relaxation of min f(x) over {x >= 0, b_lo <= sum x <= b_hi}.

    structure
      "dense"   one clique on x, budget imposed directly (standard Lasserre).
      "sparse"  cliques J_k = I_k ∪ {s_k} plus budget clique(s); needs x_cliques
                (in RIP order, e.g. `ChordalExtension.maximal_cliques`).
    Both use the same constraints on x: x_i >= 0, 1 - x_i >= 0 (if upper_bounds),
    budget, and optionally per-clique balls |I_k| - sum_{I_k} x_i^2 >= 0 (`ball`;
    implied by the box constraints under the Archimedean property, so off by default).
    Every sparse constraint is implied by the dense ones at the same order, hence
    lb_sparse <= lb_dense when ball=False.
    """
    if delta_robust is not None:  # alias kept for v1 callers
        kappa_radius = delta_robust
    n = f.n_vars
    eq_budget = abs(b_hi - b_lo) < 1e-12
    cons: List[Constraint] = []
    NT = n

    def box(i: int, k: int, hi: float = 1.0) -> None:
        cons.append(Constraint(linear_poly(NT, {i: 1.0}), k, "ineq", f"z{i}>=0"))
        if upper_bounds:
            cons.append(Constraint(linear_poly(NT, {i: -1.0}, hi), k, "ineq", f"z{i}<={hi}"))

    def ball_c(members: Sequence[int], k: int) -> None:
        if ball:
            d = {}
            for j in members:
                a = [0] * NT
                a[j] = 2
                d[tuple(a)] = -1.0
            d[(0,) * NT] = float(len(members))
            cons.append(Constraint(MultivariatePolynomial(NT, d), k, "ineq", "ball"))

    def budget_c(members: Dict[int, float], k: int) -> None:
        p = linear_poly(NT, members)
        if eq_budget:
            cons.append(Constraint(p + linear_poly(NT, {}, -b_lo), k, "eq", "budget"))
        else:
            cons.append(Constraint(p + linear_poly(NT, {}, -b_lo), k, "ineq", "budget_lo"))
            cons.append(Constraint(linear_poly(NT, {i: -v for i, v in members.items()}, b_hi), k, "ineq", "budget_hi"))

    if structure == "dense":
        cl = [list(range(n))]
        for i in range(n):
            box(i, 0)
        budget_c({i: 1.0 for i in range(n)}, 0)
        ball_c(range(n), 0)
        rel = MomentRelaxation(NT, f, cl, cons, order, robust_terms, kappa_radius, _bases(cl, cons))
        return PortfolioRelaxation(rel, n, NT, structure, cl, [list(range(n))])

    if structure != "sparse":
        raise ValueError(f"unknown structure {structure!r}")
    if x_cliques is None:
        raise ValueError("x_cliques required for structure 'sparse'")
    x_cliques = [sorted(c) for c in x_cliques]
    own = _owners(n, x_cliques)
    owning = [k for k, o in enumerate(own) if o]
    K = len(owning)
    s_id = {k: n + m for m, k in enumerate(owning)}
    chain = budget == "chain" and K > 2
    NT = n + K + (K if chain else 0)
    p_id = {k: n + K + m for m, k in enumerate(owning)} if chain else {}

    J = [sorted(set(c) | ({s_id[k]} if k in s_id else set())) for k, c in enumerate(x_cliques)]
    extra: List[List[int]] = []
    if K >= 2 and not chain:
        extra = [[s_id[k] for k in owning]]
    elif chain:
        prev = None
        for k in owning:
            extra.append(sorted({s_id[k], p_id[k]} | ({p_id[prev]} if prev is not None else set())))
            prev = k
    cl = J + extra
    seen_c = set()

    def once(key, fn):
        if key not in seen_c:
            seen_c.add(key)
            fn()

    for k in range(len(x_cliques)):
        for i in x_cliques[k]:
            once(("box", i, k), lambda i=i, k=k: box(i, k))
        if k in s_id:
            s = s_id[k]
            once(("box", s, k), lambda s=s, k=k: box(s, k, b_hi))
            cons.append(Constraint(linear_poly(NT, {s: 1.0, **{i: -1.0 for i in own[k]}}), k, "eq", f"sector{k}"))
        ball_c(x_cliques[k], k)

    if K == 1:
        k0 = owning[0]
        budget_c({s_id[k0]: 1.0}, k0)
    elif not chain:
        c = len(J)
        for k in owning:
            once(("box", s_id[k], c), lambda k=k, c=c: box(s_id[k], c, b_hi))
        budget_c({s_id[k]: 1.0 for k in owning}, c)
    else:
        prev = None
        for m, k in enumerate(owning):
            c = len(J) + m
            s, p = s_id[k], p_id[k]
            once(("box", s, c), lambda s=s, c=c: box(s, c, b_hi))
            once(("box", p, c), lambda p=p, c=c: box(p, c, b_hi))
            coef = {p: 1.0, s: -1.0}
            if prev is not None:
                coef[p_id[prev]] = -1.0
            cons.append(Constraint(linear_poly(NT, coef), c, "eq", f"chain{m}"))
            prev = k
        budget_c({p_id[owning[-1]]: 1.0}, len(cl) - 1)

    rel = MomentRelaxation(NT, f.embed(NT), cl, cons, order, robust_terms, kappa_radius, _bases(cl, cons))
    return PortfolioRelaxation(rel, n, NT, structure, x_cliques, own)


def dense_sizes(n: int, d: int = 2, upper_bounds: bool = True, eq_budget: bool = False) -> Dict[str, int]:
    """Closed-form complexity metrics of the dense relaxation (same keys as `MomentRelaxation.sizes`).

    Valid for d >= 1 with degree-1 constraints; used to report sizes that are too large to build.
    """
    from math import comb
    nb = n - 1 if eq_budget else n
    m = comb(nb + d, d)
    loc = comb(nb + d - 1, d - 1)
    n_loc = n * (2 if upper_bounds else 1) + (0 if eq_budget else 2)
    return {"max_block": max(m, loc), "psd_entries": m * m + n_loc * loc * loc,
            "unique_moments": comb(n + 2 * d, 2 * d), "n_psd_blocks": 1 + n_loc,
            "max_moment_block": m, "moment_block_entries": m * m}
