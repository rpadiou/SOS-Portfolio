"""Flat-extension test, minimizer extraction and a posteriori certificates.

Flat extension (Curto-Fialkow): rank M_d(y) = rank M_{d-d_K}(y), d_K = max_j ceil(deg g_j / 2),
per clique. Extraction follows Henrion-Lasserre (2005): with W a set of r monomials
of degree <= d-d_K spanning the range of M_{d-d_K}, G = M[W, W] = L L^T and
H_i[a, b] = y_{w_a + w_b + e_i}, the matrices N_i = L^{-1} H_i L^{-T} commute and
share an orthonormal eigenbasis Q (real Schur form of a random combination); the
i-th coordinate of the j-th atom is q_j^T N_i q_j.

A flat extension is only a *secondary* certificate. The primary one is a posteriori:
a feasible point x_hat with f(x_hat) - lambda_d <= eps (`certify`). It holds to solver accuracy:
lambda_d is the primal objective of the moment problem, not the dual objective.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla

from .local_solver import polish
from .polynomial_ring import MultivariatePolynomial
from .sos_hierarchy import MomentRelaxation


def numerical_ranks(M: np.ndarray, rel_tol: float = 1e-5, gap_min: float = 1e3) -> Dict:
    """Rank by relative threshold and by largest spectral gap."""
    sv = np.linalg.svd(0.5 * (M + M.T), compute_uv=False)
    if sv[0] <= 0:
        return {"rank_thr": 0, "rank_gap": 0, "gap_ratio": 0.0, "sv": sv}
    rank_thr = int(np.sum(sv > rel_tol * sv[0]))
    ratios = sv[:-1] / np.maximum(sv[1:], 1e-300)
    j = int(np.argmax(ratios)) if len(ratios) else 0
    rank_gap = j + 1 if len(ratios) and ratios[j] >= gap_min else len(sv)
    return {"rank_thr": rank_thr, "rank_gap": rank_gap,
            "gap_ratio": float(ratios[j]) if len(ratios) else 0.0, "sv": sv}


def extraction_degree(relaxation: MomentRelaxation, k: int) -> int:
    """d_K for clique k: max ceil(deg g / 2) over the constraints attached to it (>= 1)."""
    degs = [(c.poly.degree + 1) // 2 for c in relaxation.constraints if c.clique == k]
    return max(degs + [1])


class MinimizerExtractor:
    """Flatness and extraction on the asset cliques of a solved relaxation.

    Parameters
    ----------
    relaxation : solved `MomentRelaxation`
    y : moment vector
    n_x : number of asset variables (ids 0..n_x-1); cliques whose basis contains
        other variables (budget cliques) are skipped
    rank_tol : relative singular-value threshold. It must exceed the solver accuracy:
        with CLARABEL (1e-7) 1e-5 is used; with SCS do not trust rank decisions.
    """

    def __init__(self, relaxation: MomentRelaxation, y: np.ndarray, n_x: int,
                 rank_tol: float = 1e-5) -> None:
        self.rel, self.y, self.n_x, self.rank_tol = relaxation, np.asarray(y), n_x, rank_tol
        self.ix = relaxation.indexer
        self.d = relaxation.d
        self.asset_cliques = [k for k, b in enumerate(self.ix.basis) if b and max(b) < n_x]

    def flatness(self) -> Dict:
        per = []
        for k in self.asset_cliques:
            dK = extraction_degree(self.rel, k)
            Md = self.rel.moment_matrix(k, y=self.y)
            Mp = self.rel.moment_matrix(k, self.d - dK, y=self.y)
            a, b = numerical_ranks(Md, self.rank_tol), numerical_ranks(Mp, self.rank_tol)
            per.append({"k": k, "d_K": dK, "rank_d": a["rank_thr"], "rank_d_minus_dK": b["rank_thr"],
                        "rank_gap_d": a["rank_gap"], "rank_gap_d_minus_dK": b["rank_gap"],
                        "flat": a["rank_thr"] == b["rank_thr"],
                        "flat_gap": a["rank_gap"] == b["rank_gap"],
                        "sv_d": a["sv"], "sv_prev": b["sv"]})
        return {"flat": all(p["flat"] for p in per), "per_clique": per}

    # atoms
    def clique_atoms(self, k: int, seed: int = 0) -> Optional[Dict]:
        """Atoms (r, |basis_k|) and weights of clique k, or None if not extractable."""
        ix, d = self.ix, self.d
        dK = extraction_degree(self.rel, k)
        dp = d - dK
        Mp = self.rel.moment_matrix(k, dp, y=self.y)
        r = numerical_ranks(self.rel.moment_matrix(k, y=self.y), self.rank_tol)["rank_thr"]
        r = min(r, Mp.shape[0])
        if r == 0:
            return None
        _, _, P = sla.qr(Mp, pivoting=True)
        W = np.sort(P[:r])
        G = Mp[np.ix_(W, W)]
        G = 0.5 * (G + G.T)
        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            return None
        Ww = ix.local_words(k, dp)[W]
        basis = ix.basis[k]
        Linv = sla.solve_triangular(L, np.eye(r), lower=True)
        Ns = []
        for v in basis:
            parts = [np.broadcast_to(Ww[:, None, :], (r, r, Ww.shape[1])),
                     np.broadcast_to(Ww[None, :, :], (r, r, Ww.shape[1])),
                     np.full((r, r, 1), v + 1, dtype=np.int64)]
            H = self.y[ix.index_of_keys(ix.keys_of_words(np.concatenate(parts, axis=-1)))]
            Ns.append(Linv @ H @ Linv.T)
        theta = np.random.RandomState(seed).randn(len(Ns))
        N = sum(t * Ni for t, Ni in zip(theta, Ns))
        _, Q = sla.schur(0.5 * (N + N.T), output="real")
        atoms = np.array([[Q[:, j] @ Ni @ Q[:, j] for Ni in Ns] for j in range(r)])
        # weights: sum_j w_j m_a(atom_j) = y_{m_a} for the monomials m_a in W
        expo = ix.local_monomials(k, dp)
        Vm = np.array([[np.prod(atoms[j] ** np.array(expo[a])) for j in range(r)] for a in W])
        yW = self.y[ix.index_of_keys(ix.keys_of_words(Ww))]
        w = np.linalg.lstsq(Vm, yW, rcond=None)[0]
        return {"k": k, "vars": basis, "atoms": atoms, "weights": w, "rank": r}

    def extract(self, tol: float = 1e-4, max_points: int = 64) -> Dict:
        """Merge clique atoms by agreement on shared variables.

        Returns {"points": (m, n_x) array or None, "reason": str}. Points are
        candidates; callers must check feasibility and objective value.
        Cliques that share no variable (e.g. sectors) match every atom combination, so the
        number of partial points is the product of the atom counts; beyond `max_points` the
        extraction is declared failed (the moments are then not flat in practice).
        """
        if not self.asset_cliques:
            return {"points": None, "reason": "no asset clique"}
        partial: List[Dict[int, float]] = [{}]
        for k in self.asset_cliques:
            ca = self.clique_atoms(k)
            if ca is None:
                return {"points": None, "reason": f"clique {k}: extraction failed"}
            new: List[Dict[int, float]] = []
            matched_atoms = np.zeros(len(ca["atoms"]), dtype=bool)
            for p in partial:
                for j, a in enumerate(ca["atoms"]):
                    if all(abs(p[v] - a[i]) <= tol for i, v in enumerate(ca["vars"]) if v in p):
                        q = dict(p)
                        q.update({v: float(a[i]) for i, v in enumerate(ca["vars"])})
                        new.append(q)
                        matched_atoms[j] = True
            if not new or (partial != [{}] and not matched_atoms.all()):
                return {"points": None, "reason": f"clique {k}: atoms inconsistent on shared variables"}
            partial = new
            if len(partial) > max_points:
                return {"points": None, "reason": f"clique {k}: more than {max_points} atom combinations"}
        pts = [np.array([p.get(i, np.nan) for i in range(self.n_x)]) for p in partial]
        pts = [x for x in pts if not np.isnan(x).any()]
        if not pts:
            return {"points": None, "reason": "variables not covered by any asset clique"}
        uniq: List[np.ndarray] = []
        for x in pts:
            if not any(np.max(np.abs(x - u)) <= tol for u in uniq):
                uniq.append(x)
        return {"points": np.array(uniq), "reason": "ok"}

    def mean_point(self) -> np.ndarray:
        """First-order moments (y_{e_i}); equals the minimizer when the measure is a Dirac."""
        return np.array([self.y[self.ix.index(np.eye(self.ix.n, dtype=int)[i])] for i in range(self.n_x)])


def certify(f: MultivariatePolynomial, lower_bound: float, candidates: Sequence[np.ndarray],
            b_lo: float, b_hi: float, eps_abs: float = 1e-6, eps_rel: float = 1e-6,
            feas_tol: float = 1e-8, polish_candidates: bool = True) -> Dict:
    """A posteriori global-optimality certificate: feasible x_hat with f(x_hat) - lb <= eps.

    `lower_bound` is the primal objective returned by the solver, so the certificate holds to
    solver accuracy (the value can exceed the true optimum by about the solver residuals).

    Candidates are first polished by a local solve (extraction from a solver-accuracy
    moment matrix is only accurate to ~1e-5); feasibility is checked on the final point.
    """
    best, fb = None, np.inf
    for x in candidates:
        x = np.asarray(x, dtype=float)
        if polish_candidates:
            x = polish(f, x, b_lo, b_hi, feas_tol)
        s = x.sum()
        if x.min() >= -feas_tol and b_lo - feas_tol <= s <= b_hi + feas_tol:
            v = float(f(x))
            if v < fb:
                best, fb = x, v
    if best is None:
        return {"certified": False, "x_hat": None, "f_hat": None, "gap": None, "reason": "no feasible candidate"}
    gap = fb - lower_bound
    eps = max(eps_abs, eps_rel * abs(lower_bound))
    return {"certified": bool(gap <= eps), "x_hat": best, "f_hat": fb, "gap": float(gap), "eps": eps,
            "reason": "ok"}
