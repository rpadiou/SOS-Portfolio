"""
Sparse & Distributionally Robust Lasserre SOS Hierarchy — SOS-Portfolio
========================================================================

Implements the Waki–Kim–Kojima–Muramatsu sparse SOS relaxation at order d=2
with optional Distributionally Robust (DR) augmentation on the kurtosis tensor.

Dense vs Sparse Complexity
--------------------------
Dense (n assets, order d):
  Single PSD block: C(n+d,d) × C(n+d,d).
  n=50, d=2: 1326×1326 → ~1.76M scalar variables.

Sparse (p-clique, order d):
  Per-clique PSD block: C(p+d,d) × C(p+d,d).
  p=5 cliques, d=2: 21×21 → 441 per clique.
  10 cliques → 4,410 scalar variables. Reduction: ~400×.

Distributionally Robust Augmentation
-------------------------------------
Uncertainty set on kurtosis tensor κ̄:

    U(δ) = { κ : ‖κ - κ̄‖_F ≤ δ }

Worst-case kurtosis contribution given moment sequence y:

    max_{‖Δκ‖_F ≤ δ}  ⟨κ̄ + Δκ, y^{(4)}⟩  =  ⟨κ̄, y^{(4)}⟩ + δ · ‖y^{(4)}‖_2

where y^{(4)} = [y_α]_{|α|=4} is the vector of degree-4 moments. The DR
relaxation adds one second-order cone constraint to the moment SDP:

    min_{y, τ}  L_y(f_nominal) + δ · τ
    s.t.        ‖y^{(4)}‖_2 ≤ τ             (SOC)
                M_d^{I_k}(y) ≽ 0             (per-clique PSD)
                M_{d-1}^{I_k}(g_j · y) ≽ 0  (per-clique localizing)
                y_0 = 1                       (normalisation)

Reference: Waki et al. (2006) SIAM J. Optim. 17(1):218–242.
           Delage & Ye (2010) Management Science 56(9):1483–1495.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from .indexer import SparseIndexer
from .polynomial_ring import MultivariatePolynomial, Monomial, generate_monomials


class SparseLasserreRelaxation:
    """
    Sparse Lasserre SDP relaxation at order d for n-asset portfolios.

    Parameters
    ----------
    f : MultivariatePolynomial
        Degree-4 objective polynomial.
    constraints : List[MultivariatePolynomial]
        Constraint polynomials g_j with K = {g_j ≥ 0}.
    cliques : List[List[int]]
        Maximal cliques from chordal extension of the correlation graph.
    order : int
        Relaxation order d ≥ 2.
    delta_robust : float
        DR robustness parameter δ ≥ 0. 0 = nominal SOS, >0 = DR-SOS.
    """

    def __init__(
        self,
        f: MultivariatePolynomial,
        constraints: List[MultivariatePolynomial],
        cliques: List[List[int]],
        order: int = 2,
        delta_robust: float = 0.0,
    ) -> None:
        self.f = f
        self.constraints = constraints
        self.n = f.n_vars
        self.d = order
        self.cliques = cliques
        self.delta_robust = delta_robust

        self.indexer = SparseIndexer(n=self.n, d=order, cliques=cliques)
        self.result: Optional[Dict] = None

        # Degree-4 moment indices for DR cone
        self._deg4_indices: List[int] = [
            self.indexer.global_index(alpha)
            for alpha in self.indexer.all_global_moments()
            if sum(alpha) == 4
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def build_and_solve(
        self, solver: str = "SCS", verbose: bool = False
    ) -> Dict:
        """
        Construct and solve the sparse (DR-)SOS relaxation.

        Returns
        -------
        dict with keys:
          lower_bound, status, solve_time, moments, moment_matrices,
          n_scalar_vars, n_sdp_blocks, indexer
        """
        t0 = time.perf_counter()

        y, sdp_constraints = self._build_moment_constraints()

        # Objective: L_y(f) = Σ_α f_α y_α
        obj_expr = self._build_objective_expr(y)

        # DR augmentation: add δ · τ with ‖y^{(4)}‖_2 ≤ τ
        if self.delta_robust > 0.0 and self._deg4_indices:
            tau = cp.Variable(nonneg=True, name="tau_DR")
            y4 = cp.vstack([y[i] for i in self._deg4_indices])
            sdp_constraints.append(cp.norm(y4, 2) <= tau)
            obj_expr = obj_expr + self.delta_robust * tau

        problem = cp.Problem(cp.Minimize(obj_expr), sdp_constraints)

        solver_map = {
            "SCS": cp.SCS,
            "MOSEK": cp.MOSEK,
            "CLARABEL": cp.CLARABEL,
        }
        solver_id = solver_map.get(solver.upper(), cp.SCS)

        solver_kwargs: Dict = {"verbose": verbose}
        if solver_id == cp.SCS:
            solver_kwargs.update({"eps": 1e-4, "max_iters": 50_000})

        try:
            problem.solve(solver=solver_id, **solver_kwargs)
        except Exception as exc:
            return {
                "lower_bound": None,
                "status": f"Error: {exc}",
                "solve_time": time.perf_counter() - t0,
                "moments": None,
                "moment_matrices": {},
            }

        lb = (
            float(problem.value)
            if problem.value is not None and np.isfinite(problem.value)
            else None
        )

        # Extract per-clique moment matrix values
        moment_matrices: Dict[int, np.ndarray] = {}
        for k in range(len(self.cliques)):
            var = self._M_vars.get(k)
            if var is not None and var.value is not None:
                moment_matrices[k] = var.value

        self.result = {
            "lower_bound": lb,
            "status": problem.status,
            "solve_time": time.perf_counter() - t0,
            "moments": y.value,
            "moment_matrices": moment_matrices,
            "n_scalar_vars": y.size,
            "n_sdp_blocks": len(self.cliques),
            "indexer": self.indexer,
            "moment_matrix": moment_matrices.get(0),
        }
        return self.result

    def flat_extension_check(self, tol: float = 1e-6) -> Dict:
        """
        Per-clique Curto–Fialkow flat extension check.

        For each clique I_k, tests rank(M_d^{I_k}) = rank(M_{d-1}^{I_k}).
        Returns aggregate result (converged = True iff ALL cliques flat).
        """
        if self.result is None or not self.result["moment_matrices"]:
            return {"converged": False, "error": "No solution available"}
        if self.d < 2:
            return {"converged": False, "error": "Need d ≥ 2 for flat extension"}

        y_vals = self.result["moments"]
        if y_vals is None:
            return {"converged": False, "error": "Solver returned no moments"}

        per_clique: List[Dict] = []
        all_converged = True

        for k, clique in enumerate(self.cliques):
            M_d = self.result["moment_matrices"].get(k)
            if M_d is None:
                per_clique.append({"k": k, "converged": False, "error": "no matrix"})
                all_converged = False
                continue

            M_d_prev = self._build_local_moment_matrix_np(
                k, clique, y_vals, self.d - 1
            )

            svd_d = np.linalg.svd(M_d, compute_uv=False)
            svd_prev = np.linalg.svd(M_d_prev, compute_uv=False)

            thresh_d = tol * max(svd_d[0], 1e-12)
            thresh_prev = tol * max(svd_prev[0], 1e-12)

            rank_d = int(np.sum(svd_d > thresh_d))
            rank_prev = int(np.sum(svd_prev > thresh_prev))
            flat = rank_d == rank_prev

            per_clique.append({
                "k": k,
                "converged": flat,
                "rank_d": rank_d,
                "rank_d_minus_1": rank_prev,
                "singular_values_Md": svd_d,
                "singular_values_Md1": svd_prev,
            })
            if not flat:
                all_converged = False

        # Use first clique's singular values for legacy visualization interface
        first = per_clique[0] if per_clique else {}
        return {
            "converged": all_converged,
            "rank_d": first.get("rank_d"),
            "rank_d_minus_1": first.get("rank_d_minus_1"),
            "singular_values_Md": first.get("singular_values_Md"),
            "singular_values_Md1": first.get("singular_values_Md1"),
            "per_clique": per_clique,
        }

    # ── SDP construction ──────────────────────────────────────────────────────

    def _build_moment_constraints(
        self,
    ) -> Tuple[cp.Variable, List]:
        """
        Construct the global moment variable y and all SDP/linear constraints.

        Returns
        -------
        y : cp.Variable
            Global moment sequence, length = indexer.n_moments.
        constraints : List
            All CVXPY constraints.
        """
        idx = self.indexer
        N = idx.n_moments
        y = cp.Variable(N, name="y")
        constraints: List = []

        # Normalisation: y_0 = 1
        zero_alpha = tuple([0] * self.n)
        constraints.append(y[idx.global_index(zero_alpha)] == 1)

        self._M_vars: Dict[int, cp.Variable] = {}

        for k, clique in enumerate(self.cliques):
            p = len(clique)
            # ── Moment matrix M_d^{I_k}(y) ≽ 0 ──────────────────────────
            # Build the s_d×s_d index array (numpy, fast O(s_d²) Python ops)
            # then link M to y with ONE matrix equality instead of s_d² scalars.
            local_mons_d = idx.local_monomials(k, self.d)
            s_d = len(local_mons_d)
            M = cp.Variable((s_d, s_d), symmetric=True, name=f"M_{k}")
            self._M_vars[k] = M

            M_indices = np.empty((s_d, s_d), dtype=int)
            for i, alpha in enumerate(local_mons_d):
                for j, beta in enumerate(local_mons_d):
                    gamma_local = tuple(alpha[l] + beta[l] for l in range(p))
                    M_indices[i, j] = idx.local_to_global(k, gamma_local)

            # Single matrix equality: O(1) CVXPY constraint objects
            constraints.append(M == y[M_indices])
            constraints.append(M >> 0)

            # ── Localizing matrices per constraint per clique ──────────────
            for g_idx_constraint, g in enumerate(self.constraints):
                v_g = int(np.ceil(g.degree / 2))
                d_loc = self.d - v_g
                if d_loc < 0:
                    continue

                # Determine which variables this constraint uses
                g_vars_used: set = set()
                for mono in g.coeffs:
                    for var_i, exp in enumerate(mono):
                        if exp > 0:
                            g_vars_used.add(var_i)

                clique_set = set(clique)
                if not g_vars_used.issubset(clique_set) and g_vars_used:
                    # Global constraint (e.g. budget): handled via linear moments
                    continue

                local_mons_loc = idx.local_monomials(k, d_loc)
                s_loc = len(local_mons_loc)
                if s_loc == 0:
                    continue

                L = cp.Variable((s_loc, s_loc), symmetric=True,
                                name=f"L_{k}_g{g_idx_constraint}")

                # Build localizing index array or coefficient map
                # For single-term constraints (e.g. g_i = x_i), use fast path.
                if len(g.coeffs) == 1:
                    (g_mono, c_g) = next(iter(g.coeffs.items()))
                    L_indices = np.empty((s_loc, s_loc), dtype=int)
                    valid = True
                    for i, alpha_l in enumerate(local_mons_loc):
                        for j, beta_l in enumerate(local_mons_loc):
                            g_global = self._embed_constraint_mono(
                                k, clique, alpha_l, beta_l, g_mono
                            )
                            if g_global is not None and idx.has_moment(g_global):
                                L_indices[i, j] = idx.global_index(g_global)
                            else:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        if abs(c_g - 1.0) < 1e-14:
                            constraints.append(L == y[L_indices])
                        else:
                            constraints.append(L == c_g * y[L_indices])
                    else:
                        # Fallback: scalar loop
                        for i, alpha_l in enumerate(local_mons_loc):
                            for j, beta_l in enumerate(local_mons_loc):
                                expr = 0
                                has_term = False
                                for gm, cg in g.coeffs.items():
                                    gg = self._embed_constraint_mono(
                                        k, clique, alpha_l, beta_l, gm)
                                    if gg is not None and idx.has_moment(gg):
                                        expr = expr + cg * y[idx.global_index(gg)]
                                        has_term = True
                                constraints.append(L[i, j] == (expr if has_term else 0))
                else:
                    # Multi-term constraint: fall back to scalar loop
                    for i, alpha_l in enumerate(local_mons_loc):
                        for j, beta_l in enumerate(local_mons_loc):
                            expr = 0
                            has_term = False
                            for g_mono, c_g in g.coeffs.items():
                                g_global = self._embed_constraint_mono(
                                    k, clique, alpha_l, beta_l, g_mono)
                                if g_global is not None and idx.has_moment(g_global):
                                    expr = expr + c_g * y[idx.global_index(g_global)]
                                    has_term = True
                            constraints.append(L[i, j] == (expr if has_term else 0))

                constraints.append(L >> 0)

        # ── Global budget constraints as linear moment inequalities ────────
        # g_lo: E_μ[Σ x_i] = Σ y_{e_i} ≥ budget_lower
        # g_hi: E_μ[Σ x_i] = Σ y_{e_i} ≤ 1
        y_sum_expr = 0
        for i in range(self.n):
            alpha = [0] * self.n
            alpha[i] = 1
            a_t = tuple(alpha)
            if idx.has_moment(a_t):
                y_sum_expr = y_sum_expr + y[idx.global_index(a_t)]

        # These encode E_μ[g_lo] ≥ 0 and E_μ[g_hi] ≥ 0 as scalar constraints
        # (valid because g_lo, g_hi ≥ 0 on K and μ is a probability measure)
        # Find budget_lower from constraints
        budget_lower = 0.95
        for g in self.constraints:
            if g.n_vars == self.n:
                zero_key = tuple([0] * self.n)
                if zero_key in g.coeffs and g.coeffs[zero_key] < 0:
                    budget_lower = -g.coeffs[zero_key]
                    break

        constraints.append(y_sum_expr >= budget_lower)
        constraints.append(y_sum_expr <= 1.0)

        return y, constraints

    def _embed_constraint_mono(
        self,
        k: int,
        clique: List[int],
        alpha_l: Tuple[int, ...],
        beta_l: Tuple[int, ...],
        g_mono: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        """
        Compute α + β + γ for the localizing matrix entry, fully embedded.

        alpha_l, beta_l are LOCAL multi-indices for clique k.
        g_mono is a GLOBAL multi-index (length = self.n).

        Returns the global multi-index or None if out of range.
        """
        n = self.n
        result = list(g_mono)  # start from global g coefficient multi-index
        for local_pos, global_var in enumerate(clique):
            result[global_var] += alpha_l[local_pos] + beta_l[local_pos]
        # Check degree bound
        if sum(result) > 2 * self.d + max(
            (g.degree for g in self.constraints), default=0
        ):
            return None
        return tuple(result)

    def _build_objective_expr(self, y: cp.Variable) -> cp.Expression:
        """Build L_y(f) = Σ_α f_α y_α as a CVXPY expression."""
        idx = self.indexer
        expr: cp.Expression = 0
        for alpha, c_f in self.f.coeffs.items():
            if idx.has_moment(alpha):
                expr = expr + c_f * y[idx.global_index(alpha)]
        return expr

    def _build_local_moment_matrix_np(
        self,
        k: int,
        clique: List[int],
        y_vals: np.ndarray,
        order: int,
    ) -> np.ndarray:
        """Build the order-d' local moment matrix as a numpy array from y."""
        idx = self.indexer
        local_mons = idx.local_monomials(k, order)
        s = len(local_mons)
        M = np.zeros((s, s))
        for i, alpha in enumerate(local_mons):
            for j, beta in enumerate(local_mons):
                gamma_local = tuple(
                    alpha[l] + beta[l] for l in range(len(clique))
                )
                g_idx = idx.local_to_global(k, gamma_local)
                M[i, j] = y_vals[g_idx]
        return M


# ── Legacy dense interface (backward compatible with original main.py) ─────

class LasserreRelaxation:
    """
    Legacy dense Lasserre relaxation for the 2-asset toy model.

    Kept for backward compatibility with existing main.py and tests.
    For n > 2 assets use SparseLasserreRelaxation.
    """

    def __init__(
        self,
        f: MultivariatePolynomial,
        constraints: List[MultivariatePolynomial],
        order: int,
    ) -> None:
        self.f = f
        self.constraints = constraints
        self.n = f.n_vars
        self.d = order
        self.moment_monomials = generate_monomials(self.n, self.d)
        self.s = len(self.moment_monomials)
        self.result: Optional[Dict] = None

    def build_and_solve(self, solver: str = "SCS", verbose: bool = False) -> Dict:
        monomials_2d = generate_monomials(self.n, 2 * self.d)
        n_moments = len(monomials_2d)
        mono_to_idx = {m: i for i, m in enumerate(monomials_2d)}
        zero_idx = mono_to_idx[tuple([0] * self.n)]

        y = cp.Variable(n_moments)
        constraints: List = [y[zero_idx] == 1]

        M = cp.Variable((self.s, self.s), symmetric=True)
        for i, alpha in enumerate(self.moment_monomials):
            for j, beta in enumerate(self.moment_monomials):
                gamma = tuple(alpha[k] + beta[k] for k in range(self.n))
                if gamma in mono_to_idx:
                    constraints.append(M[i, j] == y[mono_to_idx[gamma]])
                else:
                    constraints.append(M[i, j] == 0)
        constraints.append(M >> 0)

        for g in self.constraints:
            v_g = int(np.ceil(g.degree / 2))
            d_loc = self.d - v_g
            if d_loc < 0:
                continue
            loc_monomials = generate_monomials(self.n, d_loc)
            s_loc = len(loc_monomials)
            L = cp.Variable((s_loc, s_loc), symmetric=True)

            for i, alpha in enumerate(loc_monomials):
                for j, beta in enumerate(loc_monomials):
                    expr = 0
                    has_term = False
                    for gamma, c_g in g.coeffs.items():
                        delta = tuple(
                            alpha[k] + beta[k] + gamma[k] for k in range(self.n)
                        )
                        if delta in mono_to_idx:
                            expr = expr + c_g * y[mono_to_idx[delta]]
                            has_term = True
                    constraints.append(L[i, j] == (expr if has_term else 0))
            constraints.append(L >> 0)

        obj_expr = sum(
            c_f * y[mono_to_idx[alpha]]
            for alpha, c_f in self.f.coeffs.items()
            if alpha in mono_to_idx
        )

        problem = cp.Problem(cp.Minimize(obj_expr), constraints)
        solver_map = {"SCS": cp.SCS, "MOSEK": cp.MOSEK, "CLARABEL": cp.CLARABEL}
        solver_id = solver_map.get(solver.upper(), cp.SCS)

        try:
            problem.solve(solver=solver_id, verbose=verbose)
        except Exception as e:
            return {"lower_bound": None, "status": f"Error: {e}",
                    "moments": None, "moment_matrix": None}

        lb = (
            float(problem.value)
            if problem.value is not None and np.isfinite(problem.value)
            else None
        )
        self.result = {
            "lower_bound": lb,
            "status": problem.status,
            "moments": y.value,
            "moment_matrix": M.value,
            "monomials_2d": monomials_2d,
            "moment_monomials": self.moment_monomials,
        }
        return self.result

    def flat_extension_check(self, tol: float = 1e-4) -> Dict:
        if self.result is None or self.result["moment_matrix"] is None:
            return {"converged": False, "error": "No solution available"}
        if self.d <= 1:
            return {"converged": False, "error": "Need d ≥ 2 for flat extension"}

        M_full = self.result["moment_matrix"]
        svd_full = np.linalg.svd(M_full, compute_uv=False)
        rank_d = int(np.sum(svd_full > tol * svd_full[0]))

        mono_prev = generate_monomials(self.n, self.d - 1)
        s_prev = len(mono_prev)
        monomials_2d = self.result["monomials_2d"]
        mono_to_idx = {m: i for i, m in enumerate(monomials_2d)}
        y = self.result["moments"]

        M_prev = np.zeros((s_prev, s_prev))
        for i, alpha in enumerate(mono_prev):
            for j, beta in enumerate(mono_prev):
                gamma = tuple(alpha[k] + beta[k] for k in range(self.n))
                if gamma in mono_to_idx:
                    M_prev[i, j] = y[mono_to_idx[gamma]]

        svd_prev = np.linalg.svd(M_prev, compute_uv=False)
        rank_prev = int(np.sum(svd_prev > tol * svd_prev[0]))

        return {
            "converged": rank_d == rank_prev,
            "rank_d": rank_d,
            "rank_d_minus_1": rank_prev,
            "singular_values_Md": svd_full,
            "singular_values_Md1": svd_prev,
        }
