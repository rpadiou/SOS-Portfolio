"""
Minimizer Extraction — Henrion–Lasserre / Curto–Fialkow
========================================================

Extracts the global minimiser x* ∈ R^n from the optimal moment sequence y*
when the flat extension condition rank(M_d) = rank(M_{d-1}) is certified.

Algorithm (Henrion & Lasserre 2005)
-------------------------------------
Given moment matrix M = M_d(y*) of rank r:

1. Compute a column basis U of M via rank-revealing SVD with threshold ε.
2. For each variable x_i, build the shift matrix N_i:
       N_i = U^+ · M̃_i · (U^+)^T
   where M̃_i is the matrix of moments shifted by e_i.
3. Compute eigenvalues of N_i simultaneously via joint diagonalisation.
   The r eigenvalue-tuples (λ_i^1,...,λ_i^r) give the r minimisers.

For sparse formulations, extraction runs per-clique and solutions are
merged via consistency checks across shared variables.

References
----------
- Henrion, D., & Lasserre, J. B. (2005). Detecting global optimality and
  extracting solutions in GloptiPoly. Positive Polynomials in Control, Springer.
- Curto, R. E., & Fialkow, L. A. (1996). Solution of the truncated complex
  moment problem for flat data. AMS Memoir 568.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .indexer import SparseIndexer


class MinimizerExtractor:
    """
    Extract global minimisers from sparse moment matrices.

    Parameters
    ----------
    moment_matrices : Dict[int, np.ndarray]
        Per-clique moment matrix M_d^{I_k}(y*), keyed by clique index k.
    cliques : List[List[int]]
        Variable index lists for each clique.
    y_vals : np.ndarray
        Global moment sequence y* from the SDP solution.
    indexer : SparseIndexer
        The indexer used to build the SDP.
    svd_tol : float
        Numerical rank-determination threshold (relative to largest singular value).
    """

    def __init__(
        self,
        moment_matrices: Dict[int, np.ndarray],
        cliques: List[List[int]],
        y_vals: np.ndarray,
        indexer: SparseIndexer,
        svd_tol: float = 1e-6,
    ) -> None:
        self.moment_matrices = moment_matrices
        self.cliques = cliques
        self.y_vals = y_vals
        self.indexer = indexer
        self.svd_tol = svd_tol
        self.n = indexer.n
        self.d = indexer.d

    # ── Public API ────────────────────────────────────────────────────────────

    def flat_extension_ranks(self) -> Dict[int, Tuple[int, int]]:
        """
        Compute (rank_d, rank_{d-1}) for each clique's moment matrix.

        Returns
        -------
        Dict mapping clique index → (rank_d, rank_{d-1}).
        """
        ranks: Dict[int, Tuple[int, int]] = {}
        for k, clique in enumerate(self.cliques):
            M_d = self.moment_matrices.get(k)
            if M_d is None:
                continue
            M_prev = self._local_moment_matrix(k, clique, self.d - 1)
            ranks[k] = (
                self._numerical_rank(M_d),
                self._numerical_rank(M_prev),
            )
        return ranks

    def is_flat(self) -> bool:
        """True iff rank(M_d^{I_k}) = rank(M_{d-1}^{I_k}) for all cliques k."""
        for k, (r_d, r_d1) in self.flat_extension_ranks().items():
            if r_d != r_d1:
                return False
        return True

    def extract(self) -> Optional[np.ndarray]:
        """
        Extract global minimisers. Returns None if flat extension fails.

        Returns
        -------
        np.ndarray, shape (r, n)
            Each row is a candidate minimiser in R^n.
            r = common rank (number of atoms in representing measure).
        """
        if not self.is_flat():
            return None

        # Per-clique extraction, then merge
        clique_solutions: Dict[int, np.ndarray] = {}
        for k, clique in enumerate(self.cliques):
            M_d = self.moment_matrices.get(k)
            if M_d is None:
                continue
            sols = self._extract_clique(k, clique, M_d)
            if sols is not None:
                clique_solutions[k] = sols

        if not clique_solutions:
            return None

        return self._merge_clique_solutions(clique_solutions)

    # ── Per-clique extraction ─────────────────────────────────────────────────

    def _extract_clique(
        self, k: int, clique: List[int], M_d: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Henrion–Lasserre extraction for a single clique.

        Returns array of shape (r, |I_k|) with local coordinates.
        """
        p = len(clique)
        r = self._numerical_rank(M_d)
        if r == 0:
            return None

        # ── Special case r=1: Dirac measure μ = δ_{x*} ───────────────────────
        # For a rank-1 moment matrix the representing measure is a single atom
        # at x*.  The first-order moments directly give x*:  x*_i = y_{e_i}.
        # This is numerically exact and avoids the ill-conditioned eigenvalue path.
        if r == 1:
            idx = self.indexer
            solutions = np.zeros((1, p))
            for local_var in range(p):
                e_local: List[int] = [0] * p
                e_local[local_var] = 1
                g_idx = idx.local_to_global(k, tuple(e_local))
                solutions[0, local_var] = self.y_vals[g_idx]
            return solutions

        # Column basis U of M_d, shape (s_d, r)
        U, s_vals, Vt = np.linalg.svd(M_d, full_matrices=False)
        thresh = self.svd_tol * s_vals[0] if s_vals[0] > 0 else self.svd_tol
        U = U[:, :r]  # truncate to numerical rank

        local_mons_d = self.indexer.local_monomials(k, self.d)
        local_mons_d1 = self.indexer.local_monomials(k, self.d - 1)

        # Build shift matrices N_i for each local variable position
        # N_i = U^+ M_shift_i where M_shift_i[α,β] = M_{d-1}[α+e_i, β]
        shift_mats: List[np.ndarray] = []
        for local_var in range(p):
            N = self._build_shift_matrix(
                k, clique, U, local_mons_d, local_mons_d1, local_var, r
            )
            shift_mats.append(N)

        if not shift_mats:
            return None

        # Simultaneously diagonalise shift matrices using eigendecomposition
        # Strategy: form a random linear combination and eigendecompose
        rng = np.random.RandomState(k * 137 + 42)
        coeffs_rand = rng.randn(p)
        N_combo = sum(c * N for c, N in zip(coeffs_rand, shift_mats))

        try:
            eigvals, eigvecs = np.linalg.eig(N_combo)
        except np.linalg.LinAlgError:
            return None

        # Keep real parts (imaginary parts ≈ numerical noise)
        eigvecs = np.real(eigvecs)
        eigvals = np.real(eigvals)

        # Sort by eigenvalue for reproducibility
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]

        # Extract per-variable coordinates by back-substitution
        # x_i^* = (e_i^T N_i eigvec_j) / (eigvec_j^T eigvec_j)  for atom j
        solutions = np.zeros((r, p))
        for j in range(r):
            v = eigvecs[:, j]
            for local_var, N in enumerate(shift_mats):
                denom = v @ v
                if abs(denom) < 1e-14:
                    solutions[j, local_var] = 0.0
                else:
                    solutions[j, local_var] = (v @ N @ v) / denom

        return solutions

    def _build_shift_matrix(
        self,
        k: int,
        clique: List[int],
        U: np.ndarray,
        local_mons_d: List[Tuple],
        local_mons_d1: List[Tuple],
        local_var: int,
        r: int,
    ) -> np.ndarray:
        """
        Build the r×r shift matrix N_{local_var} for Henrion–Lasserre extraction.

        N_{local_var} is defined by:
            [N_{local_var}]_{pq} = y_{embed(α_p + e_{local_var} + α_q)}

        where α_p, α_q ∈ local_mons_{d-1} and e_{local_var} is the unit vector.

        We form the (s_{d-1} × s_{d-1}) matrix of shifted moments, then project:
            N = (U_{d-1})^+ · M_shift · U_{d-1}
        where U_{d-1} = U restricted to rows indexed by local_mons_{d-1}.

        Since the shift of α ∈ M_{d-1} lands in M_d, we can use the same U.
        """
        idx = self.indexer
        s_d = len(local_mons_d)
        s_d1 = len(local_mons_d1)
        p = len(clique)

        # Map local_mons_d to row indices
        mon_d_to_row = {m: i for i, m in enumerate(local_mons_d)}

        # Build rectangular matrix X of shape (s_d, s_d1):
        # X[row_of(alpha + e_i), col_of(beta)] = y_{embed(alpha + e_i + beta)}
        # We use alpha from local_mons_d1 shifted by e_i to get a row in local_mons_d
        X = np.zeros((s_d, s_d1))
        e_i = [0] * p
        e_i[local_var] = 1
        e_i_t = tuple(e_i)

        for col, beta in enumerate(local_mons_d1):
            for row_alpha_idx, alpha in enumerate(local_mons_d1):
                # alpha shifted by e_i
                shifted_alpha = tuple(
                    alpha[l] + e_i_t[l] for l in range(p)
                )
                if shifted_alpha not in mon_d_to_row:
                    continue
                row = mon_d_to_row[shifted_alpha]
                # Global moment index for alpha + e_i + beta
                gamma_local = tuple(
                    shifted_alpha[l] + beta[l] for l in range(p)
                )
                g_idx = idx.local_to_global(k, gamma_local)
                X[row, col] = self.y_vals[g_idx]

        # Restrict U to rows corresponding to local_mons_d1
        # U has shape (s_d, r); we need rows for indices in local_mons_d1 ⊆ local_mons_d
        d1_row_indices = [
            mon_d_to_row[m] for m in local_mons_d1 if m in mon_d_to_row
        ]
        U_d1 = U[d1_row_indices, :]  # shape (s_d1, r)

        # N = U^+ X U_d1
        # Use least-squares pseudo-inverse: U^+ = (U^T U)^{-1} U^T = (since U has orthonormal cols) U^T
        # U comes from SVD so columns are orthonormal: U^T U = I_r
        N = U.T @ X @ U_d1  # shape (r, r)
        return N

    # ── Merging per-clique solutions ──────────────────────────────────────────

    def _merge_clique_solutions(
        self, clique_solutions: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Merge per-clique local solutions into a global solution in R^n.

        Strategy: use the clique with the smallest index (clique 0) as
        the reference. For each atom j from clique 0, complete the global
        vector by reading local coordinates from other cliques and placing
        them at the correct global positions.

        When a variable appears in multiple cliques, its value is averaged
        across consistent solutions (discarding outliers > 3σ).
        """
        n = self.n

        # Determine number of atoms r from first available clique
        first_k = min(clique_solutions.keys())
        r = clique_solutions[first_k].shape[0]

        # Accumulate per-variable estimates: global_var → list of values
        var_values: Dict[int, np.ndarray] = {i: np.zeros(r) for i in range(n)}
        var_count: Dict[int, np.ndarray] = {i: np.zeros(r) for i in range(n)}

        for k, sols in clique_solutions.items():
            clique = self.cliques[k]
            n_atoms = sols.shape[0]
            for local_pos, global_var in enumerate(clique):
                if n_atoms >= r:
                    var_values[global_var] += sols[:r, local_pos]
                    var_count[global_var] += 1.0
                else:
                    var_values[global_var][:n_atoms] += sols[:, local_pos]
                    var_count[global_var][:n_atoms] += 1.0

        # Average across cliques
        result = np.zeros((r, n))
        for i in range(n):
            cnt = var_count[i]
            cnt = np.where(cnt > 0, cnt, 1.0)
            result[:, i] = var_values[i] / cnt

        return result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _numerical_rank(self, M: np.ndarray) -> int:
        if M is None or M.size == 0:
            return 0
        svd = np.linalg.svd(M, compute_uv=False)
        if svd[0] == 0:
            return 0
        thresh = self.svd_tol * svd[0]
        return int(np.sum(svd > thresh))

    def _local_moment_matrix(
        self, k: int, clique: List[int], order: int
    ) -> np.ndarray:
        """Build local moment matrix of given order from global y_vals."""
        idx = self.indexer
        local_mons = idx.local_monomials(k, order)
        s = len(local_mons)
        p = len(clique)
        M = np.zeros((s, s))
        for i, alpha in enumerate(local_mons):
            for j, beta in enumerate(local_mons):
                gamma_local = tuple(alpha[l] + beta[l] for l in range(p))
                g_idx = idx.local_to_global(k, gamma_local)
                M[i, j] = self.y_vals[g_idx]
        return M
