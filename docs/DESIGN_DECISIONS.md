# Design decisions

Numbers are deliberately absent from this file; they live in `results/` and are
rendered into the README by `scripts/make_readme_tables.py`.

## One builder for dense and sparse

`MomentRelaxation` (sos_hierarchy.py) builds a clique-based moment relaxation from a
list of cliques and constraints attached to cliques. The dense relaxation is the
same code with one clique. Bound comparisons therefore compare the same constraint
set; there is no separate "reference" implementation that could differ silently.
All moment-matrix index arrays are produced by vectorised integer encoding of
monomials (indexer.py); there are no scalar CVXPY constraints in Python loops.

A constraint whose support is not contained in its clique raises `ValueError`.
v1.0 instead skipped such constraints and replaced the budget by a constraint on
`E[sum x]`, which decoupled the cliques (finding E3).

## Sparse bound against dense bound

At equal order and without the ball constraint, `lb_sparse <= lb_dense` for the star and chain budgets
(proof sketch in the docstring of `build_portfolio_relaxation`, numerical check in
`tests/regression/test_sparse_lifting.py`). The five instances of experiment A where the gap exceeds
1e-3 (`scripts/check_sparse_gap.py`, `results/sparse_gap.csv`) all have two cliques; for the three
4-asset ones the order-3 bounds agree, and the ball constraint changes nothing. This is measured,
not proved: the missing cross-clique moments at order 2 are the likely cause, and order 3 was not run
for the two 8-asset instances.

With overlapping cliques the budget clique closes a cycle and the running intersection property
fails; the builder warns. The relaxation stays valid.

## Why the budget needs auxiliary variables

`b_lo <= sum_i x_i <= b_hi` involves every variable, so it is supported in no
clique of a sparse decomposition. Options:

1. Merge all cliques: dense.
2. Drop it or impose it on moments of order 1 only: not a relaxation of the stated
   problem in a useful sense (E3).
3. Sector sums `s_k = sum_{i in O_k} x_i` (`O_k` = variables owned by clique `k`).
   The equalities live in `J_k = I_k ∪ {s_k}`, the budget in `S = {s_1..s_K}`.
   `J_k ∩ S = {s_k}`, so for disjoint cliques the star tree satisfies the running intersection property
   (with overlapping cliques it does not; see below).
   This is what is implemented (`budget="star"`).
4. For large `K`, the clique `S` has a moment block of size `C(K+2,2)`. Partial sums
   `p_k = p_{k-1} + s_k` give cliques `{s_k, p_{k-1}, p_k}` of size 3
   (`budget="chain"`). Both variants are tested against the dense bound.

Cost: the auxiliary variables add moments and blocks. The size metrics in
`results/complexity.csv` include them; the v1.0 figures did not, because v1.0 did
not impose the budget.

## Eliminating dependent variables from the PSD basis

An equality `h = 0` with `h` linear makes monomials containing one of its variables
linear combinations of the others, so a moment matrix built on all monomials is
singular by construction and has no strictly feasible point; the interior-point
solver then stalls (`optimal_inaccurate`) or fails. `_bases` removes, for each
linear equality of a clique, the variable with the largest index from that clique's
PSD basis. The equalities `L(h x^beta) = 0` (all `beta` of degree <= 2d-1 in the
full clique) are still imposed, so the relaxation is equivalent and well posed.
Redundant equality rows are removed by pivoted QR.

## Order d = 2 only

The objective has degree 4, so `d >= 2` is required (`d = 1` raises; the v1.0
"d=1 bound" silently dropped the degree-4 terms). `d = 2` is the smallest order
that contains the objective, and the dense SDP at `d = 3` is already too large for
`n` beyond a handful of assets (dense block `C(n+3,3)`). Order 3 is used in tests
(`n <= 3`) to check monotonicity of the hierarchy.

## Solver

CLARABEL is the default: interior point, handles PSD and SOC cones, reports status and
iteration counts. The code sets three tolerances, `tol_gap_abs`, `tol_gap_rel` and `tol_feas`,
to 1e-7 unless the caller passes others (`MomentRelaxation.solve`); CLARABEL's own default is
1e-8 for each. Every other setting keeps CLARABEL's default, including the reduced tolerances
(gap 5e-5, feasibility 1e-4) at which the solver reports `AlmostSolved`, which CVXPY maps to
`optimal_inaccurate`. The reason recorded for 1e-7 is that 1e-8 stalls just above the threshold
on rank-deficient optima; it was not re-measured here. `optimal_inaccurate` is reported as
`inaccurate=True`, never as success. SCS is available with `eps=1e-6` and `max_iters=100000` by
default in the code; it is first-order (accuracy of the order of 1e-4 to 1e-6, recorded and not
re-measured), so rank decisions are not trustworthy with it.

## Certificates and the rank rule

The primary certificate is a posteriori: a feasible `x_hat` with
`f(x_hat) - lambda_d <= eps`. It holds whatever the numerical rank of the
moment matrix, to solver accuracy: `lambda_d` is the primal objective of the moment problem
returned by the solver, not the dual objective (see the appendix of the paper). Candidates come from the extraction and from the first-order
moments, and are polished by a local solve (extraction from a solver-accuracy
moment matrix is accurate to ~1e-5 only). The local best is not a candidate: the
certificate must originate from the moments.

The flat-extension test (`rank M_d = rank M_{d-d_K}`) is secondary. Rank is reported
by relative singular-value threshold (1e-5, above the solver accuracy) and by
largest spectral gap.

## Extraction

Henrion-Lasserre with a real Schur decomposition of a random combination of the
symmetrised multiplication matrices; atoms coordinates are `q_j' N_i q_j`. Clique
atoms are merged by agreement on shared variables. When two atoms coincide on the
shared variables the merge returns all combinations (never a silent pick) and the
certificate filters by feasibility and objective.

## Risk model and its limits

`f(x) = x'Sigma x + sum eta_i x_i^3 + kappa-terms + concentration penalty`. The last term
(`impact_*` in the code) is a stylised penalty on the weights, quartic so that `f` has degree four;
it is not calibrated and not derived from an execution model. The model has no return term,
the kappa-terms are `sum kappa_ij x_i^2 x_j^2`, which is *not* the fourth moment of a
portfolio return (that is a dense tensor contraction `sum kappa_ijkl x_i x_j x_k x_l`),
and the markets are synthetic. The sparsity is a modelling choice (cross terms only
within sectors), not a property of data. Results describe this polynomial family.

## Ball extension

The ball is on the coefficients of the modelled kurtosis monomials (`||Delta kappa||_F <= delta`), so
`max_{Delta} <Delta, z(x)> = delta ||W z(x)||_2`, `z` = monomials `x_i^4`, `x_i^2 x_j^2`,
`W` = 1 on diagonals, `sqrt(2)` off-diagonals (kappa symmetric). In the relaxation,
`z(x)` is replaced by the moments `y`; for a Dirac measure this is exact and in general
`||E z|| <= E||z||`, so the bound is that of the moment-relaxed robust problem.
This is parametric robustness; it is not the Delage-Ye moment-uncertainty set on the
return distribution.
