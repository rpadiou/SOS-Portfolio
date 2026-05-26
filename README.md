# SOS-Portfolio

**Global Portfolio Optimization via Sparse & Distributionally Robust Lasserre SOS Hierarchy**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-51%2F51%20passing-brightgreen)]()
[![Methods](https://img.shields.io/badge/Methods-Sparse%20SOS%20%7C%20DR--SDP%20%7C%20Chordal%20Sparsity-red)]()

Portfolio risk encoded as a degree-4 polynomial (variance + skewness + kurtosis + Almgren–Chriss impact) and solved **globally** using the Lasserre SOS hierarchy. At hierarchy order d=2, the Curto–Fialkow flat extension condition certifies the unique global minimizer — a guarantee inaccessible to gradient-based methods.

Scales to **n=50+ assets** via correlative sparsity (Waki–Kim–Kojima–Muramatsu 2006): chordal extension of the correlation graph decomposes the single exponential-size SDP into small per-clique blocks, achieving a **398×** reduction in scalar variables for n=50.

> **Companion paper (methodology, proofs, numerical results):**
> **[paper/sos_portfolio.pdf](paper/sos_portfolio.pdf)**

---

> **Note — showcase repository.**
> This project is the result of several months of personal work at the intersection of two interests: computational algebra, studied as part of my coursework at [Télécom Paris](https://www.telecom-paris.fr/), and quantitative finance. The goal was to take the theoretical machinery of the Lasserre SOS hierarchy all the way to a working, tested implementation — and to understand, from first principles, what a *certified* global optimum actually means.

---

## Motivation

Markowitz mean-variance optimization assumes Gaussian returns and solves a convex quadratic program. Real equity returns are fat-tailed (excess kurtosis ≈ 8–10), negatively skewed, and subject to non-linear market impact — all of which are exactly polynomial, but degree-4 polynomials are non-convex.

Gradient-based methods (L-BFGS-B, SLSQP) find local minima but provide **no mathematical certificate** of global optimality. For an institutional portfolio manager, this is insufficient: a committee of risk or a regulator requires a traceable proof that the found allocation is globally optimal, not merely locally optimal.

The SOS hierarchy provides this certificate via the flat extension condition: when rank(M₂) = rank(M₁) = 1, the solution is provably the unique global minimizer. This is achieved in 0.11 s on the 2-asset problem and scales to n=50 assets via chordal sparsity.

---

## Results

### 2-asset problem: dense hierarchy (exact, certified)

| Order d | Matrix size | Lower bound λ_d | Gap to f* | Time |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 3×3 | 0.016988 | 0.170383 | 0.05 s |
| **2** | **6×6** | **0.187371** | **< 10⁻⁶** | **0.11 s** |
| 3 | 10×10 | 0.187371 | < 10⁻⁶ | 0.31 s |

At d=2: rank(M₂) = rank(M₁) = 1 → **flat extension certified**.
Unique global minimizer: **x₁* = 0.6058, x₂* = 0.3442**, f* = 0.187371.

### Sparse SOS: n=15 assets (3 cliques × 5 assets)

| Method | Bound | SDP variables | Time |
|:--|:---:|:---:|:---:|
| Dense d=2 (theoretical) | f* | 18,496 scalar | > 60 s |
| **Sparse d=2 (implemented)** | **f*** | **376 scalar** | **< 1 s** |

Certified gap: lb ≤ f* ≤ ub, flat extension per clique. 51/51 unit tests pass.

### Complexity reduction: sparse vs dense for n=50

| Configuration | SDP scalar variables | Reduction |
|:--|:---:|:---:|
| Dense n=50, d=2 | 1,326² = 1,758,276 | 1× |
| Sparse n=50, 10 cliques of 5 | 10 × 21² = 4,410 | **398×** |

---

## Architecture

```
SOS-Portfolio/
├── src/
│   ├── polynomial_ring.py      # R[x₁,...,xₙ]: sparse coeff dict, graded-lex monomial enum
│   ├── portfolio_problem.py    # Degree-4 objective + semialgebraic constraints
│   ├── graph_sparsity.py       # ChordalExtension: MDO → PEO → maximal cliques → RIP
│   ├── indexer.py              # SparseIndexer: bijection local↔global moment indices
│   ├── sos_hierarchy.py        # SparseLasserreRelaxation + LasserreRelaxation (dense)
│   ├── extractor.py            # MinimizerExtractor: Curto-Fialkow + Henrion-Lasserre
│   ├── local_solver.py         # L-BFGS-B + SLSQP for arbitrary n
│   └── visualization.py        # 3D surface, convergence, moment spectra
├── paper/
│   ├── sos_portfolio.tex       # Companion paper (LaTeX source)
│   └── sos_portfolio.pdf       # Compiled paper
├── tests/
│   └── test_sparse_sos.py      # 51 unit tests across 9 test classes
├── main.py                     # CLI: --mode sparse|demo --n-assets --n-clusters --delta-robust
└── requirements.txt
```

**Data flow (sparse mode):**
`SyntheticMarket` → `build_block_adjacency` → `ChordalExtension` → `SparseIndexer` → `SparseLasserreRelaxation` → `MinimizerExtractor` + `scipy_optimize`

---

## Usage

```bash
pip install -r requirements.txt

# 2-asset demo (dense hierarchy, all figures)
python main.py --mode demo --save-figs

# Sparse pipeline — n=15 assets, 3 clusters
python main.py --mode sparse --n-assets 15 --n-clusters 3

# Sparse + Distributionally Robust (δ=0.02 uncertainty on kurtosis)
python main.py --mode sparse --n-assets 15 --n-clusters 3 --delta-robust 0.02

# Run all 51 unit tests
python -m pytest tests/ -v
```

Note: the tests import the `src` package directly. To run the tests locally either set the project root on `PYTHONPATH` or install the package in editable mode:

```bash
# Option 1: run tests without installing
PYTHONPATH=. python -m pytest tests/ -v

# Option 2: install editable and run tests
pip install -e .
python -m pytest tests/ -v
```

Solver options: `SCS` (default, open-source), `CLARABEL` (recommended for n>10), `MOSEK` (commercial, fastest).

---

**Author:** Raphaël Padiou — full methodology, proofs, and numerical results in [paper/sos_portfolio.pdf](paper/sos_portfolio.pdf).
