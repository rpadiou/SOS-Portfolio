"""SOS-Portfolio: Global Portfolio Optimization via Putinar's Positivstellensatz."""

from .polynomial_ring import MultivariatePolynomial, generate_monomials
from .portfolio_problem import (
    build_objective,
    build_constraints,
    build_objective_from_market,
    build_constraints_from_market,
    SyntheticMarket,
)
from .sos_hierarchy import LasserreRelaxation, SparseLasserreRelaxation
from .local_solver import scipy_optimize, analyze_local_minima
from .visualization import plot_risk_surface, plot_convergence, plot_moment_matrix_spectrum
from .graph_sparsity import ChordalExtension, build_correlation_graph, build_block_adjacency
from .indexer import SparseIndexer
from .extractor import MinimizerExtractor

__all__ = [
    "MultivariatePolynomial",
    "generate_monomials",
    "build_objective",
    "build_constraints",
    "build_objective_from_market",
    "build_constraints_from_market",
    "SyntheticMarket",
    "LasserreRelaxation",
    "SparseLasserreRelaxation",
    "scipy_optimize",
    "analyze_local_minima",
    "plot_risk_surface",
    "plot_convergence",
    "plot_moment_matrix_spectrum",
    "ChordalExtension",
    "build_block_adjacency",
    "build_correlation_graph",
    "SparseIndexer",
    "MinimizerExtractor",
]
