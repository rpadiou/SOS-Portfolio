"""
Visualization Module — SOS-Portfolio
======================================

Generates publication-quality figures for the SOS hierarchy results.

Figures Produced
-----------------
1. Risk Surface: 3D plot of f(x_1, x_2) over the feasible set K.
   Reveals the non-convex landscape with multiple local minima.

2. Convergence Plot: λ_d vs hierarchy order d, showing monotone increase
   towards f* (certified by Lasserre 2001).

3. Feasible Region (Algebraic Variety): 2D contour + feasible set boundary,
   overlaid with local minima found by L-BFGS-B and SOS solution.

4. Moment Matrix Spectrum: Singular values of M_d(y) to visualize the
   rank condition for flat extension convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import List, Dict, Optional, Tuple

from .portfolio_problem import (
    MultivariatePolynomial, evaluate_objective_grid, get_feasible_grid, BUDGET_LOWER
)


# ── Plot style ────────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "monospace",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}

SOS_COLOR = "#58a6ff"       # blue for SOS bounds
LBFGS_COLOR = "#f97316"    # orange for L-BFGS-B
OPT_COLOR = "#3fb950"       # green for global optimum
SURFACE_CMAP = "plasma"


def plot_risk_surface(f: MultivariatePolynomial,
                      sos_result: Optional[Dict] = None,
                      lbfgs_result: Optional[Dict] = None,
                      n_points: int = 80,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the non-convex risk surface f(x_1, x_2) as a 3D landscape.

    The feasible set K is highlighted. Local minima from L-BFGS-B and the
    SOS global lower bound plane are overlaid.

    Parameters
    ----------
    f : MultivariatePolynomial
        Objective polynomial.
    sos_result : dict, optional
        Result from LasserreRelaxation.build_and_solve().
    lbfgs_result : dict, optional
        Result from scipy_optimize().
    n_points : int
        Surface resolution.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
    """
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle("SOS-Portfolio: Non-Convex Portfolio Risk Surface",
                     fontsize=15, color="#e6edf3", fontweight="bold", y=0.98)

        ax_3d = fig.add_subplot(121, projection="3d")
        ax_2d = fig.add_subplot(122)

        # ── 3D Surface ───────────────────────────────────────────────────
        X1, X2, Z = evaluate_objective_grid(f, n_points)
        _, _, mask = get_feasible_grid(n_points)

        Z_plot = Z.copy()
        Z_plot[~mask] = np.nan  # only show feasible region

        surf = ax_3d.plot_surface(X1, X2, Z_plot, cmap=SURFACE_CMAP,
                                   alpha=0.85, linewidth=0, antialiased=True)

        if lbfgs_result:
            for res in lbfgs_result.get("all_results", [])[:5]:
                x = res["x"]
                z = f(x)
                ax_3d.scatter([x[0]], [x[1]], [z], color=LBFGS_COLOR,
                               s=40, zorder=5, alpha=0.8)

        if lbfgs_result and lbfgs_result.get("x_opt") is not None:
            x_opt = lbfgs_result["x_opt"]
            z_opt = f(x_opt)
            ax_3d.scatter([x_opt[0]], [x_opt[1]], [z_opt],
                           color=OPT_COLOR, s=120, zorder=10,
                           marker="*", label="L-BFGS-B best")

        if sos_result and sos_result.get("lower_bound") is not None:
            lb = sos_result["lower_bound"]
            ax_3d.plot_surface(X1, X2, lb * np.ones_like(X1),
                                alpha=0.25, color=SOS_COLOR)
            ax_3d.text(0.5, 0.5, lb, f"λ_SOS = {lb:.4f}",
                       color=SOS_COLOR, fontsize=9)

        ax_3d.set_xlabel("x₁ (Asset 1 weight)", labelpad=8)
        ax_3d.set_ylabel("x₂ (Asset 2 weight)", labelpad=8)
        ax_3d.set_zlabel("f(x) Risk", labelpad=8)
        ax_3d.set_title("Risk Landscape (feasible region only)", pad=12)
        ax_3d.tick_params(colors="#8b949e")
        cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.5, pad=0.1, label="f(x) — Portfolio Risk")
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
        cbar.set_label("f(x) — Portfolio Risk", color="#8b949e")

        # ── 2D Contour (Algebraic Variety view) ─────────────────────────
        Z_2d = Z.copy()
        Z_2d[~mask] = np.nan

        contour = ax_2d.contourf(X1, X2, Z_2d, levels=30, cmap=SURFACE_CMAP, alpha=0.9)
        ax_2d.contour(X1, X2, Z_2d, levels=15, colors="white", alpha=0.2, linewidths=0.5)

        # Feasible set boundary
        budget_line_x = np.linspace(0, 1, 200)
        ax_2d.plot(budget_line_x, 1.0 - budget_line_x, color="#30363d",
                   linewidth=2, label="x₁+x₂=1")
        ax_2d.plot(budget_line_x, BUDGET_LOWER - budget_line_x,
                   color="#30363d", linewidth=2, linestyle="--",
                   label=f"x₁+x₂={BUDGET_LOWER}")
        ax_2d.axvline(0, color="#30363d", linewidth=1.5, alpha=0.6)
        ax_2d.axhline(0, color="#30363d", linewidth=1.5, alpha=0.6)

        # L-BFGS-B local minima scatter
        if lbfgs_result:
            all_res = lbfgs_result.get("all_results", [])
            xs = [r["x"][0] for r in all_res]
            ys = [r["x"][1] for r in all_res]
            ax_2d.scatter(xs, ys, color=LBFGS_COLOR, s=25, alpha=0.6,
                          label="L-BFGS-B endpoints", zorder=5)
            if lbfgs_result.get("x_opt") is not None:
                x_opt = lbfgs_result["x_opt"]
                ax_2d.scatter([x_opt[0]], [x_opt[1]], color=OPT_COLOR,
                               s=200, marker="*", zorder=10,
                               label=f"L-BFGS-B best f={lbfgs_result['f_opt']:.4f}")

        ax_2d.set_xlim(0, 1)
        ax_2d.set_ylim(0, 1)
        ax_2d.set_xlabel("x₁ (Asset 1 weight)")
        ax_2d.set_ylabel("x₂ (Asset 2 weight)")
        ax_2d.set_title("Algebraic Variety of K (2D Contour)")
        ax_2d.legend(fontsize=8, facecolor="#21262d", edgecolor="#30363d",
                     labelcolor="#e6edf3", loc="upper right")
        fig.colorbar(contour, ax=ax_2d, label="f(x)")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig


def plot_convergence(sos_bounds: List[float],
                     orders: List[int],
                     lbfgs_best: Optional[float] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the SOS hierarchy convergence: λ_d ↗ f* as d increases.

    Shows the monotone sequence of lower bounds produced by the Lasserre
    hierarchy, with the L-BFGS-B best as a reference upper bound.

    Parameters
    ----------
    sos_bounds : List[float]
        Lower bounds λ_d for each hierarchy order d.
    orders : List[int]
        Hierarchy orders [d_1, d_2, ...].
    lbfgs_best : float, optional
        Best objective value found by L-BFGS-B (upper bound on f*).
    save_path : str, optional
        Save path.

    Returns
    -------
    plt.Figure
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        valid = [(d, lb) for d, lb in zip(orders, sos_bounds) if lb is not None]
        if valid:
            d_vals, lb_vals = zip(*valid)
            ax.plot(d_vals, lb_vals, color=SOS_COLOR, linewidth=2.5,
                    marker="o", markersize=10, markerfacecolor=SOS_COLOR,
                    label="SOS lower bound λ_d")
            ax.fill_between(d_vals, lb_vals, alpha=0.15, color=SOS_COLOR)

            for d, lb in zip(d_vals, lb_vals):
                ax.annotate(f"$\\lambda_{{{d}}}$ = {lb:.5f}",
                             xy=(d, lb), xytext=(d + 0.08, lb + 0.002),
                             color=SOS_COLOR, fontsize=10,
                             arrowprops=dict(arrowstyle="->", color=SOS_COLOR,
                                             lw=1.2))

        if lbfgs_best is not None:
            ax.axhline(lbfgs_best, color=LBFGS_COLOR, linestyle="--",
                       linewidth=2, label=f"L-BFGS-B best: {lbfgs_best:.5f}")
            ax.fill_between(orders or [1, 3], lbfgs_best,
                             max(lb_vals) if valid else lbfgs_best,
                             alpha=0.1, color=LBFGS_COLOR)

        ax.set_xlabel("Hierarchy order d")
        ax.set_ylabel("Lower bound λ_d on f*")
        ax.set_title(
            r"Lasserre Hierarchy Convergence: $\lambda_1 \leq \lambda_2 \leq \cdots \nearrow f^* = \min_{x \in K} f(x)$",
            pad=15
        )
        ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(orders)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig


def plot_moment_matrix_spectrum(flat_ext_results: List[Dict],
                                 orders: List[int],
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot singular value spectra of M_d(y) to visualize flat extension.

    The flat extension condition rank(M_d) = rank(M_{d-1}) is visible as
    a gap in the singular value spectrum — values below the gap are numerical
    noise, those above indicate the true rank.

    Parameters
    ----------
    flat_ext_results : List[Dict]
        Results from LasserreRelaxation.flat_extension_check() per order.
    orders : List[int]
        Hierarchy orders.
    save_path : str, optional
        Save path.

    Returns
    -------
    plt.Figure
    """
    with plt.rc_context(STYLE):
        n_orders = len(orders)
        fig, axes = plt.subplots(1, n_orders, figsize=(5 * n_orders, 5))
        if n_orders == 1:
            axes = [axes]

        fig.suptitle(r"Moment Matrix $M_d(y)$ Singular Value Spectrum — "
                     r"Flat Ext.: rank$(M_d)$ = rank$(M_{d-1})$ $\Rightarrow$ Global Opt.",
                     fontsize=13, color="#e6edf3", y=1.02)

        for ax, d, res in zip(axes, orders, flat_ext_results):
            if "error" in res or res.get("singular_values_Md") is None:
                ax.text(0.5, 0.5, "d = 1: flat extension\nrequires d ≥ 2",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#8b949e", fontsize=10, linespacing=1.8)
                ax.set_title(f"d = {d}  [N/A — need d ≥ 2]", color="#8b949e")
                ax.axis("off")
                continue

            svd = res["singular_values_Md"]
            svd_prev = res.get("singular_values_Md1", None)

            ax.semilogy(range(1, len(svd) + 1), svd,
                        color=SOS_COLOR, marker="o", markersize=6,
                        linewidth=1.5, label=f"σ(M_{d})")

            if svd_prev is not None:
                ax.semilogy(range(1, len(svd_prev) + 1), svd_prev,
                            color=LBFGS_COLOR, marker="s", markersize=6,
                            linewidth=1.5, linestyle="--",
                            label=f"σ(M_{d-1})")

            converged = res.get("converged", False)
            rank_d = res.get("rank_d", "?")
            rank_prev = res.get("rank_d_minus_1", "?")

            status = ("✓ FLAT" if converged else "✗ NOT FLAT")
            color = OPT_COLOR if converged else "#f85149"
            ax.set_title(f"d = {d}   [{status}]\nrank(M_d)={rank_d}, "
                         f"rank(M_{{d-1}})={rank_prev}", color=color)
            ax.set_xlabel("Index")
            ax.set_ylabel("Singular value (log scale)")
            ax.legend(facecolor="#21262d", edgecolor="#30363d",
                      labelcolor="#e6edf3", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        return fig
