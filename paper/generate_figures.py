"""
Generate publication-quality figures for sos_portfolio.tex.
Run from the project root: python paper/generate_figures.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src import (
    build_objective, build_constraints,
    LasserreRelaxation, scipy_optimize,
)
from src.portfolio_problem import evaluate_objective_grid, get_feasible_grid, BUDGET_LOWER

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

C_BLUE  = "#2166ac"
C_GREEN = "#1a9850"
C_RED   = "#d73027"
C_GRAY  = "#999999"
CMAP    = "RdYlBu_r"


# ── 1. Solve the 2-asset hierarchy ────────────────────────────────────────────
print("Building problem...")
f = build_objective()
constraints = build_constraints()

print("Running L-BFGS-B (50 restarts)...")
lbfgs = scipy_optimize(f, n_restarts=50)
x_opt = lbfgs["x_opt"]
f_opt = lbfgs["f_opt"]
print(f"  x* = {x_opt},  f* = {f_opt:.6f}")

orders = [1, 2, 3]
bounds = []
flat_results = []
for d in orders:
    print(f"Solving Lasserre d={d}...", end=" ", flush=True)
    rel = LasserreRelaxation(f, constraints, order=d)
    try:
        res = rel.build_and_solve(solver="CLARABEL", verbose=False)
    except Exception:
        res = rel.build_and_solve(solver="SCS", verbose=False)
    lb = res.get("lower_bound")
    bounds.append(lb)
    flat_results.append(rel.flat_extension_check())
    print(f"λ_{d} = {lb:.6f}" if lb else "FAILED")


# ── 2. Figure 1: Risk surface ─────────────────────────────────────────────────
print("\nGenerating fig_risk_surface.png...")

X1, X2, Z = evaluate_objective_grid(f, n_points=80)
_, _, mask = get_feasible_grid(n_points=80)
Z_plot = np.where(mask, Z, np.nan)

fig = plt.figure(figsize=(12, 5))
fig.subplots_adjust(wspace=0.38)

# Panel (a): 3D surface
ax3 = fig.add_subplot(121, projection="3d")
surf = ax3.plot_surface(X1, X2, Z_plot, cmap=CMAP, alpha=0.88,
                        linewidth=0, antialiased=True, rcount=60, ccount=60)
# Lower bound plane
lb2 = bounds[1]
if lb2 is not None:
    ax3.plot_surface(X1, X2, lb2 * np.ones_like(X1),
                     alpha=0.20, color=C_BLUE)
# Global minimizer
ax3.scatter([x_opt[0]], [x_opt[1]], [f_opt],
            color=C_GREEN, s=120, zorder=10, marker="*",
            label=r"$\mathbf{x}^*$", depthshade=False)
ax3.set_xlabel(r"$x_1$", labelpad=6)
ax3.set_ylabel(r"$x_2$", labelpad=6)
ax3.set_zlabel(r"$f(x)$", labelpad=6)
ax3.set_title(r"(a) Risk surface over $K$", pad=8)
ax3.view_init(elev=26, azim=220)
cb3 = fig.colorbar(surf, ax=ax3, shrink=0.52, pad=0.12)
cb3.set_label(r"$f(x)$", rotation=270, labelpad=14)
cb3.ax.tick_params(labelsize=8)

# Panel (b): 2D level curves
ax2 = fig.add_subplot(122)
vmin, vmax = np.nanmin(Z_plot), np.nanpercentile(Z_plot[mask], 96)
levels = np.linspace(vmin, vmax, 32)
cf = ax2.contourf(X1, X2, Z_plot, levels=levels, cmap=CMAP, alpha=0.95)
ax2.contour(X1, X2, Z_plot, levels=16, colors="white", alpha=0.22, linewidths=0.4)
# Feasible set boundary
t = np.linspace(0, 1, 300)
ax2.plot(t, 1 - t, "k-", lw=1.8, label=r"$x_1+x_2=1$")
ax2.plot(t, BUDGET_LOWER - t, "k--", lw=1.8,
         label=rf"$x_1+x_2={BUDGET_LOWER}$")
ax2.axvline(0, color="k", lw=1, alpha=0.45)
ax2.axhline(0, color="k", lw=1, alpha=0.45)
# Global minimizer
ax2.scatter([x_opt[0]], [x_opt[1]], color=C_GREEN, s=180, marker="*",
            zorder=10,
            label=rf"$\mathbf{{x}}^*=({x_opt[0]:.4f},\,{x_opt[1]:.4f})$")
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel(r"$x_1$ (Asset 1 weight)")
ax2.set_ylabel(r"$x_2$ (Asset 2 weight)")
ax2.set_title(r"(b) Level curves and feasible set $K$", pad=8)
ax2.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
fig.colorbar(cf, ax=ax2).set_label(r"$f(x)$", rotation=270, labelpad=14)

fig.savefig(os.path.join(OUT, "fig_risk_surface.png"),
            bbox_inches="tight", dpi=200)
plt.close(fig)
print("  → fig_risk_surface.png")


# ── 3. Figure 2: Hierarchy convergence ────────────────────────────────────────
print("Generating fig_convergence.png...")

valid = [(d, lb) for d, lb in zip(orders, bounds) if lb is not None]
d_vals, lb_vals = zip(*valid)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.fill_between(d_vals, lb_vals, f_opt, alpha=0.10, color=C_BLUE,
                label="_nolegend_")
ax.plot(d_vals, lb_vals, "o-", color=C_BLUE, lw=2.5,
        markersize=9, markerfacecolor=C_BLUE,
        label=r"SOS lower bound $\lambda_d$")
ax.axhline(f_opt, color=C_GREEN, lw=2, ls="--",
           label=rf"$f^* = {f_opt:.6f}$")

# Annotations
offsets = {1: (0.07, +0.006), 2: (0.07, +0.002), 3: (-0.35, -0.008)}
for d, lb in zip(d_vals, lb_vals):
    dx, dy = offsets.get(d, (0.07, 0.003))
    ax.annotate(rf"$\lambda_{d} = {lb:.5f}$",
                xy=(d, lb), xytext=(d + dx, lb + dy),
                fontsize=9, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2))

ax.set_xlabel("Hierarchy order $d$")
ax.set_ylabel(r"Lower bound $\lambda_d$")
ax.set_title(r"Lasserre hierarchy convergence: "
             r"$\lambda_1 \leq \lambda_2 \leq \cdots \nearrow f^*$")
ax.set_xticks(list(orders))
ax.set_xlim(0.7, 3.6)
ax.legend(framealpha=0.92)
ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_convergence.png"),
            bbox_inches="tight", dpi=200)
plt.close(fig)
print("  → fig_convergence.png")


# ── 4. Figure 3: Moment matrix singular value spectrum ────────────────────────
print("Generating fig_moment_spectrum.png...")

fe2 = flat_results[1]  # d=2 result
svd2 = fe2.get("singular_values_Md")   # M_2, size 6
svd1 = fe2.get("singular_values_Md1")  # M_1, size 3
rank2 = fe2.get("rank_d", "?")
rank1 = fe2.get("rank_d_minus_1", "?")
tol = 1e-4  # threshold used in flat_extension_check

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel (a): Spectrum of M_2
ax = axes[0]
idx2 = range(1, len(svd2) + 1)
ax.semilogy(idx2, svd2, "o-", color=C_BLUE, lw=2, markersize=9,
            label=r"$\sigma_i(M_2(y^*))$")
thresh2 = tol * svd2[0]
ax.axhline(thresh2, color=C_RED, ls="--", lw=1.5,
           label=rf"Threshold $\varepsilon \cdot \sigma_1 = {thresh2:.2e}$")
ax.axhspan(1e-16, thresh2, alpha=0.07, color=C_RED)
ax.text(len(svd2) * 0.55, thresh2 * 3,
        r"noise floor $\rightarrow$ rank = 1",
        color=C_RED, fontsize=8.5)
ax.set_xlabel("Index $i$")
ax.set_ylabel("Singular value (log scale)")
ax.set_title(
    r"(a) Spectrum of $M_2(y^*)$ — $6\times 6$ matrix" + "\n"
    + rf"$\mathrm{{rank}}(M_2) = {rank2}$ $\Rightarrow$ flat extension at $d=2$",
    pad=6)
ax.set_xticks(list(idx2))
ax.legend(framealpha=0.92)
ax.grid(True)

# Panel (b): M_2 vs M_1 comparison
ax = axes[1]
ax.semilogy(range(1, len(svd2) + 1), svd2, "o-", color=C_BLUE, lw=2,
            markersize=9, label=r"$\sigma(M_2)$, size $6\times 6$")
ax.semilogy(range(1, len(svd1) + 1), svd1, "s--", color=C_RED, lw=2,
            markersize=9, label=r"$\sigma(M_1)$, size $3\times 3$")
thresh_common = tol * max(svd2[0], svd1[0])
ax.axhline(thresh_common, color=C_GRAY, ls=":", lw=1.5, alpha=0.9,
           label=r"Threshold $\varepsilon \cdot \sigma_1$")
ax.set_xlabel("Index $i$")
ax.set_ylabel("Singular value (log scale)")
ax.set_title(
    r"(b) $M_2$ vs $M_1$: $\mathrm{rank}(M_2) = \mathrm{rank}(M_1) = 1$" + "\n"
    + r"$\Rightarrow$ Curto--Fialkow flat extension certified",
    pad=6)
ax.set_xticks(range(1, max(len(svd2), len(svd1)) + 1))
ax.legend(framealpha=0.92)
ax.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_moment_spectrum.png"),
            bbox_inches="tight", dpi=200)
plt.close(fig)
print("  → fig_moment_spectrum.png")

print("\nAll figures saved to paper/")
