"""Paper figures, read from results/ (no hard-coded values).

    python scripts/two_asset_nonconvex.py && python paper/generate_figures.py

Fig. 1  non-convex 2-asset stress instance: f on K and its local minima
Fig. 2  hierarchy bounds versus local minima
Fig. 3  singular values of M_2 and M_1 at the order-2 optimum
Experiment figures (A, B) are written to results/figures/ by the experiment summarisers and copied here.
"""
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sos_portfolio import MultivariatePolynomial

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load():
    d = json.load(open(os.path.join(RES, "two_asset_nonconvex.json")))
    f = MultivariatePolynomial(2, {tuple(eval(k)): v for k, v in d["coefficients"].items()})
    return d, f


def fig1(d, f):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    t = np.linspace(0, 1, 401)
    for s, ls in [(1.0, "-"), (0.95, "--")]:
        u = t * s
        a.plot(u, [f(np.array([x, s - x])) for x in u], ls, color=BLUE, label=f"$x_1+x_2={s}$")
    lb = d["hierarchy"]["d2"]["lb"]
    a.axhline(lb, color=GREEN, lw=1, ls=":", label=r"$\lambda_2$")
    for m in d["local_minima"]:
        a.plot(m["x"][0], m["f"], "o", color=RED, ms=6)
        a.annotate(f"{100 * m['share']:.0f}% of starts", (m["x"][0], m["f"]), textcoords="offset points",
                   xytext=(6, 8), fontsize=8)
    a.set_xlabel(r"$x_1$ (with $x_2 = s - x_1$, $s$ = total invested)")
    a.set_ylabel(r"$f$"); a.legend(frameon=False, fontsize=8); a.set_title("(a) $f$ along the budget boundaries", fontsize=9)
    g = np.linspace(0, 1, 160)
    X, Y = np.meshgrid(g, g)
    Z = np.array([[f(np.array([x, y])) for x in g] for y in g])
    c = b.contourf(X, Y, Z, levels=24, cmap="viridis", alpha=0.85)
    b.plot([0, 1], [1, 0], color="white", lw=1.2); b.plot([0, 0.95], [0.95, 0], color="white", lw=1.2, ls="--")
    for m in d["local_minima"]:
        b.plot(*m["x"], "o", color=RED, mec="white", ms=7)
    b.set_xlabel("$x_1$"); b.set_ylabel("$x_2$"); b.set_aspect("equal")
    b.set_title("(b) $f$ on $[0,1]^2$; $K$ is the strip between the white lines", fontsize=9)
    fig.colorbar(c, ax=b, shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_risk_surface.png"), dpi=170); plt.close(fig)


def fig2(d):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ds = ["d2", "d3"]
    ax.plot([2, 3], [d["hierarchy"][k]["lb"] for k in ds], "o-", color=BLUE, label=r"lower bound $\lambda_d$")
    for i, m in enumerate(d["local_minima"]):
        ax.axhline(m["f"], color=RED if i else GREEN, ls="--", lw=1,
                   label=("global minimum" if i == 0 else "other local minimum"))
    ax.set_xticks([2, 3]); ax.set_xlabel("relaxation order $d$"); ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_convergence.png"), dpi=170); plt.close(fig)


def fig3(d):
    h = d["hierarchy"]["d2"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.semilogy(range(1, len(h["sv_d"]) + 1), np.maximum(h["sv_d"], 1e-16), "o-", color=BLUE, label=r"$M_2$ ($6\times 6$)")
    ax.semilogy(range(1, len(h["sv_prev"]) + 1), np.maximum(h["sv_prev"], 1e-16), "s-", color=ORANGE, label=r"$M_1$ ($3\times 3$)")
    ax.axhline(1e-5 * h["sv_d"][0], color=GREY, ls=":", label="rank threshold $10^{-5}\\sigma_1$")
    ax.set_xlabel("index"); ax.set_ylabel("singular value"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_moment_spectrum.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    d, f = load()
    fig1(d, f); fig2(d); fig3(d)
    src = os.path.join(RES, "figures")
    if os.path.isdir(src):
        for fn in os.listdir(src):
            shutil.copy(os.path.join(src, fn), os.path.join(HERE, fn))
