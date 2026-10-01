"""Paper figures, read from results/ (no hard-coded values).

    python scripts/two_asset_nonconvex.py && python paper/generate_figures.py

Fig. 1  non-convex 2-asset stress instance: f on K and its local minima
The experiment figures used in the paper are written to results/figures/ by the summarisers and copied here.
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


if __name__ == "__main__":
    d, f = load()
    fig1(d, f)
    for fn in ("exp_a_success_vs_K.png", "exp_b_regret_n10-mixed-s0.png", "exp_c_cumulative.png"):
        shutil.copy(os.path.join(RES, "figures", fn), os.path.join(HERE, fn))
