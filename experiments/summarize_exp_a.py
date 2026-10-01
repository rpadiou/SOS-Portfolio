"""Summarise results/exp_a.jsonl -> results/exp_a.csv, results/exp_a_summary.md, figures."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
COL = {"calibrated": "#0072B2", "mixed": "#E69F00", "stress": "#D55E00"}   # Okabe-Ito
Ks = [1, 5, 20, 100]


def load(path):
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    df["regret1"] = (df["ub_K1"] - df["f_cert"]) / np.maximum(df["f_cert"].abs(), 1e-3)
    df["dd"] = df["lb_dense"] - df["lb_sparse"]
    return df.sort_values(["regime", "n", "seed"]).reset_index(drop=True)


def pct(s):
    return f"{100 * np.mean(s):.0f}%"


def table(df):
    rows = []
    for (reg, n), g in df.groupby(["regime", "n"]):
        gr = g["gap_rel"].dropna()
        row = {"regime": reg, "n": n, "N": len(g), "multi-min": pct(g["n_minima"] >= 2),
               "certified": pct(g["certified"]), "flat": pct(g["flat"].fillna(False).astype(bool)),
               "p(1 start fails)": f"{g['p_single_fail'].mean():.2f}"}
        for K in Ks:
            row[f"fail K={K}"] = pct(g[f"fail_K{K}"])
        row["gap_rel med"] = f"{gr.median():.1e}" if len(gr) else ""
        row["gap_rel p90"] = f"{gr.quantile(0.9):.1e}" if len(gr) else ""
        row["gap_rel max"] = f"{gr.max():.1e}" if len(gr) else ""
        row["t_sos med (s)"] = f"{g['t_sparse'].median():.2f}"
        row["t_local100 med (s)"] = f"{g['t_local'].median():.2f}"
        dd = g["dd"].dropna()
        row["dense-sparse max"] = f"{dd.max():.1e}" if len(dd) else "n/a"
        row["inaccurate"] = int(g["inaccurate_sparse"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def md(t):
    cols = list(t.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values.tolist()])


def figures(df):
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    regs = [r for r in COL if r in set(df["regime"])]
    ns = sorted(df["n"].unique())
    # 1. single-start regret
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for r in regs:
        g = df[df.regime == r].groupby("n")["regret1"]
        ax.plot(g.median().index, g.quantile(0.9).clip(lower=1e-9), "o-", color=COL[r], label=f"{r} (p90)")
        ax.plot(g.median().index, g.median().clip(lower=1e-9), "o--", color=COL[r], alpha=0.5, label=f"{r} (median)")
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("number of assets n"); ax.set_ylabel("regret of a single start\n(f - f_cert) / max(|f_cert|, 1e-3)")
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "figures/exp_a_regret_single_start.png"), dpi=160); plt.close(fig)
    # 2. success probability vs K
    fig, axes = plt.subplots(1, len(regs), figsize=(3.2 * len(regs), 3.2), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], regs):
        for n in ns:
            g = df[(df.regime == r) & (df.n == n)]
            ax.plot(Ks, [1 - g[f"fail_K{K}"].mean() for K in Ks], "o-", label=f"n={n}")
        ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.set_title(r, fontsize=9)
        ax.set_xlabel("number of starts K"); ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("P(best of K reaches f_cert)"); axes[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "figures/exp_a_success_vs_K.png"), dpi=160); plt.close(fig)
    # 3. time
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    g = df.groupby("n")
    ax.plot(ns, g["t_sparse"].median().reindex(ns), "o-", color="#0072B2", label="sparse SOS (median)")
    gd = df.dropna(subset=["t_dense"]).groupby("n")["t_dense"].median()
    ax.plot(gd.index, gd.values, "s-", color="#CC79A7", label="dense SOS (median)")
    ax.plot(ns, g["t_local"].median().reindex(ns), "^-", color="#009E73", label="100 local runs (median)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("number of assets n"); ax.set_ylabel("wall time (s)"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "figures/exp_a_time.png"), dpi=160); plt.close(fig)
    # 4. dense - sparse
    d = df.dropna(subset=["dd"])
    if len(d):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for r in regs:
            g = d[d.regime == r]
            ax.scatter(g["n"] + np.random.RandomState(0).uniform(-0.25, 0.25, len(g)), g["dd"].abs().clip(lower=1e-10),
                       s=8, color=COL[r], alpha=0.6, label=r)
        ax.set_yscale("log"); ax.set_xlabel("number of assets n"); ax.set_ylabel("|lb_dense - lb_sparse|")
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(RES, "figures/exp_a_dense_minus_sparse.png"), dpi=160); plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "exp_a.jsonl")
    df = load(path)
    df.to_csv(os.path.join(RES, "exp_a.csv"), index=False)
    t = table(df)
    viol = int(((df["lb_sparse"] - df["ub_K100"]) > 1e-5 * np.maximum(1, df["ub_K100"].abs())).sum())
    lines = ["# Experiment A summary (generated by experiments/summarize_exp_a.py)", "",
             f"Instances: {len(df)}. Bound validity violations (lb_sparse > best local UB + 1e-5 relative): {viol}.",
             f"Sparse solves with status other than `optimal`: {int((df['status_sparse'] != 'optimal').sum())}; "
             f"no bound returned: {int(df['lb_sparse'].isna().sum())}.", "",
             "Definitions: see experiments/PROTOCOL.md. `certified` = a posteriori certificate at d=2; "
             "`fail K` = best of K local runs misses f_cert by more than tau; gap_rel = (UB_100 - lb)/max(|lb|, 1e-3).", "",
             md(t), ""]
    open(os.path.join(RES, "exp_a_summary.md"), "w").write("\n".join(lines))
    figures(df)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
