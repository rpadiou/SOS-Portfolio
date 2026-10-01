"""Render result tables from results/ into README.md between <!-- BEGIN:name --> / <!-- END:name --> markers.

Also writes each table to results/tables/<name>.md. Nothing in the README results section is typed by hand.
"""
import json
import os
import re

import numpy as np
import pandas as pd

RES = "results"


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values.tolist()])


def sci(x, nd=1):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}e}"


def t_two_asset():
    d = json.load(open(f"{RES}/two_asset.json"))
    nc = json.load(open(f"{RES}/two_asset_nonconvex.json"))
    rows = [{"instance": "2-asset benchmark (validation)", "min Hessian eig on K": f"{d['min_hessian_eigenvalue_on_K']:.3f}",
             "local minima": 1, "lambda_2": f"{d['full']['lb_d2']:.6f}", "lambda_3": f"{d['full']['lb_d3']:.6f}",
             "best local f": f"{d['full']['f_local']:.6f}"},
            {"instance": f"2-asset stress (seed {nc['seed']}, skew {nc['skew_scale']:g})",
             "min Hessian eig on K": f"{nc['min_hessian_eig']:.3f}", "local minima": len(nc["local_minima"]),
             "lambda_2": f"{nc['hierarchy']['d2']['lb']:.6f}", "lambda_3": f"{nc['hierarchy']['d3']['lb']:.6f}",
             "best local f": f"{min(m['f'] for m in nc['local_minima']):.6f}"}]
    return md(pd.DataFrame(rows))


def t_ablation():
    d = json.load(open(f"{RES}/two_asset.json"))
    rows = [{"problem": "full", "x1": f"{d['full']['x_local'][0]:.3f}", "x2": f"{d['full']['x_local'][1]:.3f}", "f*": f"{d['full']['f_local']:.5f}"}]
    for k, v in d["ablation"].items():
        rows.append({"problem": k.replace("_", " "), "x1": f"{v['x_local'][0]:.3f}", "x2": f"{v['x_local'][1]:.3f}", "f*": f"{v['f_local']:.5f}"})
    return md(pd.DataFrame(rows))


def t_complexity():
    c = pd.read_csv(f"{RES}/complexity.csv")
    c["PSD entries"] = c["psd_entries"].map("{:,}".format)
    c["unique moments"] = c["unique_moments"].map("{:,}".format)
    t = c[["n", "structure", "budget", "cliques", "max_block", "PSD entries", "unique moments", "wall_s", "iterations", "status"]]
    t = t.rename(columns={"max_block": "largest PSD block", "wall_s": "wall (s)"})
    return md(t.fillna(""))


def t_dense():
    d = pd.read_csv(f"{RES}/dense_baseline.csv")
    t = d[["n", "structure", "status", "lower_bound", "wall_s", "peak_rss_mb", "max_block", "unique_moments"]].copy()
    t["lower_bound"] = t["lower_bound"].map("{:.6f}".format)
    t["wall_s"] = t["wall_s"].map("{:.2f}".format)
    t["peak_rss_mb"] = t["peak_rss_mb"].map("{:.0f}".format)
    t = t.rename(columns={"wall_s": "wall (s)", "peak_rss_mb": "peak RSS (MB)", "max_block": "largest PSD block"})
    e = pd.read_csv(f"{RES}/dense_extrapolation.csv")
    e["value"] = e.apply(lambda r: f"{r.estimate:.2e} [{r.pi95_low:.1e}, {r.pi95_high:.1e}] {r.unit}", axis=1)
    e = e[["quantity", "n", "value", "exponent_on_block_size", "status"]]
    return md(t) + "\n\nDense extrapolation (log-log fit on the dense runs with n >= 6; 95% prediction interval; not measured):\n\n" + md(e)


def t_exp_a():
    p = f"{RES}/exp_a_summary.md"
    return open(p).read().split("\n", 6)[-1] if os.path.exists(p) else "not run"


def t_file(name):
    p = f"{RES}/{name}"
    if not os.path.exists(p):
        return "not run"
    return "\n".join(open(p).read().splitlines()[2:])


TABLES = {"two_asset": t_two_asset, "ablation": t_ablation, "complexity": t_complexity, "dense": t_dense,
          "exp_a": lambda: t_file("exp_a_summary.md"), "exp_b": lambda: t_file("exp_b_summary.md"),
          "exp_c": lambda: t_file("exp_c_summary.md")}


def main():
    os.makedirs(f"{RES}/tables", exist_ok=True)
    readme = open("README.md").read()
    for name, fn in TABLES.items():
        try:
            body = fn()
        except FileNotFoundError:
            body = "not run"
        open(f"{RES}/tables/{name}.md", "w").write(body + "\n")
        readme = re.sub(rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}",
                        readme, flags=re.S)
    open("README.md", "w").write(readme)


if __name__ == "__main__":
    main()
