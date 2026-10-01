"""Measured dense vs sparse SDP cost (order d=2, CLARABEL).

Each (n, structure) runs in its own subprocess under an address-space limit and a
wall-clock timeout; the parent records wall time, peak RSS of the child, status,
bound and sizes. Nothing is extrapolated here (see scripts/complexity_table.py).

Instance: SyntheticMarket.generate(n, nc, seed=3, skew_scale=5, kurt_base=0.3, impact_base=0.05).
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time

NS = [2, 4, 6, 8, 10, 12, 15]


def n_clusters(n: int) -> int:
    return 1 if n == 2 else (3 if n % 3 == 0 and n >= 9 else 2)


def worker(n: int, structure: str) -> None:
    from sos_portfolio import (ChordalExtension, SyntheticMarket, build_objective_from_market,
                               build_portfolio_relaxation)
    m = SyntheticMarket.generate(n, n_clusters(n), 3, skew_scale=5.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques if structure == "sparse" else None
    t = time.perf_counter()
    r = build_portfolio_relaxation(f, cl, 0.95, 1.0, 2, structure).solve("CLARABEL")
    out = {"n": n, "structure": structure, "status": r["status"], "lower_bound": r["lower_bound"],
           "iterations": r["iterations"], "wall_s": time.perf_counter() - t,
           "build_s": r["build_time"], "solve_s": r["solve_time"],
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, **r["sizes"]}
    print("RESULT " + json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("N", "STRUCT"))
    ap.add_argument("--ns", type=int, nargs="*", default=NS)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--mem-gb", type=float, default=10.0)
    ap.add_argument("--out", default="results/dense_baseline.csv")
    a = ap.parse_args()
    if a.worker:
        resource.setrlimit(resource.RLIMIT_AS, (int(a.mem_gb * 2**30),) * 2)
        worker(int(a.worker[0]), a.worker[1])
        return
    rows = []
    for n in a.ns:
        for st in ("dense", "sparse"):
            if st == "sparse" and n == 2:
                continue
            t = time.perf_counter()
            try:
                p = subprocess.run([sys.executable, __file__, "--worker", str(n), st, "--mem-gb", str(a.mem_gb)],
                                   capture_output=True, text=True, timeout=a.timeout)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                row = json.loads(line[0][7:]) if line else {"n": n, "structure": st,
                                                             "status": "failed: " + p.stderr.strip().splitlines()[-1][:120] if p.stderr.strip() else "failed"}
            except subprocess.TimeoutExpired:
                row = {"n": n, "structure": st, "status": f"timeout>{a.timeout:.0f}s", "wall_s": time.perf_counter() - t}
            rows.append(row)
            print(row, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("n", "structure", "status"), k))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
