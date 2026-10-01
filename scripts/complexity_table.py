"""Complexity metrics of the dense and sparse relaxations -> results/complexity.csv and .md.

Four quantities per configuration (definitions in docs/DESIGN_DECISIONS.md):
  max_block       size of the largest PSD block
  psd_entries     sum over all PSD blocks (moment + localizing) of size^2
  unique_moments  number of free moment variables y_alpha
  wall_s / iters  measured wall time and solver iterations (only where actually solved)

Dense sizes at large n are closed-form (`dense_sizes`, checked against the builder in
tests/test_complexity.py); their time is *not measured* and is left empty. Measured
dense times come from scripts/dense_baseline.py.
"""
from __future__ import annotations

import csv
import time

from sos_portfolio import (ChordalExtension, SyntheticMarket, build_objective_from_market,
                           build_portfolio_relaxation)
from sos_portfolio.sos_hierarchy import dense_sizes

CONFIGS = [(15, 3), (30, 6), (50, 10)]


def main() -> None:
    rows = []
    for n, nc in CONFIGS:
        d = dense_sizes(n)
        rows.append({"n": n, "cliques": 1, "clique_size": n, "structure": "dense", "budget": "direct", **d,
                     "wall_s": "", "iterations": "", "status": "sizes only (closed form), not solved"})
        m = SyntheticMarket.generate(n, nc, 0)
        f = build_objective_from_market(m)
        cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
        for budget in ("star", "chain"):
            pr = build_portfolio_relaxation(f, cl, 0.95, 1.0, 2, "sparse", budget=budget)
            t = time.perf_counter()
            r = pr.solve("CLARABEL")
            rows.append({"n": n, "cliques": len(cl), "clique_size": max(map(len, cl)), "structure": "sparse",
                         "budget": budget, **r["sizes"], "wall_s": round(time.perf_counter() - t, 2),
                         "iterations": r["iterations"], "status": r["status"]})
            print(rows[-1], flush=True)
    keys = list(rows[0])
    with open("results/complexity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
