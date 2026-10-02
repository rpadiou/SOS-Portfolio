"""The 3-asset, 5-day example of Appendix B of the paper, computed with the code of the experiments.

Writes results/toy_example.json. tests/test_metric_definitions.py checks the same numbers against values worked out by hand.

    python scripts/toy_example.py
"""
import importlib.util
import json

import numpy as np

from sos_portfolio import empirical as em

R = np.array([[0.04, 0.02, 0.00], [-0.02, -0.04, 0.03], [0.06, 0.03, 0.00], [-0.06, -0.03, -0.03], [0.03, 0.00, 0.03]])
COST = 0.001

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("summarize_exp_c", "experiments/summarize_exp_c.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    W = np.full((5, 3), 1.0 / 3.0)
    gross, net, turn = mod.simulate(W, [0, 1, 2, 3, 4], R, COST)
    p = em.performance(net, gross)
    out = {"returns": R.tolist(), "gross": gross.tolist(), "turnover": turn.tolist(), "net": net.tolist(),
           "mean_turnover": float(np.mean(turn[1:])), "eff_n_equal": float(1 / (W[0] ** 2).sum()),
           "eff_n_uneven": float(1 / (np.array([0.5, 0.25, 0.25]) ** 2).sum()), **p}
    json.dump(out, open("results/toy_example.json", "w"), indent=1)
    print(json.dumps(p, indent=1))
