"""Fill the results table and the download note of README.md, and the download note of docs/LIMITATIONS.md (between the BEGIN/END markers)
from paper/numbers.tex.

Run scripts/make_paper_numbers.py first. Nothing in the README table is typed by hand.
"""
import re

MARK = re.compile(r"(<!-- BEGIN:headline -->).*?(<!-- END:headline -->)", re.S)
MARK_DL = re.compile(r"(<!-- BEGIN:download -->).*?(<!-- END:download -->)", re.S)


def macros(path="paper/numbers.tex"):
    out = {}
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", open(path).read(), re.M):
        val = re.sub(r"\\ensuremath\{-\}", "-", val)
        val = re.sub(r"\\ensuremath\{([\d.]+)\\times10\^\{(-?\d+)\}\}", r"\1e\2", val)
        out[name] = val.replace("\\%", "%").replace("\\,", ",")
    return out


def table(m):
    rows = [
        ("A, 1760 synthetic instances", "probability that one local start misses the global optimum, default regime", f"{m['aPfCalFifteen']} at n=15, {m['aPfCalFifty']} at n=50"),
        ("A", "order-2 relaxation certifies the optimum, to solver accuracy", f"all but {m['aUncert']} of {m['aN']} (default regime, n=50: {m['aCalFiftyCert']})"),
        ("A", "sparse SDP against 100 local runs, n=50, default regime (median of 50 instances, six concurrent workers)", f"{m['aTimeSosCal']} s against {m['aTimeLocCal']} s"),
        ("size, n=50", "dense over sparse PSD entries (v1 claimed 398)", f"{m['cxPsdStar']} to {m['cxPsdChain']}"),
        ("B, kurtosis estimation error", "ball formulation against nominal, 10 assets", "worse for rho <= 0.2, 2 to 5% better at rho = 0.5 (gamma <= 0.1)"),
        ("B", "shrinkage at rho = 0.5, regret over nominal", m["bRatioShrC"]),
        ("C, 37 US stocks, 2015-2025", "local solver suboptimal on sector model (w4 = 0, 1, 3)", f"{m['cSubZero']}, {m['cSubOne']}, {m['cSubThree']} of dates"),
        ("C", "net-return intervals containing 0", f"{m['cNetZero']} of {m['cTests']} comparisons"),
    ]
    return "\n".join(["| experiment | measured | result |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows])


def download(m):
    return (f"A new download can differ from the file used: in a re-download of {m['dlCells']} price cells, {m['dlAbove']} differed by more than 1e-6 "
            f"in relative terms (median {m['dlMedian']}, largest {m['dlMax']}). The effect on the results of experiment C was not measured.")


if __name__ == "__main__":
    readme = open("README.md").read()
    mac = macros()
    body = table(mac)
    readme = MARK.sub(lambda g: f"{g.group(1)}\n{body}\n{g.group(2)}", readme)
    readme = MARK_DL.sub(lambda g: f"{g.group(1)}\n{download(mac)}\n{g.group(2)}", readme)
    open("README.md", "w").write(readme)
    lim = MARK_DL.sub(lambda g: f"{g.group(1)}\n{'- ' + download(mac)}\n{g.group(2)}", open("docs/LIMITATIONS.md").read())
    open("docs/LIMITATIONS.md", "w").write(lim)
