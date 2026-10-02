import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_main(*args):
    return subprocess.run([sys.executable, str(ROOT / "main.py"), *args], capture_output=True, text=True,
                          cwd=ROOT, timeout=120).stdout


def test_demo_certifies_the_two_asset_benchmark():
    out = run_main("--mode", "demo")
    assert "certified=True" in out
    x = [float(v) for v in re.search(r"x_hat = \[([^\]]+)\]", out).group(1).split()]
    assert abs(x[0] - 0.6058) < 1e-3 and abs(x[1] - 0.3442) < 1e-3
