#!/usr/bin/env python3
"""Make the rho-averaged-per-pT jackknife stat convergence plot (zjet rho).

Builds the zjet rho spec Unfolder for both grooming modes (systematics off --
the jackknife re-unfolds run in _compute_stat_unc regardless) and calls only
Unfolder.plot_jackknife_convergence_pt_avg. Writes
    <spec.output_dir>/unfold/jackknife_convergence_pt_avg_{mode}.pdf

    python scripts/plot_jk_convergence_pt_avg.py [--tag original]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

import matplotlib

matplotlib.use("Agg")

from unfold.tools.unfolder_core import Unfolder, get_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="original")
    parser.add_argument("--observable", default="rho")
    args = parser.parse_args()

    spec = get_spec("zjet", args.observable, args.tag)
    for mode, groomed in (("ungroomed", False), ("groomed", True)):
        unfolder = Unfolder(spec, groomed, do_syst=False)
        unfolder.plot_jackknife_convergence_pt_avg(show=False, pt_min=200)
        out = f"{spec.output_dir}unfold/jackknife_convergence_pt_avg_{mode}.pdf"
        print(f"[zjet {args.observable} {mode}] wrote {out}")


if __name__ == "__main__":
    main()
