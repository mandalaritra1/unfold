#!/usr/bin/env python3
"""Regenerate the AN-24-162 self-closure figures on the arc_r2 production.

The closure figures in the AN (fig:closure-zjet-*) predate the arc_r2
binning and carry a legacy "PYTHIA7" legend label. This driver re-runs the
self-closure (PYTHIA reco unfolded through the PYTHIA response, compared to
PYTHIA truth) on the arc_r2 spec, statistics only -- no systematic
variations are unfolded, per the AN convention that the closure test is a
stability check, not an uncertainty estimate.

Run from the repository root:

    source scripts/setup_root.sh
    .venv/bin/python scripts/make_closure_plots_arc_r2.py

Outputs land in outputs/zjet/rho/closure_arc_r2/ as closure_<mode>_<i>.pdf.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import mplhep as hep

from unfold.tools.unfolder_core import Unfolder, get_spec

# plot_unfolded_fancy inherits whatever style the preceding run_all_plots
# figures set; a direct call needs the CMS style applied explicitly.
hep.style.use("CMS")

OUTPUT_DIR = REPO_ROOT / "outputs/zjet/rho/closure_arc_r2"


def main() -> None:
    spec = get_spec("zjet", "rho", "arc_r2")
    # model_envelope is a spec-level knob, so do_syst=False alone does NOT keep
    # the modelling term out of the band: the PS/HAD envelope would still be
    # re-unfolded and drawn on top of the stat band. The closure test is a
    # stability check, not an uncertainty estimate -- statistical only.
    study_spec = replace(
        spec, output_dir=str(OUTPUT_DIR) + "/", model_envelope=False,
    )
    for groomed in (False, True):
        mode = "groomed" if groomed else "ungroomed"
        print(f"\n=== closure | {mode} ===")
        unfolder = Unfolder(
            study_spec,
            groomed,
            do_syst=False,
            closure=True,
            cms_label="Internal",
            compute_jackknife_stat=False,
        )
        unfolder.plot_unfolded_fancy(show=False)
    print(f"\nWrote closure figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
