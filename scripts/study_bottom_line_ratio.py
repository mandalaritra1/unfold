#!/usr/bin/env python3
r"""Bottom-line *ratio* diagnostic per pT slice, emitted as PNGs for the ARC deck.

The visual companion to the chi2 bar chart: each panel overlays
  Unfolded / True        (truth space, data-stat error band)
  Measured / Reco_MC     (reco space)
so the ARC can see directly that unfolding does not amplify the data-PYTHIA
ratio (bottom-line test, Figs. 45-46). Reuses the production
Unfolder.plot_bottom_line; only _finalize_plot is shimmed so each panel lands as
a PNG in the slide figs/ instead of the categorized PDF layout.

Usage:
    source scripts/setup_root.sh
    .venv/bin/python scripts/study_bottom_line_ratio.py [--merge30]
"""
from __future__ import annotations

import argparse
import sys
import types
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

from unfold.tools.unfolder_core import Unfolder, get_spec

DEFAULT_SLIDE_DIR = Path("/Users/aritra/projects/smp_jetmass_run2/review/figs")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge30", action="store_true",
                    help="collapse gen binning below log10(rho^2) = -3.0")
    ap.add_argument("--slide-dir", type=Path, default=DEFAULT_SLIDE_DIR,
                    help=f"optional figure handoff directory (default: {DEFAULT_SLIDE_DIR})")
    ap.add_argument("--no-slide-copy", action="store_true",
                    help="keep figures only in the generated output directory")
    args = ap.parse_args()
    slide_dir = None if args.no_slide_copy else args.slide_dir.expanduser()

    tag_suffix = "_merge30" if args.merge30 else ""
    out = REPO / "outputs" / "zjet" / "rho" / ("bottom_line_ratio" + tag_suffix)
    out.mkdir(parents=True, exist_ok=True)
    spec = get_spec("zjet", "rho", "original")
    if args.merge30:
        spec = replace(spec, gen_merge_below=-3.0)
    spec = replace(spec, output_dir=str(out) + "/")

    for groomed in (True, False):
        mode = "groomed" if groomed else "ungroomed"
        print(f"\n=== {mode}{' (gen merged < -3.0)' if args.merge30 else ''} ===")
        u = Unfolder(spec, groomed, do_syst=False, cms_label="Internal",
                     compute_jackknife_stat=False)

        # Shim _finalize_plot so each per-pT panel is written as a PNG we control.
        def finalize(self, save_path=None, show=False, fig=None):
            if save_path is not None and fig is not None:
                idx = int(Path(save_path).stem.rsplit("_", 1)[1])  # 0,1,2 -> pt1,2,3
                png = out / f"bottomline_ratio{tag_suffix}_{mode}_pt{idx + 1}.png"
                fig.savefig(png, dpi=170, bbox_inches="tight", pad_inches=0.1)
                print("WROTE", png.name)
                if slide_dir is not None and slide_dir.is_dir():
                    (slide_dir / png.name).write_bytes(png.read_bytes())
            plt.close(fig if fig is not None else plt.gcf())

        u._finalize_plot = types.MethodType(finalize, u)
        u.plot_bottom_line(show=False)


if __name__ == "__main__":
    main()
