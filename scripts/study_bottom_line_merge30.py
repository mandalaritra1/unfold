#!/usr/bin/env python3
r"""Bottom-line chi2 summary with gen merged below log10(rho^2) = -3.0.

ARC U1 follow-up. The apparent unfolded/truth deviation is driven by the low-rho
double-peak region of the groomed mass. Collapsing the gen binning below
log10(rho^2) = -3.0 into a single low-rho bin (ObservableSpec.gen_merge_below)
tests whether the bottom-line inequality chi2_unfold <= chi2_smeared still holds
under the coarser low-rho truth binning, and how the unfolded chi2 moves.

The bar chart is byte-for-byte the production plot_bottom_line_chi2_summary
layout (copied here only so the PNG lands directly in the ARC slide figs with a
merge-specific filename, instead of the method's categorized PDF).

Usage:
    source scripts/setup_root.sh
    .venv/bin/python scripts/study_bottom_line_merge30.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.CMS)

from unfold.tools.unfolder_core import Unfolder, get_spec

MERGE_BELOW = -3.0
DEFAULT_OUTPUT_DIR = REPO / "outputs" / "zjet" / "rho" / "bottom_line_merge30"
DEFAULT_SLIDE_DIR = Path("/Users/aritra/projects/smp_jetmass_run2/review/figs")


def plot(u, rows, glob, save_png):
    """Grouped bar chart of chi2_smeared vs chi2_unfold per pT slice + Global."""
    labels, groups = [], []
    for r in rows:
        labels.append(f"{r['pt_lo']}-{r['pt_hi']}" if r["pt_hi"] else f"{r['pt_lo']}-∞")
        groups.append(r)
    labels.append("Global")
    groups.append(glob)

    x = np.arange(len(labels))
    w = 0.38
    c_sm = [g["smeared"]["chi2"] for g in groups]
    c_unf = [g["unfolded"]["chi2"] for g in groups]
    fig, ax = plt.subplots(figsize=(12, 9))
    b1 = ax.bar(x - w / 2, c_sm, w, color="#e42536", alpha=0.85,
                label=r"$\chi^2_\mathrm{smeared}$  (reco space, data vs folded PYTHIA)")
    b2 = ax.bar(x + w / 2, c_unf, w, color="#5790fc", alpha=0.9,
                label=r"$\chi^2_\mathrm{unfold}$  (truth space, unfolded vs PYTHIA gen)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\chi^2$ vs PYTHIA8 (data stat only)")
    ax.set_xlabel(r"$p_T$ slice (GeV)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for bars in (b1, b2):
        key = "smeared" if bars is b1 else "unfolded"
        for rect, g in zip(bars, groups):
            ax.annotate(f"ndf={g[key]['ndof']}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=9)
    passed = all(g["unfolded"]["chi2"] <= g["smeared"]["chi2"] + 1e-9 for g in groups)
    mode = "Groomed" if u.groomed else "Ungroomed"
    ax.legend(
        title=(f"{mode}: bottom-line test  "
               r"$\chi^2_\mathrm{unfold}\leq\chi^2_\mathrm{smeared}$  "
               + (r"$\checkmark$ PASS" if passed else r"$\times$ FAIL")
               + "\n(gen merged below $\\log_{10}(\\rho^2)=-3.0$)"),
        loc="upper right",
    )
    ax.set_ylim(top=max(c_sm) * 4)
    hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(), fontsize=20)
    fig.savefig(save_png, dpi=175, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("WROTE", save_png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"generated-output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--slide-dir", type=Path, default=DEFAULT_SLIDE_DIR,
                        help=f"optional figure handoff directory (default: {DEFAULT_SLIDE_DIR})")
    parser.add_argument("--no-slide-copy", action="store_true",
                        help="keep figures only in the generated output directory")
    args = parser.parse_args()
    out_dir = args.output_dir.expanduser()
    slide_dir = None if args.no_slide_copy else args.slide_dir.expanduser()

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = replace(get_spec("zjet", "rho", "original"),
                   gen_merge_below=MERGE_BELOW, output_dir=str(out_dir) + "/")
    for groomed in (True, False):
        mode = "groomed" if groomed else "ungroomed"
        print(f"\n=== {mode}  (gen merged < {MERGE_BELOW}) ===")
        u = Unfolder(spec, groomed, do_syst=False, cms_label="Internal",
                     compute_jackknife_stat=False)
        print("gen edges (merged, pt bin 1):", u.gen_edges_by_pt[1])
        rows, glob = u.bottom_line_test_by_pt()
        for r in rows:
            lab = f"{r['pt_lo']}-{r['pt_hi']}" if r["pt_hi"] else f"{r['pt_lo']}-inf"
            print(f"  {lab:>10}: smeared {r['smeared']['chi2']:8.1f}/{r['smeared']['ndof']:<3d}"
                  f"  unfold {r['unfolded']['chi2']:8.1f}/{r['unfolded']['ndof']:<3d}"
                  f"  {'PASS' if r['unfolded']['chi2'] <= r['smeared']['chi2'] + 1e-9 else 'FAIL'}")
        print(f"  {'GLOBAL':>10}: smeared {glob['smeared']['chi2']:8.1f}/{glob['smeared']['ndof']:<3d}"
              f"  unfold {glob['unfolded']['chi2']:8.1f}/{glob['unfolded']['ndof']:<3d}"
              f"  {'PASS' if glob['unfolded']['chi2'] <= glob['smeared']['chi2'] + 1e-9 else 'FAIL'}")
        png = out_dir / f"bottomline_chi2_merge30_{mode}.png"
        plot(u, rows, glob, png)
        if slide_dir is not None and slide_dir.is_dir():
            slide_png = slide_dir / f"bottomline_chi2_merge30_{mode}.png"
            slide_png.write_bytes(png.read_bytes())
            print("COPIED ->", slide_png)


if __name__ == "__main__":
    main()
