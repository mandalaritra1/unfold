#!/usr/bin/env python3
"""Nominal-only comparison: arc_r2 (old per-year JMS/JMR) vs jmsjmr_unity.

Unfolds the same data through the two response matrices (do_syst=False, no
jackknife) and plots, per pT bin, the ratio of the normalized unfolded
spectra (unity / arc_r2), with the arc_r2 data statistical uncertainty as a
grey band for scale. Numbers are also printed.

Run from the repository root:

    source scripts/setup_root.sh
    .venv/bin/python scripts/compare_jmsjmr_unity_nominal.py
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from unfold.tools.unfolder_core import Unfolder, get_spec, unflatten_gen_by_pt

hep.style.use("CMS")

OUT_DIR = REPO_ROOT / "outputs/zjet/rho/jmsjmr_unity_compare"


def normalized_by_pt(unfolder):
    """Per-pT normalized (unit area over shown bins, /binwidth) spectra + stat."""
    y = unflatten_gen_by_pt(unfolder.y_unf, unfolder.gen_edges_by_pt)
    e = unflatten_gen_by_pt(unfolder.ye_unf, unfolder.gen_edges_by_pt)
    out = {}
    # skip the 185-200 migration buffer (index 0): not reported in the AN
    for i in range(1, len(unfolder.pt_edges) - 1):
        widths = np.diff(unfolder.gen_edges_by_pt[i])
        total = y[i].sum()
        out[i] = (y[i] / widths / total, e[i] / widths / total, unfolder.gen_edges_by_pt[i])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for groomed in (True, False):
        mode = "groomed" if groomed else "ungroomed"
        results = {}
        for tag in ("arc_r2", "jmsjmr_unity"):
            u = Unfolder(get_spec("zjet", "rho", tag), groomed, do_syst=False,
                         cms_label="Internal", compute_jackknife_stat=False)
            results[tag] = (normalized_by_pt(u), u)
        old, uref = results["arc_r2"]
        new, _ = results["jmsjmr_unity"]

        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        print(f"\n===== {mode} =====")
        for ax, i in zip(axes, sorted(old)):
            vo, eo, edges = old[i]
            vn, _, _ = new[i]
            ratio = np.divide(vn, vo, out=np.ones_like(vo), where=vo != 0)
            stat = np.divide(eo, vo, out=np.zeros_like(vo), where=vo != 0)
            x = np.repeat(edges, 2)[1:-1]
            ax.fill_between(x, np.repeat(1 - stat, 2), np.repeat(1 + stat, 2),
                            color="0.8", label="arc_r2 data stat.")
            hep.histplot(ratio, edges, ax=ax, color="#e42536", lw=2,
                         label="unity / arc_r2 (nominal)")
            ax.axhline(1.0, color="gray", ls="--")
            lo = int(uref.pt_edges[i])
            hi = int(uref.pt_edges[i + 1]) if i + 1 < len(uref.pt_edges) - 1 else None
            title = f"{lo}-{hi} GeV" if hi else f">{lo} GeV"
            ax.set_xlim(*uref._observable_xlim(i))
            ax.set_ylim(0.7, 1.3)
            ax.set_xlabel(uref._observable_label())
            ax.legend(title=rf"$p_T$ {title}", fontsize=16, loc="upper right")
            shown = uref._shown_gen_mask(i)
            print(f"  pt {title}:")
            for b in np.where(shown)[0]:
                print(f"    bin [{edges[b]:6.2f},{edges[b+1]:6.2f}] : "
                      f"ratio-1 = {ratio[b]-1:+.4f}   stat = {stat[b]:.4f}")
        axes[0].set_ylabel("unity / arc_r2")
        hep.cms.label("Internal", data=True, lumi=138, com=13, ax=axes[0], fontsize=18)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"nominal_ratio_{mode}.{ext}", dpi=150)
        plt.close(fig)
        print(f"  wrote {OUT_DIR}/nominal_ratio_{mode}.pdf/.png")


if __name__ == "__main__":
    main()
