#!/usr/bin/env python3
"""Clean CMS-style 'unfolded result' comparison for the Z+jet rho data-prior test.

Re-plots, in mplhep CMS style, the nominal-response vs data-prior-response
unfolded data (per-pT normalized) with a data-prior/nominal ratio panel, for the
analysis note. Reads the study artifacts -- no re-unfolding, no ROOT -- and shows
only the reported space: the low-pT migration sink bin is dropped, and each pT
panel starts at its own shown rho floor so the hidden low-rho buffer bins and the
merged underflow bin never appear.

    python scripts/plot_data_prior_unfolded_comparison.py --tag arc_r2

Inputs : outputs/zjet/rho/<tag>_data_prior_test/artifacts/{mode}_data_prior_test.npz
Outputs: outputs/zjet/rho/<tag>_data_prior_test/data_prior_unfolded_comparison_{mode}.pdf (+ .png)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unfold.utils.cms_plot import save_cms_label_flavors

# Spec strings copied from RHO_..._SPEC in unfold.tools.unfolder_core (avoids a
# ROOT import just to fetch two label strings).
YLABEL = r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{d\log_{10}(\rho^2)\,dp_T}$"
XLABEL = {"ungroomed": r"$\log_{10}(\rho^2)$, Ungroomed",
          "groomed":   r"$\log_{10}(\rho^2)$, Groomed"}

# Per-pT lower edge of the SHOWN (reported) rho space, aligned with pt_edges.
# Mirrors spec.bl_shown_floors_* / xlim_lower_* in unfold.tools.unfolder_core;
# arc_r2 crops the groomed display at the per-slice 5 GeV mass floor and hides
# the -5..-3.5 buffer bins entirely.
SHOWN_FLOORS = {
    "original": {"ungroomed": [-2.5] * 4, "groomed": [-4.5] * 4},
    "arc_r2":   {"ungroomed": [-2.5] * 4, "groomed": [-2.5, -3.0, -3.0, -3.5]},
    # Approved ARC phase space: UL unity JMS/JMR inputs, and the response-limited
    # first shown groomed bin of the 400-inf slice excluded (-3.5 -> -3.0).
    "jmsjmr_unity_groomed400_floor3":
                {"ungroomed": [-2.5] * 4, "groomed": [-2.5, -3.0, -3.0, -3.0]},
}

NOMINAL_COLOR = "black"
PRIOR_COLOR = "#2ca02c"
BAND_COLOR = "#9e9e9e"
BAND_ALPHA = 0.45
PT_MIN = 200.0  # drop the low-pT migration sink bin (0-200 original, 185-200 arc_r2)


def pt_label(lo: float, hi: float) -> str:
    if np.isfinite(hi) and hi < 13000:
        return rf"${lo:g} < p_{{T}} < {hi:g}$ GeV"
    return rf"$p_{{T}} > {lo:g}$ GeV"


def plot_mode(mode: str, tag: str, study_dir: Path, cms_label: str) -> Path:
    d = np.load(study_dir / "artifacts" / f"{mode}_data_prior_test.npz", allow_pickle=True)
    pt_edges = np.asarray(d["pt_edges"], dtype=float)
    edges_by_pt = d["gen_edges_by_pt"]
    nom, nom_e = d["nominal_normalized"], d["nominal_normalized_err"]
    wgt, wgt_e = d["weighted_normalized"], d["weighted_normalized_err"]
    nom_stat = d["nominal_stat"]
    floors = SHOWN_FLOORS[tag][mode]

    # Keep pT slices above the sink bin; within a slice keep only bins at or
    # above its shown floor. That drops the merged low-rho underflow bin in both
    # productions and, on arc_r2, also the hidden 0.25-wide buffer bins.
    panels = []
    for i in range(len(pt_edges) - 1):
        lo, hi = pt_edges[i], pt_edges[i + 1]
        if hi <= PT_MIN:
            continue
        all_edges = np.asarray(edges_by_pt[i], dtype=float)
        keep = all_edges[:-1] >= floors[i] - 1e-9
        edges = np.append(all_edges[:-1][keep], all_edges[-1])
        centers = 0.5 * (edges[:-1] + edges[1:])
        xerr = 0.5 * np.diff(edges)
        panels.append(dict(
            lo=lo, hi=hi, floor=floors[i], centers=centers, xerr=xerr, edges=edges,
            nom=np.asarray(nom[i], float)[keep], nom_e=np.asarray(nom_e[i], float)[keep],
            wgt=np.asarray(wgt[i], float)[keep], wgt_e=np.asarray(wgt_e[i], float)[keep],
            stat=np.asarray(nom_stat[i], float)[keep],
        ))

    # Adaptive, symmetric ratio window: contains every shift point (the claim)
    # and the whole statistical band it is judged against.
    def rel(vals, ref):
        return np.abs(np.divide(vals, ref, out=np.zeros_like(vals), where=ref != 0))

    max_dev = max(
        max(float(np.max(np.abs(np.divide(p["wgt"], p["nom"],
                                          out=np.ones_like(p["wgt"]),
                                          where=p["nom"] != 0) - 1.0))),
            float(np.max(rel(p["stat"], p["nom"]))))
        for p in panels
    )
    half = float(np.clip(1.15 * max_dev, 0.05, 0.5))
    ratio_ylim = (1 - half, 1 + half)

    n = len(panels)
    with plt.style.context(hep.style.CMS):
        fig = plt.figure(figsize=(5.8 * n, 6.4))
        outer = fig.add_gridspec(1, n, wspace=0.34, left=0.075, right=0.985,
                                 top=0.88, bottom=0.12)
        first_main = last_main = None
        for c, p in enumerate(panels):
            sub = outer[0, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(sub[0])
            rax = fig.add_subplot(sub[1], sharex=ax)
            if c == 0:
                first_main = ax
            last_main = ax

            # STATISTICAL uncertainty of the nominal measurement, as a band
            # behind the markers -- the yardstick for the prior shift. Not the
            # total: the two unfoldings share the same data and the same
            # detector systematics, so those cancel in the ratio, and the
            # systematic budget already carries a PS/HAD modelling envelope,
            # which would make the comparison circular as well as too generous.
            ax.stairs(p["nom"] + p["stat"], p["edges"],
                      baseline=p["nom"] - p["stat"], fill=True,
                      color=BAND_COLOR, alpha=BAND_ALPHA, lw=0, zorder=0,
                      label="Nominal stat. unc.")

            # nominal: filled black circle; data-prior: open green square on top
            # (they agree, so the open square lets the black point show through).
            # Marker error bars are statistical only.
            ax.errorbar(p["centers"], p["nom"], yerr=p["stat"], xerr=p["xerr"],
                        fmt="o", ms=7, lw=1.4, color=NOMINAL_COLOR, zorder=3,
                        label="Nominal response")
            ax.errorbar(p["centers"], p["wgt"], yerr=p["wgt_e"], xerr=p["xerr"],
                        fmt="s", ms=10, lw=1.4, color=PRIOR_COLOR, zorder=4,
                        mfc="none", mew=1.8, label="Data-prior response")

            ratio = np.divide(p["wgt"], p["nom"], out=np.ones_like(p["wgt"]),
                              where=p["nom"] != 0)
            # The two unfolds use the SAME data, so their uncertainties are
            # almost fully correlated and largely cancel in the ratio -- adding
            # them in quadrature on the points would badly overstate the spread.
            # Show instead the nominal measurement's relative statistical
            # uncertainty as a band: the claim is that the prior shift is small
            # compared to it, which the reader can then check by eye.
            rel_stat = rel(p["stat"], p["nom"])
            rax.stairs(1.0 + rel_stat, p["edges"], baseline=1.0 - rel_stat, fill=True,
                       color=BAND_COLOR, alpha=BAND_ALPHA, lw=0, zorder=0)
            rax.errorbar(p["centers"], ratio, xerr=p["xerr"], fmt="s", ms=7, lw=1.4,
                         color=PRIOR_COLOR, zorder=3)
            rax.axhline(1.0, color="black", lw=1.0, ls=":")

            ymin = min(p["nom"].min(), p["wgt"].min(), (p["nom"] - p["stat"]).min())
            ymax = max(p["nom"].max(), p["wgt"].max(), (p["nom"] + p["stat"]).max())
            ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax * 1.32)  # headroom for pT label
            ax.text(0.05, 0.95, pt_label(p["lo"], p["hi"]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=18)
            ax.set_xlim(p["floor"], 0.0)
            rax.set_ylim(*ratio_ylim)
            rax.set_xlabel(XLABEL[mode], fontsize=18)
            ax.tick_params(labelsize=15, labelbottom=False)
            rax.tick_params(labelsize=15)
            ax.grid(alpha=0.25)
            rax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel(YLABEL, fontsize=20)
                rax.set_ylabel("Data-prior / nominal", fontsize=13)

        hep.cms.label(cms_label, data=True, ax=first_main, fontsize=18, rlabel="")
        last_main.text(1.0, 1.01, r"138 fb$^{-1}$ (13 TeV)",
                       transform=last_main.transAxes, ha="right", va="bottom", fontsize=16)
        handles, labels = first_main.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=15,
                   frameon=False, bbox_to_anchor=(0.5, 0.995))

        out = study_dir / f"data_prior_unfolded_comparison_{mode}.pdf"
        save_cms_label_flavors(fig, out, cms_label, bbox_inches="tight")
        save_cms_label_flavors(fig, out.with_suffix(".png"), cms_label,
                               dpi=200, bbox_inches="tight")
        plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("groomed", "ungroomed", "both"), default="both")
    ap.add_argument("--tag", default="arc_r2", choices=tuple(SHOWN_FLOORS),
                    help="Z+jet rho production tag; selects the study directory "
                         "and the per-pT shown rho floors.")
    ap.add_argument("--cms-label", default="Preliminary")
    args = ap.parse_args()
    study_dir = REPO_ROOT / f"outputs/zjet/rho/{args.tag}_data_prior_test"
    modes = ("ungroomed", "groomed") if args.mode == "both" else (args.mode,)
    for m in modes:
        print(f"[{m}] wrote {plot_mode(m, args.tag, study_dir, args.cms_label)}")


if __name__ == "__main__":
    main()
