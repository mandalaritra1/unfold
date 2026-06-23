#!/usr/bin/env python3
"""Clean CMS-style 'unfolded result' comparison for the Z+jet rho data-prior test.

Re-plots, in mplhep CMS style, the nominal-response vs data-prior-response
unfolded data (per-pT normalized) with a data-prior/nominal ratio panel, for the
analysis note. Reads the study artifacts -- no re-unfolding, no ROOT -- and drops
the 0-200 GeV pT sink bin and the merged low-rho underflow bin.

    python scripts/plot_data_prior_unfolded_comparison.py

Inputs : outputs/zjet/rho/original_data_prior_test/artifacts/{mode}_data_prior_test.npz
Outputs: outputs/zjet/rho/original_data_prior_test/data_prior_unfolded_comparison_{mode}.pdf (+ .png)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "outputs/zjet/rho/original_data_prior_test"

# Spec strings copied from RHO_..._SPEC in unfold.tools.unfolder_core (avoids a
# ROOT import just to fetch two label strings).
YLABEL = r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{d\log_{10}(\rho^2)\,dp_T}$"
XLABEL = {"ungroomed": r"$\log_{10}(\rho^2)$, Ungroomed",
          "groomed":   r"$\log_{10}(\rho^2)$, Groomed"}
XLIM_LOWER = {"ungroomed": -2.5, "groomed": -4.5}

NOMINAL_COLOR = "black"
PRIOR_COLOR = "#2ca02c"
PT_MIN = 200.0  # drop the 0-200 GeV sink bin


def pt_label(lo: float, hi: float) -> str:
    if np.isfinite(hi) and hi < 13000:
        return rf"${lo:g} < p_{{T}} < {hi:g}$ GeV"
    return rf"$p_{{T}} > {lo:g}$ GeV"


def plot_mode(mode: str) -> Path:
    d = np.load(STUDY_DIR / "artifacts" / f"{mode}_data_prior_test.npz", allow_pickle=True)
    pt_edges = np.asarray(d["pt_edges"], dtype=float)
    edges_by_pt = d["gen_edges_by_pt"]
    nom, nom_e = d["nominal_normalized"], d["nominal_normalized_err"]
    wgt, wgt_e = d["weighted_normalized"], d["weighted_normalized_err"]

    # Keep pT slices above the sink bin; drop the merged low-rho underflow bin (edge[0]).
    panels = []
    for i in range(len(pt_edges) - 1):
        lo, hi = pt_edges[i], pt_edges[i + 1]
        if hi <= PT_MIN:
            continue
        edges = np.asarray(edges_by_pt[i], dtype=float)[1:]  # drop underflow bin
        centers = 0.5 * (edges[:-1] + edges[1:])
        xerr = 0.5 * np.diff(edges)
        panels.append(dict(
            lo=lo, hi=hi, centers=centers, xerr=xerr,
            nom=np.asarray(nom[i], float)[1:], nom_e=np.asarray(nom_e[i], float)[1:],
            wgt=np.asarray(wgt[i], float)[1:], wgt_e=np.asarray(wgt_e[i], float)[1:],
        ))

    # Adaptive, symmetric ratio window (never clips).
    max_dev = max(
        np.max(np.abs(np.divide(p["wgt"], p["nom"],
                                out=np.ones_like(p["wgt"]), where=p["nom"] != 0) - 1.0))
        for p in panels
    )
    half = float(np.clip(1.15 * max_dev, 0.1, 0.5))
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

            # nominal: filled black circle; data-prior: open green square on top
            # (they agree, so the open square lets the black point show through).
            ax.errorbar(p["centers"], p["nom"], yerr=p["nom_e"], xerr=p["xerr"],
                        fmt="o", ms=7, lw=1.4, color=NOMINAL_COLOR,
                        label="Nominal response")
            ax.errorbar(p["centers"], p["wgt"], yerr=p["wgt_e"], xerr=p["xerr"],
                        fmt="s", ms=10, lw=1.4, color=PRIOR_COLOR,
                        mfc="none", mew=1.8, label="Data-prior response")

            ratio = np.divide(p["wgt"], p["nom"], out=np.ones_like(p["wgt"]),
                              where=p["nom"] != 0)
            rax.errorbar(p["centers"], ratio, xerr=p["xerr"], fmt="s", ms=7, lw=1.4,
                         color=PRIOR_COLOR)
            rax.axhline(1.0, color="black", lw=1.0, ls=":")

            ymin = min(p["nom"].min(), p["wgt"].min())
            ymax = max(p["nom"].max(), p["wgt"].max())
            ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax * 1.32)  # headroom for pT label
            ax.text(0.05, 0.95, pt_label(p["lo"], p["hi"]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=18)
            ax.set_xlim(XLIM_LOWER[mode], 0.0)
            rax.set_ylim(*ratio_ylim)
            rax.set_xlabel(XLABEL[mode], fontsize=18)
            ax.tick_params(labelsize=15, labelbottom=False)
            rax.tick_params(labelsize=15)
            ax.grid(alpha=0.25)
            rax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel(YLABEL, fontsize=20)
                rax.set_ylabel("Data-prior / nominal", fontsize=13)

        hep.cms.label("Internal", data=True, ax=first_main, fontsize=18, rlabel="")
        last_main.text(1.0, 1.01, r"138 fb$^{-1}$ (13 TeV)",
                       transform=last_main.transAxes, ha="right", va="bottom", fontsize=16)
        handles, labels = first_main.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=15,
                   frameon=False, bbox_to_anchor=(0.5, 0.995))

        out = STUDY_DIR / f"data_prior_unfolded_comparison_{mode}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("groomed", "ungroomed", "both"), default="both")
    args = ap.parse_args()
    modes = ("ungroomed", "groomed") if args.mode == "both" else (args.mode,)
    for m in modes:
        print(f"[{m}] wrote {plot_mode(m)}")


if __name__ == "__main__":
    main()
