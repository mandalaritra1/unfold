#!/usr/bin/env python
"""mumu vs ee data shape comparison of reco-level log10(rho^2) for AN-24-162.

One square CMS panel per (grooming, reported pT bin), main + ratio subaxes;
the AN grids the per-pT panels with \includegraphics. Data-only comparison:
the MC rho histograms carry no channel axis in the arc_r2 production.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

DEFAULT_INPUT = Path("/Users/aritra/Projects/unfold/inputs/zjet/rho/arc_r2/data_all.pkl")
DEFAULT_OUTPUT_DIR = Path("/Users/aritra/Projects/AN-24-162/figures/zplusjet/data_MC/rho_channel")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"reco-level data pkl (default: {DEFAULT_INPUT})")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"directory for figures (default: {DEFAULT_OUTPUT_DIR})")
args = parser.parse_args()

INPUT = args.input.expanduser()
OUTDIR = args.output_dir.expanduser()
OUTDIR.mkdir(parents=True, exist_ok=True)

MM_COLOR = "black"
EE_COLOR = "#5790fc"

PT_LABELS = {1: r"$200 < p_T < 290$ GeV", 2: r"$290 < p_T < 400$ GeV", 3: r"$p_T > 400$ GeV"}
XLIM = {"g": (-5.0, 0.0), "u": (-4.0, 0.0)}

with open(INPUT, "rb") as f:
    data = pickle.load(f)


def channel_hist(h, channel):
    names = [str(c) for c in h.axes["dataset"]]
    if channel == "mm":
        keep = [n for n in names if "SingleMuon" in n]
    else:
        keep = [n for n in names if "SingleElectron" in n or "EGamma" in n]
    return sum(h[{"dataset": n, "systematic": "nominal"}] for n in keep)


for groom in ("g", "u"):
    h2 = data[f"ptjet_rhojet_{groom}_reco"]
    for ptbin in (1, 2, 3):
        hists = {}
        for chan in ("mm", "ee"):
            h1 = channel_hist(h2, chan)[{"ptreco": ptbin}]
            vals = h1.values()
            errs = np.sqrt(h1.variances())
            edges = h1.axes[0].edges
            widths = np.diff(edges)
            norm = (vals * widths).sum()
            hists[chan] = (vals / norm, errs / norm, edges)

        fig, (ax, rax) = plt.subplots(
            2, 1, sharex=True, layout="constrained",
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
        centers = 0.5 * (hists["mm"][2][1:] + hists["mm"][2][:-1])
        for chan, color, marker, label in (
            ("mm", MM_COLOR, "o", r"$\mu\mu$ data"),
            ("ee", EE_COLOR, "s", r"$ee$ data"),
        ):
            v, e, edges = hists[chan]
            ax.errorbar(centers, v, yerr=e, fmt=marker, color=color, label=label, markersize=7)

        vmm, emm, _ = hists["mm"]
        vee, eee, _ = hists["ee"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(vmm > 0, vee / vmm, np.nan)
            rerr = np.where(vmm > 0, ratio * np.sqrt(
                np.where(vee > 0, (eee / np.where(vee > 0, vee, 1)) ** 2, 0)
                + (emm / np.where(vmm > 0, vmm, 1)) ** 2), np.nan)
        rax.errorbar(centers, ratio, yerr=rerr, fmt="s", color=EE_COLOR, markersize=6)
        rax.axhline(1.0, color="gray", ls="--", lw=1)

        lo, hi = XLIM[groom]
        ax.set_xlim(lo, hi)
        vis = (centers > lo) & (centers < hi)
        vmax = max(np.nanmax(hists["mm"][0][vis]), np.nanmax(hists["ee"][0][vis]))
        ax.set_ylim(0, vmax * 1.5)
        rax.set_ylim(0.35, 1.65)
        rax.set_xlabel(r"$\log_{10}\rho^2$")
        ax.set_ylabel("Normalized events / bin width")
        rax.set_ylabel(r"$ee/\mu\mu$", fontsize=20)
        ax.legend(title=PT_LABELS[ptbin], loc="upper left")
        hep.cms.label("Internal", data=True, ax=ax, loc=0, rlabel=r"138 fb$^{-1}$ (13 TeV)")

        gname = "groomed" if groom == "g" else "ungroomed"
        stem = OUTDIR / f"data_channel_rho_{gname}_pt{ptbin}"
        fig.savefig(f"{stem}.pdf")
        fig.savefig(f"{stem}.png", dpi=120)
        plt.close(fig)
        print("wrote", stem)
