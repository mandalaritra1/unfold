#!/usr/bin/env python3
"""Reco-level distributions: data vs the two MC variants (old/new JMS/JMR).

Companion to compare_jmsjmr_unity_datamc_reco.py -- same inputs and
normalization (unit area / bin width over the shown window), but showing the
spectra themselves: data as points, arc_r2-table MC and unity-table MC as
step curves.

Run from the repository root:

    .venv/bin/python scripts/plotting/plot_jmsjmr_unity_datamc_dist.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pickle

hep.style.use("CMS")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs/zjet/rho/jmsjmr_unity_compare"

PT_LABELS = {1: "200-290 GeV", 2: "290-400 GeV", 3: ">400 GeV"}
SHOWN_FLOOR = {
    "g": {1: -3.0, 2: -3.0, 3: -3.5},
    "u": {1: -2.5, 2: -2.5, 3: -2.5},
}


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def reco_spectrum(h, ipt):
    names = [a.name for a in h.axes]
    if "dataset" in names:
        h = h[{"dataset": sum}]
    h = h[{"systematic": "nominal", "ptreco": ipt}]
    edges = np.asarray(h.axes[0].edges)
    return h.values(), np.sqrt(h.variances()), edges


def norm_shape(vals, errs, edges, floor):
    widths = np.diff(edges)
    mask = edges[:-1] >= floor - 1e-9
    total = vals[mask].sum()
    v = np.where(mask, vals / widths / total, np.nan)
    e = np.where(mask, errs / widths / total, np.nan)
    return v, e


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load(REPO_ROOT / "inputs/zjet/rho/arc_r2/data_all.pkl")
    mc_old = load(REPO_ROOT / "inputs/zjet/rho/arc_r2/pythia_all.pkl")
    mc_new = load(REPO_ROOT / "inputs/zjet/rho/jmsjmr_unity/pythia_all.pkl")

    for tag, mode in (("g", "groomed"), ("u", "ungroomed")):
        key = f"ptjet_rhojet_{tag}_reco"
        fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
        for ax, ipt in zip(axes, (1, 2, 3)):
            floor = SHOWN_FLOOR[tag][ipt]
            dv, de, edges = reco_spectrum(data[key], ipt)
            ov, _, _ = reco_spectrum(mc_old[key], ipt)
            nv, _, _ = reco_spectrum(mc_new[key], ipt)
            dvn, den = norm_shape(dv, de, edges, floor)
            ovn, _ = norm_shape(ov, np.zeros_like(ov), edges, floor)
            nvn, _ = norm_shape(nv, np.zeros_like(nv), edges, floor)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hep.histplot(ovn, edges, ax=ax, color="#5790fc", ls="--", lw=2,
                         label="MC (arc_r2 tables)")
            hep.histplot(nvn, edges, ax=ax, color="#e42536", lw=2,
                         label="MC (unity JMS/JMR)")
            good = np.isfinite(dvn)
            ax.errorbar(centers[good], dvn[good], yerr=den[good],
                        xerr=np.diff(edges)[good] / 2, fmt="o", color="k",
                        markersize=6, label="Data")
            ax.set_xlim(floor, 0)
            ax.set_ylim(0, 1.55 * np.nanmax([dvn, ovn, nvn]))
            ax.set_xlabel(rf"$\log_{{10}}(\rho^2)$, {mode}")
            ax.legend(title=rf"$p_T$ {PT_LABELS[ipt]}", fontsize=15,
                      loc="upper left")
        axes[0].set_ylabel("normalized / bin width")
        hep.cms.label("Internal", data=True, lumi=138, com=13, ax=axes[0], fontsize=18)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"datamc_dist_{mode}.{ext}", dpi=150)
        plt.close(fig)
        print(f"wrote {OUT_DIR}/datamc_dist_{mode}.pdf/.png")


if __name__ == "__main__":
    main()
