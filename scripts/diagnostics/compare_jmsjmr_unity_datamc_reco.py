#!/usr/bin/env python3
"""Reco-level data/MC before vs after the JMS/JMR unity fix.

Normalized-shape comparison (no backgrounds; the validation selection is
~98.5% DY-pure): Run 2 data over PYTHIA nominal, with the MC taken from the
arc_r2 production (old per-year JMS/JMR tables) and from the jmsjmr_unity
reskim (UL unity +-1%/+-2%). Data is identical in both, so any difference
between the two ratio curves is purely the response-side fix.

Run from the repository root:

    .venv/bin/python scripts/diagnostics/compare_jmsjmr_unity_datamc_reco.py
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
# reported reco-level windows (match the AN shown floors)
SHOWN_FLOOR = {
    "g": {1: -3.0, 2: -3.0, 3: -3.5},
    "u": {1: -2.5, 2: -2.5, 3: -2.5},
}


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def reco_spectrum(h, ipt):
    """(values, errors, edges) for pT slice ipt, dataset-summed, nominal."""
    names = [a.name for a in h.axes]
    if "dataset" in names:
        h = h[{"dataset": sum}]
    h = h[{"systematic": "nominal", "ptreco": ipt}]
    edges = np.asarray(h.axes[0].edges)
    return h.values(), np.sqrt(h.variances()), edges


def norm_shape(vals, errs, edges, floor):
    """Unit-area/binwidth over the shown window; bins below floor masked."""
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
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        print(f"\n===== {mode} (reco level, normalized shapes) =====")
        for ax, ipt in zip(axes, (1, 2, 3)):
            floor = SHOWN_FLOOR[tag][ipt]
            dv, de, edges = reco_spectrum(data[key], ipt)
            ov, _, _ = reco_spectrum(mc_old[key], ipt)
            nv, _, _ = reco_spectrum(mc_new[key], ipt)
            dvn, den = norm_shape(dv, de, edges, floor)
            ovn, _ = norm_shape(ov, np.zeros_like(ov), edges, floor)
            nvn, _ = norm_shape(nv, np.zeros_like(nv), edges, floor)
            r_old = dvn / ovn
            r_new = dvn / nvn
            stat = np.divide(den, dvn, out=np.zeros_like(dvn), where=dvn != 0)
            x = np.repeat(edges, 2)[1:-1]
            ax.fill_between(x, np.repeat(1 - stat, 2), np.repeat(1 + stat, 2),
                            color="0.85", label="data stat.")
            hep.histplot(r_old, edges, ax=ax, color="#5790fc", ls="--", lw=2,
                         label="Data / MC (arc_r2 tables)")
            hep.histplot(r_new, edges, ax=ax, color="#e42536", lw=2,
                         label="Data / MC (unity JMS/JMR)")
            ax.axhline(1.0, color="gray", ls=":")
            ax.set_xlim(floor, 0)
            ax.set_ylim(0.7, 1.4)
            ax.set_xlabel(rf"$\log_{{10}}(\rho^2)$, {mode}")
            ax.legend(title=rf"$p_T$ {PT_LABELS[ipt]}", fontsize=15,
                      loc="upper right")
            good = np.isfinite(r_old)
            print(f"  pt {PT_LABELS[ipt]}:")
            for b in np.where(good)[0]:
                print(f"    [{edges[b]:6.3f},{edges[b+1]:6.3f}] "
                      f"D/MC old={r_old[b]:.3f} new={r_new[b]:.3f} "
                      f"(delta {r_new[b]-r_old[b]:+.3f}, stat {stat[b]:.3f})")
        axes[0].set_ylabel("Data / MC")
        hep.cms.label("Internal", data=True, lumi=138, com=13, ax=axes[0], fontsize=18)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"datamc_reco_{mode}.{ext}", dpi=150)
        plt.close(fig)
        print(f"  wrote {OUT_DIR}/datamc_reco_{mode}.pdf/.png")


if __name__ == "__main__":
    main()
