#!/usr/bin/env python3
"""Groomed rho with gen merged below -3.0: regularized vs unregularized overlay.

ARC follow-up. Collapses the groomed gen binning below log10(rho^2) = -3.0 into a
single low-rho bin (via ObservableSpec.gen_merge_below), then overlays the
unregularized (tau=0) and ratio-curvature (L-curve) unfolded spectra per pT
slice, with EMatrix statistical errors. Reuses the production TUnfold machinery
through study_regularization_grid.run().

Usage: source scripts/setup_root.sh && python scripts/study_merge30_reg.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# reg mode -> (study_regularization_grid mode key, legend label, file tag)
REG_CHOICES = {
    "ratio_curv": ("ratio_curv", "ratio-curv (curvature of $x/x_{MC}$)", "ratiocurv"),
    "curvature": ("curvature", "curvature of distribution $x$", "curv"),
    "derivative": ("derivative", "derivative of distribution $x$", "deriv"),
    "size": ("size", "size of distribution $x$", "size"),
}

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

import ROOT  # noqa: F401  (study_regularization_grid sets up the ROOT decls)
from unfold.tools.unfolder_core import Unfolder, get_spec
import study_regularization_grid as G
from study_regularization_rho import normalize

OUT = REPO / "outputs" / "zjet" / "rho" / "validation"
OUT.mkdir(parents=True, exist_ok=True)


def norm_cov(y, cov, edges_by_pt):
    """Per-pT-slice normalization to a density, with covariance propagated."""
    y = np.asarray(y, float); cov = np.asarray(cov, float)
    val = np.zeros_like(y); J = np.zeros((y.size, y.size))
    off = 0
    for e in edges_by_pt:
        n = len(e) - 1; w = np.diff(np.asarray(e, float)); idx = np.arange(off, off + n)
        b = y[idx]; s = b.sum()
        if s > 0:
            val[idx] = b / w / s
            for a, gi in enumerate(idx):
                for c, gj in enumerate(idx):
                    d = 1.0 if a == c else 0.0
                    J[gi, gj] = (d * s - b[a]) / (s * s) / w[a]
        off += n
    return val, J @ cov @ J.T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reg-mode", choices=list(REG_CHOICES), default="ratio_curv")
    args = ap.parse_args()
    g_mode, reg_label, ftag = REG_CHOICES[args.reg_mode]

    spec = replace(get_spec("zjet", "rho", "original"), gen_merge_below=-3.0,
                   output_dir=str(OUT) + "/merge30_reg_compare/")
    u = Unfolder(spec, groomed=True, do_syst=False, compute_jackknife_stat=False)
    print(f"reg mode = {g_mode}")
    print("groomed gen edges (merged < -3.0):", u.gen_edges_by_pt[1])

    resp = u.mosaic_dict["nominal"]; misses = u.misses_2d
    true_flat = resp.sum(axis=0) + misses
    meas = u._apply_fake_correction(np.array(u.mosaic_2d, float), "nominal", False, False)
    meas_self = np.asarray(u.mosaic.sum(axis=1), float)
    var = u.corrected_measured_variances

    # data unfolds: unregularized (tau=0) and ratio-curvature (L-curve)
    y0, c0, _ = G.run(u, resp, misses, true_flat, meas, mode="none", tau=0.0,
                      name="d_none", variances=var)
    yr, cr, tau = G.run(u, resp, misses, true_flat, meas, mode=g_mode,
                        scan="lcurve", name="d_reg", variances=var)
    # self-closure (unfold the MC's own matched reco -> should return MC truth)
    ys0, _, _ = G.run(u, resp, misses, true_flat, meas_self, mode="none", tau=0.0, name="s_none")
    ysr, _, _ = G.run(u, resp, misses, true_flat, meas_self, mode=g_mode, tau=tau, name="s_reg")

    n0, cn0 = norm_cov(y0, c0, u.gen_edges_by_pt); e0 = np.sqrt(np.clip(np.diag(cn0), 0, None))
    nr, cnr = norm_cov(yr, cr, u.gen_edges_by_pt); er = np.sqrt(np.clip(np.diag(cnr), 0, None))
    ntrue = normalize(u, true_flat)
    sc0 = normalize(u, ys0); scr = normalize(u, ysr)

    starts, counts, off = [], [], 0
    for e in u.gen_edges_by_pt:
        starts.append(off); counts.append(len(e) - 1); off += len(e) - 1
    rel0, relr, cl0, clr = [], [], [], []
    for ipt in range(1, len(starts)):
        s, c = starts[ipt], counts[ipt]
        for j in range(s + 1, s + c):           # skip merged tail bin
            if n0[j] > 0: rel0.append(e0[j] / n0[j])
            if nr[j] > 0: relr.append(er[j] / nr[j])
        for j in range(s, s + c):
            if ntrue[j] > 0:
                cl0.append(abs(sc0[j] / ntrue[j] - 1)); clr.append(abs(scr[j] / ntrue[j] - 1))
    red = 100.0 * (1.0 - np.median(relr) / np.median(rel0))
    print(f"L-curve tau = {tau:.4g}")
    print(f"stat reduction (median rel. err, reported bulk) = {red:.1f}%")
    print(f"self-closure max |x/true-1|: unreg = {max(cl0)*100:.3f}%  reg = {max(clr)*100:.3f}%")

    pt = [int(x) for x in u.pt_edges]
    fig, axes = plt.subplots(1, 3, figsize=(31, 10.6), layout="constrained")
    for k, ipt in enumerate(range(1, 4)):
        ax = axes[k]; s, c = starts[ipt], counts[ipt]; idx = np.arange(s, s + c)
        e = np.asarray(u.gen_edges_by_pt[ipt], float); ctr = 0.5 * (e[:-1] + e[1:]); w = np.diff(e)
        hep.histplot(ntrue[idx], e, ax=ax, color="0.55", ls=":", lw=2, label="PYTHIA8 gen")
        ax.errorbar(ctr - 0.05, n0[idx], yerr=e0[idx], xerr=0.5 * w, fmt="o", color="#1f2937",
                    ms=9, capsize=0, label=r"unregularized ($\tau=0$)")
        ax.errorbar(ctr + 0.05, nr[idx], yerr=er[idx], fmt="s", color="#c0392b", ms=9,
                    mfc="none", mew=2.4, label=rf"{reg_label} ($\tau={tau:.2g}$)")
        hi = pt[ipt + 1] if ipt + 1 < len(pt) - 1 else None
        rl = f"{pt[ipt]}-{hi} GeV" if hi else f"{pt[ipt]}+ GeV"
        hep.cms.label("Preliminary", data=True, rlabel="138 fb$^{-1}$ (13 TeV)", ax=ax, loc=0)
        ax.text(0.96, 0.93, rl, transform=ax.transAxes, fontsize=26, ha="right", va="top")
        ax.set_xlabel(r"$\log_{10}(\rho^2)$, Groomed")
        ax.set_xlim(-7.3, 0.3)
        ymax = max((n0[idx] + e0[idx]).max(), (nr[idx] + er[idx]).max())
        ax.set_ylim(0, ymax * 1.55)
        if k == 0:
            ax.set_ylabel(r"$\frac{1}{\sigma}\,\frac{\mathrm{d}\sigma}{\mathrm{d}\log_{10}(\rho^2)}$")
            ax.legend(loc="upper left", fontsize=20, frameon=False)
            ax.text(0.045, 0.52,
                    f"gen merged below $-3.0$\nstat. reduction {red:.1f}%\n"
                    f"self-closure max {max(clr)*100:.1f}%",
                    transform=ax.transAxes, fontsize=18)
    out = OUT / f"merge30_{ftag}_vs_unreg_groomed.png"
    fig.savefig(out, dpi=110)
    print("WROTE", out)


if __name__ == "__main__":
    main()
