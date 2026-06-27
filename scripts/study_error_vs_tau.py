#!/usr/bin/env python3
r"""Unfolded-bin statistical error vs regularization strength (tau).

Quantifies the bias/variance trade-off of the ratio-curvature regularization:
how much do the TUnfold-propagated (EMatrix) statistical errors on the unfolded
bins shrink as tau increases, relative to the unregularized (tau=0) production
result?

For each grooming mode the inputs are loaded once and the nominal data spectrum
is re-unfolded in place at a grid of tau values (regularization='none' for the
tau=0 baseline, 'ratio_curvature' otherwise). The per-bin fractional error is
read from the propagated covariance:

    input-stat : sqrt(diag(cov_data_np)) / |y_unf|     (GetEmatrixInput, data stat)
    total-stat : sqrt(diag(cov_np))      / |y_unf|     (GetEmatrixTotal, input+matrix)

A table is printed (per reported pT slice: median over reported bins and the
worst low-rho tail bin) and a plot of the median total-stat error vs tau is
written to outputs/zjet/validation/error_vs_tau_<mode>.{pdf,png}, with the
tau=0 baseline drawn as a dashed line per slice and the L-curve tau marked.

Usage:
    source scripts/setup_root.sh
    python scripts/study_error_vs_tau.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

TAU_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def _offsets(edges_by_pt):
    counts = [len(e) - 1 for e in edges_by_pt]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    return starts, counts


def frac_errors(unf, regularization, tau):
    """Re-unfold the nominal data in place and return per-bin fractional errors."""
    unf.regularization = regularization
    unf.tau = tau
    unf._perform_unfold(systematic="nominal")
    val = np.abs(np.asarray(unf.y_unf, dtype=float))
    inp = np.sqrt(np.clip(np.diag(np.asarray(unf.cov_data_np, float)), 0.0, None))
    tot = np.sqrt(np.clip(np.diag(np.asarray(unf.cov_np, float)), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        f_inp = np.where(val > 0, inp / val, np.nan)
        f_tot = np.where(val > 0, tot / val, np.nan)
    return f_inp, f_tot


def per_slice_summary(unf, frac):
    """Return {pt_label: (median_reported, lowrho_tail_bin)} fractional error."""
    gstart, gcount = _offsets(unf.gen_edges_by_pt)
    out = {}
    for i in unf._reported_pt_indices():
        if unf.pt_edges[i] < 200:
            continue
        # reported bins exclude the [-10,-4.5] underflow sink (index 0)
        idx = list(range(gstart[i] + 1, gstart[i] + gcount[i]))
        hi = int(unf.pt_edges[i + 1]) if i + 1 < len(unf.pt_edges) - 1 else None
        label = f"{int(unf.pt_edges[i])}-{hi}" if hi else f"{int(unf.pt_edges[i])}-inf"
        vals = frac[idx]
        out[label] = (float(np.nanmedian(vals)), float(vals[0]))  # idx[0] = lowest reported rho
    return out


def run_mode(groomed, taus):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from dataclasses import replace
    from unfold.tools.unfolder_core import Unfolder, get_spec

    reg_spec = get_spec("zjet", "rho", "original_jacobian_reg")
    # Build once (L-curve scan runs here since reg_spec.tau is None); inputs and
    # the ratio-curvature L conditions are then reused for every tau.
    unf = Unfolder(reg_spec, groomed, do_syst=False, compute_jackknife_stat=False)
    lcurve_tau = float(unf.tau)

    points = [("none", 0.0, "0 (no reg)")] + [("ratio_curvature", t, f"{t:g}") for t in taus]
    # insert the L-curve tau in order
    points.append(("ratio_curvature", lcurve_tau, f"{lcurve_tau:.2g} (L-curve)"))
    points.sort(key=lambda p: p[1])

    results = []  # (label, tau, summary_total, summary_input)
    for reg, tau, label in points:
        f_inp, f_tot = frac_errors(unf, reg, tau)
        results.append((label, tau, per_slice_summary(unf, f_tot),
                        per_slice_summary(unf, f_inp)))

    mode = "groomed" if groomed else "ungroomed"
    slices = list(results[0][2].keys())

    print(f"\n=== {mode}: median (low-rho tail) FRACTIONAL stat error on unfolded bins ===")
    print(f"  L-curve tau = {lcurve_tau:.3g}")
    header = "  tau".ljust(16) + "".join(f"{s+' GeV':>22}" for s in slices)
    print(header)
    print("  " + "-" * (len(header)))
    for label, tau, tot, _inp in results:
        cells = "".join(
            f"{100*tot[s][0]:7.1f}% ({100*tot[s][1]:5.1f}%)".rjust(22) for s in slices
        )
        print(f"  {label:<14}{cells}")
    print("  (cell = median over reported bins (worst low-rho tail bin); total stat = input+matrix)")

    # Plot: median total-stat error vs tau, one line per slice; tau=0 dashed.
    colors = ["#5790fc", "#f89c20", "#e42536", "#964a8b"]
    fig, ax = plt.subplots(figsize=(11, 8))
    reg_pts = [(t, r) for (lab, t, r, _i) in results if t > 0]
    taus_x = [t for t, _ in reg_pts]
    base = {lab: tot for (lab, t, tot, _i) in results if t == 0}["0 (no reg)"]
    for k, s in enumerate(slices):
        y = [100 * r[s][0] for _t, r in reg_pts]
        ax.plot(taus_x, y, "-o", color=colors[k % len(colors)], label=f"{s} GeV")
        ax.axhline(100 * base[s][0], color=colors[k % len(colors)], ls="--", alpha=0.6, lw=1)
    ax.axvline(lcurve_tau, color="gray", ls=":", lw=1.5)
    ax.text(lcurve_tau, ax.get_ylim()[1], "  L-curve", color="gray", va="top", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel(r"regularization strength  $\tau$")
    ax.set_ylabel("median stat. error on unfolded bins (%)")
    ax.legend(title=(("Groomed" if groomed else "Ungroomed")
                     + r"  (dashed = $\tau$=0 baseline)"))
    hep.cms.label("Internal", data=True, lumi="138", com="13", fontsize=18, ax=ax)
    out = REPO_ROOT / "outputs" / "zjet" / "validation" / f"error_vs_tau_{mode}"
    fig.tight_layout()
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["groomed", "ungroomed"],
                    choices=["groomed", "ungroomed"])
    ap.add_argument("--taus", nargs="+", type=float, default=TAU_GRID)
    args = ap.parse_args()
    for mode in args.modes:
        run_mode(mode == "groomed", args.taus)


if __name__ == "__main__":
    main()
