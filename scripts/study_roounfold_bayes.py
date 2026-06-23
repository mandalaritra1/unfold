#!/usr/bin/env python3
"""Proof-of-concept: D'Agostini (RooUnfoldBayes) vs TUnfold for Z+jet rho.

Builds the zjet rho ``Unfolder`` (which prepares the response, the fake-corrected
measured data, the misses, and the jackknife replicas, and runs the nominal
TUnfold unfold + its jackknife). It then re-unfolds the *same* inputs and the
*same* jackknife replicas through RooUnfoldBayes and overlays the two: central
unfolded result with jackknife statistical uncertainties, plus a Bayes/TUnfold
ratio panel.

The jackknife is identical to the TUnfold path -- the stat uncertainty is the
sqrt(10/9)-scaled spread of the 10 input-varied and 10 matrix-varied re-unfolds,
combined in quadrature -- which is the whole point: the CMS-recommended iterative
Bayes pathway while still using jackknife for the statistical uncertainty.

    source scripts/setup_root.sh
    source scripts/setup_roounfold.sh
    python scripts/study_roounfold_bayes.py --tag original --n-iter 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unfold.tools.roounfold_backend import bayes_unfold
from unfold.tools.unfolder_core import Unfolder, get_spec, unflatten_gen_by_pt

JK_SCALE = np.sqrt(10.0 / 9.0)  # leave-one-out jackknife std scaling (matches core)


def fake_correct(unfolder, measured_flat):
    """data * (1 - fake_fraction), clipped >= 0 -- mirrors Unfolder._apply_fake_correction."""
    corrected = np.asarray(measured_flat, float) * (1.0 - unfolder.fake_fraction_2d)
    return np.clip(corrected, 0.0, None)


def bayes_with_jackknife(unfolder, n_iter):
    """Run RooUnfoldBayes on the Unfolder's prepared inputs + jackknife replicas.

    Returns dict with the central unfolded vector and the fractional input /
    matrix / total jackknife statistical uncertainties (same definitions as
    Unfolder._compute_stat_unc).
    """
    resp = np.asarray(unfolder.mosaic, float)
    misses = np.asarray(unfolder.misses_2d, float)
    truth = resp.sum(axis=0) + misses
    meas = fake_correct(unfolder, unfolder.mosaic_2d)

    central, _ = bayes_unfold(resp, meas, truth, n_iter=n_iter, tag="nom")

    # Input jackknife: vary the measured spectrum, fixed response.
    input_unf = []
    for jk_meas in unfolder.mosaic_2d_jk_list:
        y, _ = bayes_unfold(resp, fake_correct(unfolder, jk_meas), truth,
                            n_iter=n_iter, tag="jkin")
        input_unf.append(y)

    # Matrix jackknife: vary the response, fixed measured spectrum.
    matrix_unf = []
    for jk_resp in unfolder.mosaic_jk_list:
        jk_resp = np.asarray(jk_resp, float)
        jk_truth = jk_resp.sum(axis=0) + misses
        y, _ = bayes_unfold(jk_resp, meas, jk_truth, n_iter=n_iter, tag="jkmat")
        matrix_unf.append(y)

    input_std = JK_SCALE * np.std(input_unf, axis=0)
    matrix_std = JK_SCALE * np.std(matrix_unf, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(central != 0, central, np.nan)
        input_frac = np.abs(input_std / denom)
        matrix_frac = np.abs(matrix_std / denom)
    total_frac = np.sqrt(np.nan_to_num(input_frac) ** 2 + np.nan_to_num(matrix_frac) ** 2)
    return {
        "central": central,
        "input_frac": np.nan_to_num(input_frac),
        "matrix_frac": np.nan_to_num(matrix_frac),
        "total_frac": total_frac,
    }


def plot_comparison(unfolder, bayes, mode, out_dir, n_iter):
    pt_edges = unfolder.pt_edges
    edges_by_pt = unfolder.gen_edges_by_pt

    tu_unf_by_pt = unflatten_gen_by_pt(unfolder.y_unf, edges_by_pt)
    tu_frac_by_pt = unflatten_gen_by_pt(unfolder.stat_unc_frac, edges_by_pt)
    bay_unf_by_pt = unflatten_gen_by_pt(bayes["central"], edges_by_pt)
    bay_frac_by_pt = unflatten_gen_by_pt(bayes["total_frac"], edges_by_pt)

    # Skip the 0-200 GeV sink bin (pt index 0), like the analysis plots.
    panels = [i for i in range(len(edges_by_pt)) if pt_edges[i + 1] > 200.0]
    ncol = len(panels)
    fig = plt.figure(figsize=(5.4 * ncol, 6.2))
    outer = fig.add_gridspec(1, ncol, wspace=0.3, left=0.07, right=0.985,
                             top=0.9, bottom=0.12)

    for c, i in enumerate(panels):
        edges = np.asarray(edges_by_pt[i], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        sub = outer[0, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(sub[0])
        rax = fig.add_subplot(sub[1], sharex=ax)

        tu, tuf = tu_unf_by_pt[i], tu_frac_by_pt[i]
        ba, baf = bay_unf_by_pt[i], bay_frac_by_pt[i]
        ax.errorbar(centers, tu, yerr=np.abs(tu) * tuf, fmt="o", ms=6, lw=1.3,
                    color="black", label="TUnfold (+ jackknife)")
        ax.errorbar(centers, ba, yerr=np.abs(ba) * baf, fmt="s", ms=7, lw=1.3,
                    color="#d62728", mfc="none", mew=1.6,
                    label=f"RooUnfoldBayes n={n_iter} (+ jackknife)")

        lo, hi = pt_edges[i], pt_edges[i + 1]
        lbl = (rf"${lo:g}<p_T<{hi:g}$ GeV" if hi < 13000 else rf"$p_T>{lo:g}$ GeV")
        ax.text(0.04, 0.95, lbl, transform=ax.transAxes, ha="left", va="top", fontsize=13)
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel("Unfolded yield")
            ax.legend(fontsize=9, loc="lower center")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(tu != 0, ba / tu, np.nan)
        rax.axhline(1.0, color="black", lw=1.0, ls=":")
        rax.plot(centers, ratio, "s", ms=5, color="#d62728")
        rax.set_ylim(0.8, 1.2)
        rax.set_ylabel("Bayes/TU", fontsize=10)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$", fontsize=12)
        rax.grid(alpha=0.25)

    fig.suptitle(f"Z+jet rho ({mode}): D'Agostini (RooUnfoldBayes) vs TUnfold", fontsize=14)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bayes_vs_tunfold_{mode}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="original")
    ap.add_argument("--n-iter", type=int, default=4)
    args = ap.parse_args()

    spec = get_spec("zjet", "rho", args.tag)
    out_dir = REPO_ROOT / spec.output_dir / "roounfold_bayes"

    summary_lines = [f"Z+jet rho RooUnfoldBayes (n_iter={args.n_iter}) vs TUnfold\n"]
    for mode, groomed in (("ungroomed", False), ("groomed", True)):
        print(f"[{mode}] building Unfolder (TUnfold + jackknife) ...")
        u = Unfolder(spec, groomed, do_syst=False)
        print(f"[{mode}] running RooUnfoldBayes on the same inputs + replicas ...")
        bayes = bayes_with_jackknife(u, args.n_iter)
        out = plot_comparison(u, bayes, mode, out_dir, args.n_iter)
        print(f"[{mode}] wrote {out}")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(u.y_unf != 0, bayes["central"] / u.y_unf, np.nan)
        dev = np.abs(ratio - 1.0)
        dev = dev[np.isfinite(dev)]
        tu_med = float(np.median(u.stat_unc_frac[np.isfinite(u.stat_unc_frac)]))
        ba_med = float(np.median(bayes["total_frac"][np.isfinite(bayes["total_frac"])]))
        summary_lines += [
            f"{mode}:",
            f"  central Bayes/TUnfold: mean |r-1| = {dev.mean():.4f}, max = {dev.max():.4f}",
            f"  median jackknife stat frac: TUnfold = {tu_med:.4f}, Bayes = {ba_med:.4f}",
            "",
        ]

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary)
    print(f"wrote {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
