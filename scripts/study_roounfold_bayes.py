#!/usr/bin/env python3
"""Compare D'Agostini (RooUnfoldBayes) vs TUnfold for rho, any channel.

For each grooming mode it overlays the two unfolded results (per-pT normalized)
with their statistical uncertainties and a Bayes/TUnfold ratio panel.

- zjet: builds the spec ``Unfolder`` (TUnfold + jackknife), then re-unfolds the
  *same* inputs and the *same* jackknife replicas through RooUnfoldBayes, so the
  Bayes statistical uncertainty is jackknife-based too.
- dijet/trijet: builds two prepared-input ``Unfolder``s (one per backend); these
  channels have no jackknife inputs, so both stat uncertainties come from the
  propagated input covariance.

    source scripts/setup_root.sh
    source scripts/setup_roounfold.sh
    python scripts/study_roounfold_bayes.py --channel zjet  --tag original --n-iter 4
    python scripts/study_roounfold_bayes.py --channel dijet --year 2018 --n-iter 4
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
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
from unfold.tools.rho_channel_inputs import build_prepared_rho_inputs, discover_rho_channel_files
from unfold.tools.unfolder_core import (
    RHO_FIXED_JEC_SPEC,
    Unfolder,
    get_spec,
    unflatten_gen_by_pt,
)

JK_SCALE = np.sqrt(10.0 / 9.0)  # leave-one-out jackknife std scaling (matches core)
PT_MIN = 200.0  # drop the 0-200 GeV sink bin from the panels


def fake_correct(unfolder, measured_flat):
    corrected = np.asarray(measured_flat, float) * (1.0 - unfolder.fake_fraction_2d)
    return np.clip(corrected, 0.0, None)


def bayes_with_jackknife(unfolder, n_iter):
    """RooUnfoldBayes on the Unfolder's prepared inputs + jackknife replicas."""
    resp = np.asarray(unfolder.mosaic, float)
    misses = np.asarray(unfolder.misses_2d, float)
    truth = resp.sum(axis=0) + misses
    meas = fake_correct(unfolder, unfolder.mosaic_2d)

    central, _ = bayes_unfold(resp, meas, truth, n_iter=n_iter, tag="nom")
    input_unf = [
        bayes_unfold(resp, fake_correct(unfolder, jk), truth, n_iter=n_iter, tag="jkin")[0]
        for jk in unfolder.mosaic_2d_jk_list
    ]
    matrix_unf = []
    for jk_resp in unfolder.mosaic_jk_list:
        jk_resp = np.asarray(jk_resp, float)
        jk_truth = jk_resp.sum(axis=0) + misses
        matrix_unf.append(bayes_unfold(jk_resp, meas, jk_truth, n_iter=n_iter, tag="jkmat")[0])

    input_std = JK_SCALE * np.std(input_unf, axis=0)
    matrix_std = JK_SCALE * np.std(matrix_unf, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(central != 0, central, np.nan)
        total = np.sqrt(np.nan_to_num(input_std / denom) ** 2 + np.nan_to_num(matrix_std / denom) ** 2)
    return central, np.nan_to_num(total)


def zjet_results(tag, n_iter, groomed):
    spec = get_spec("zjet", "rho", tag)
    u = Unfolder(spec, groomed, do_syst=False)
    bay_central, bay_frac = bayes_with_jackknife(u, n_iter)
    out_dir = REPO_ROOT / spec.output_dir / "roounfold_bayes"
    return (u.gen_edges_by_pt, u.pt_edges,
            (u.y_unf, u.stat_unc_frac), (bay_central, bay_frac), out_dir)


def channel_results(channel, year, n_iter, groomed):
    files = discover_rho_channel_files(REPO_ROOT / "inputs", channel, year)
    prepared = build_prepared_rho_inputs(files)
    mode = "groomed" if groomed else "ungroomed"
    out_rel = f"outputs/{channel}/{year}/rho/roounfold_bayes/"

    def build(method):
        spec = replace(
            RHO_FIXED_JEC_SPEC,
            output_dir=out_rel,
            method=method,
            n_iter=n_iter,
            xlim_lower_groomed=(
                prepared.binning[mode].gen_rho_edges_by_pt[0][0]
                if channel == "dijet" and groomed
                else RHO_FIXED_JEC_SPEC.xlim_lower_groomed
            ),
        )
        return Unfolder.from_prepared_inputs(
            spec, groomed,
            mc_inputs=prepared.mc, data_inputs=prepared.data,
            analysis_binning=prepared.binning[mode],
            systematics=prepared.systematics, lumi=59.7, com=13.0,
        )

    u_tu = build("tunfold")
    u_bay = build("roounfold_bayes")
    return (u_tu.gen_edges_by_pt, u_tu.pt_edges,
            (u_tu.y_unf, u_tu.stat_unc_frac), (u_bay.y_unf, u_bay.stat_unc_frac),
            REPO_ROOT / out_rel)


def plot_comparison(edges_by_pt, pt_edges, tu, bayes, mode, title, out_dir, n_iter):
    tu_unf = unflatten_gen_by_pt(tu[0], edges_by_pt)
    tu_frac = unflatten_gen_by_pt(tu[1], edges_by_pt)
    ba_unf = unflatten_gen_by_pt(bayes[0], edges_by_pt)
    ba_frac = unflatten_gen_by_pt(bayes[1], edges_by_pt)

    panels = [i for i in range(len(edges_by_pt)) if pt_edges[i + 1] > PT_MIN]
    ncol = len(panels)
    fig = plt.figure(figsize=(5.4 * ncol, 6.2))
    outer = fig.add_gridspec(1, ncol, wspace=0.3, left=0.07, right=0.985, top=0.9, bottom=0.12)

    for c, i in enumerate(panels):
        edges = np.asarray(edges_by_pt[i], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        sub = outer[0, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax = fig.add_subplot(sub[0])
        rax = fig.add_subplot(sub[1], sharex=ax)

        tu_y, tu_e = tu_unf[i], tu_frac[i]
        ba_y, ba_e = ba_unf[i], ba_frac[i]
        ax.errorbar(centers, tu_y, yerr=np.abs(tu_y) * tu_e, fmt="o", ms=6, lw=1.3,
                    color="black", label="TUnfold")
        ax.errorbar(centers, ba_y, yerr=np.abs(ba_y) * ba_e, fmt="s", ms=7, lw=1.3,
                    color="#d62728", mfc="none", mew=1.6,
                    label=f"RooUnfoldBayes n={n_iter}")

        lo, hi = pt_edges[i], pt_edges[i + 1]
        lbl = (rf"${lo:g}<p_T<{hi:g}$ GeV" if hi < 13000 else rf"$p_T>{lo:g}$ GeV")
        ax.text(0.04, 0.95, lbl, transform=ax.transAxes, ha="left", va="top", fontsize=13)
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel("Unfolded yield (per-pT normalized)")
            ax.legend(fontsize=9, loc="lower center")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(tu_y != 0, ba_y / tu_y, np.nan)
        rax.axhline(1.0, color="black", lw=1.0, ls=":")
        rax.plot(centers, ratio, "s", ms=5, color="#d62728")
        rax.set_ylim(0.8, 1.2)
        rax.set_ylabel("Bayes/TU", fontsize=10)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$", fontsize=12)
        rax.grid(alpha=0.25)

    fig.suptitle(f"{title} ({mode}): D'Agostini (RooUnfoldBayes) vs TUnfold", fontsize=14)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bayes_vs_tunfold_{mode}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="zjet", choices=("zjet", "dijet", "trijet"))
    ap.add_argument("--year", default="2018", help="dijet/trijet only")
    ap.add_argument("--tag", default="original", help="zjet only")
    ap.add_argument("--n-iter", type=int, default=4)
    args = ap.parse_args()

    title = {"zjet": "Z+jet rho", "dijet": "Dijet rho", "trijet": "Trijet rho"}[args.channel]
    summary = [f"{title} RooUnfoldBayes (n_iter={args.n_iter}) vs TUnfold\n"]
    out_dir = None
    for mode, groomed in (("ungroomed", False), ("groomed", True)):
        print(f"[{args.channel} {mode}] unfolding both backends ...")
        if args.channel == "zjet":
            edges, pt_edges, tu, bayes, out_dir = zjet_results(args.tag, args.n_iter, groomed)
        else:
            edges, pt_edges, tu, bayes, out_dir = channel_results(
                args.channel, args.year, args.n_iter, groomed)
        out = plot_comparison(edges, pt_edges, tu, bayes, mode, title, out_dir, args.n_iter)
        print(f"[{args.channel} {mode}] wrote {out}")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(tu[0] != 0, bayes[0] / tu[0], np.nan)
        dev = np.abs(ratio - 1.0)
        dev = dev[np.isfinite(dev)]
        tu_med = float(np.median(tu[1][np.isfinite(tu[1])]))
        ba_med = float(np.median(bayes[1][np.isfinite(bayes[1])]))
        summary += [
            f"{mode}:",
            f"  central Bayes/TUnfold: mean |r-1| = {dev.mean():.4f}, max = {dev.max():.4f}",
            f"  median stat frac: TUnfold = {tu_med:.4f}, Bayes = {ba_med:.4f}",
            "",
        ]

    text = "\n".join(summary)
    print("\n" + text)
    (out_dir / "summary.txt").write_text(text)
    print(f"wrote {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
