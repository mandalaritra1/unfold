#!/usr/bin/env python3
"""Compare D'Agostini (RooUnfoldBayes) vs TUnfold for rho, any channel.

For each grooming mode it overlays, per-pT normalized, the two unfolded results
with their statistical uncertainties and the PYTHIA gen prediction, plus a
Bayes/TUnfold ratio panel.

Both backends are run as full ``Unfolder``s (TUnfold and RooUnfoldBayes), so each
carries its own statistical uncertainty: zjet keeps the jackknife; dijet/trijet
have no jackknife inputs, so it is the propagated input covariance.

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

from unfold.tools.rho_channel_inputs import build_prepared_rho_inputs, discover_rho_channel_files
from unfold.tools.unfolder_core import (
    RHO_FIXED_JEC_SPEC,
    Unfolder,
    get_spec,
    unflatten_gen_by_pt,
)

PT_MIN = 200.0  # drop the 0-200 GeV sink bin from the panels


def build_both(channel, year, tag, n_iter, groomed):
    """Build a TUnfold and a RooUnfoldBayes ``Unfolder`` for one mode."""
    if channel == "zjet":
        spec_tu = get_spec("zjet", "rho", tag)
        u_tu = Unfolder(spec_tu, groomed, do_syst=False)
        u_bay = Unfolder(
            replace(spec_tu, method="roounfold_bayes", n_iter=n_iter),
            groomed, do_syst=False,
        )
        return u_tu, u_bay, REPO_ROOT / spec_tu.output_dir / "roounfold_bayes"

    files = discover_rho_channel_files(REPO_ROOT / "inputs", channel, year)
    prepared = build_prepared_rho_inputs(files)
    mode = "groomed" if groomed else "ungroomed"
    out_rel = f"outputs/{channel}/{year}/rho/roounfold_bayes/"

    def build(method):
        spec = replace(
            RHO_FIXED_JEC_SPEC,
            output_dir=out_rel, method=method, n_iter=n_iter,
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

    return build("tunfold"), build("roounfold_bayes"), REPO_ROOT / out_rel


def _per_pt(unfolder, key):
    return [np.asarray(r[key], float) for r in unfolder.normalized_results]


def plot_comparison(u_tu, u_bay, mode, title, out_dir, n_iter):
    edges_by_pt = u_tu.gen_edges_by_pt
    pt_edges = u_tu.pt_edges
    tu_unf = _per_pt(u_tu, "unfolded")
    ba_unf = _per_pt(u_bay, "unfolded")
    py_gen = _per_pt(u_tu, "true")  # PYTHIA gen prediction (per-pT normalized)
    tu_frac = unflatten_gen_by_pt(u_tu.stat_unc_frac, edges_by_pt)
    ba_frac = unflatten_gen_by_pt(u_bay.stat_unc_frac, edges_by_pt)

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

        tu_y, ba_y, py_y = tu_unf[i], ba_unf[i], py_gen[i]
        # PYTHIA gen as a dotted step reference (matches the production plots).
        ax.stairs(py_y, edges, color="#1f77b4", ls=":", lw=2.0, label="PYTHIA8 (gen)")
        ax.errorbar(centers, tu_y, yerr=np.abs(tu_y) * tu_frac[i], fmt="o", ms=6, lw=1.3,
                    color="black", label="TUnfold")
        ax.errorbar(centers, ba_y, yerr=np.abs(ba_y) * ba_frac[i], fmt="s", ms=7, lw=1.3,
                    color="#d62728", mfc="none", mew=1.6,
                    label=f"RooUnfoldBayes n={n_iter}")

        lo, hi = pt_edges[i], pt_edges[i + 1]
        lbl = (rf"${lo:g}<p_T<{hi:g}$ GeV" if hi < 13000 else rf"$p_T>{lo:g}$ GeV")
        ax.text(0.04, 0.95, lbl, transform=ax.transAxes, ha="left", va="top", fontsize=13)
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(u_tu._normalized_ylabel())
            ax.legend(fontsize=9, loc="lower center")

        with np.errstate(divide="ignore", invalid="ignore"):
            r_tu = np.where(tu_y != 0, py_y / tu_y, np.nan)
            r_ba = np.where(ba_y != 0, py_y / ba_y, np.nan)
        rax.axhline(1.0, color="black", lw=1.0, ls=":")
        rax.plot(centers, r_tu, "o", ms=5, color="black", label="PYTHIA/TUnfold")
        rax.plot(centers, r_ba, "s", ms=5, color="#d62728", mfc="none", mew=1.4,
                 label="PYTHIA/Bayes")
        rax.set_ylim(0.8, 1.2)
        rax.set_ylabel("PYTHIA / unfolded", fontsize=10)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$", fontsize=12)
        rax.grid(alpha=0.25)
        if c == 0:
            rax.legend(fontsize=8, loc="lower left", ncol=2)

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
        u_tu, u_bay, out_dir = build_both(args.channel, args.year, args.tag, args.n_iter, groomed)
        out = plot_comparison(u_tu, u_bay, mode, title, out_dir, args.n_iter)
        print(f"[{args.channel} {mode}] wrote {out}")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(u_tu.y_unf != 0, u_bay.y_unf / u_tu.y_unf, np.nan)
        dev = np.abs(ratio - 1.0)
        dev = dev[np.isfinite(dev)]
        tu_med = float(np.median(u_tu.stat_unc_frac[np.isfinite(u_tu.stat_unc_frac)]))
        ba_med = float(np.median(u_bay.stat_unc_frac[np.isfinite(u_bay.stat_unc_frac)]))
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
