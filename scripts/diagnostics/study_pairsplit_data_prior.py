#!/usr/bin/env python3
"""Prior-dependence (data-prior) test for the pair-split channels.

The response matrix is built from PYTHIA8, so the migrations and efficiency
it encodes carry the PYTHIA generator-level $\\rho$ shape as a prior.  This
test checks that the unfolded result does not depend on that choice:

1. unfold the data with the nominal response;
2. reweight the generator-level spectrum, per analysis gen bin, so that it
   matches the nominal unfolded data (response columns and misses are scaled
   coherently, which is exactly a rebuild of the response from the
   bin-reweighted simulation);
3. unfold the same data a second time with this data-prior response;
4. compare the two normalized spectra.

Writes, under ``outputs/pairsplit_run2/validation_data_prior/<channel>_<mode>/``:
  - ``data_prior_pt<i>.pdf``  per-pT-slice spectrum + ratio panels
  - ``data_prior_metrics.txt`` per-slice mean/max departures vs the stat band

Usage:
  source scripts/setup_root.sh
  .venv/bin/python scripts/diagnostics/study_pairsplit_data_prior.py \
      [--channel dijet trijet] [--grooming-mode groomed ungroomed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")

import numpy as np

import run_pairsplit_unfolding as runner
from unfold.tools.pairsplit_run2_inputs import (
    load_pairsplit_run2_inputs,
    prepare_pairsplit_inputs,
)
from unfold.utils.merge_helpers import unflatten_gen_by_pt

OUT_ROOT = REPO_ROOT / "outputs" / "pairsplit_run2" / "validation_data_prior"


def normalize(u, flat):
    """Unit area per pT slice, bin-width divided."""
    flat = np.asarray(flat, dtype=float)
    out = np.zeros_like(flat)
    offset = 0
    for edges in u.gen_edges_by_pt:
        edges = np.asarray(edges, dtype=float)
        nbins = len(edges) - 1
        widths = np.diff(edges)
        block = flat[offset : offset + nbins]
        denom = block.sum()
        out[offset : offset + nbins] = block / widths / (denom if denom else 1.0)
        offset += nbins
    return out


def build_unfolder(channel: str, grooming_mode: str):
    """Nominal-only pair-split unfolder via the runner's prepared path."""

    import ROOT

    from unfold.tools.unfolder_core import Unfolder

    argv = ["--channel", channel, "--grooming-mode", grooming_mode, "--no-model-envelope"]
    if channel == "dijet" and grooming_mode == "groomed":
        argv += ["--normalization-window", "peak"]
    args = runner.parse_args(argv)
    args = runner.channel_resolved_args(args, channel)

    inputs = load_pairsplit_run2_inputs(channel, input_root=args.input_root)
    prepared = prepare_pairsplit_inputs(
        inputs,
        args.binning,
        ("nominal",),
        grooming_mode=grooming_mode,
    )
    out_dir = OUT_ROOT / f"{channel}_{grooming_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = runner.build_pairsplit_spec(
        channel, out_dir, args, grooming_mode=grooming_mode
    )
    if hasattr(spec, "model_envelope"):
        from dataclasses import replace

        spec = replace(spec, model_envelope=False)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    u = Unfolder.from_prepared_inputs(
        spec,
        grooming_mode == "groomed",
        mc_inputs=prepared.mc_inputs,
        data_inputs=prepared.data_inputs,
        analysis_binning=prepared.analysis_binning,
        systematics=("nominal",),
        measured_covariance=prepared.measured_covariance,
        first_reported_pt_bin=prepared.first_reported_pt_bin,
        cms_label=args.cms_label,
        lumi=args.lumi,
        com=args.com,
    )
    reported_minimum = float(
        getattr(
            prepared.analysis_binning,
            "reported_two_log10_rho_minimum",
            -3.5 if grooming_mode == "groomed" else -2.5,
        )
    )
    return u, out_dir, reported_minimum


def run_mode(channel: str, grooming_mode: str):
    print(f"\n=== Pair-split data-prior test: {channel} {grooming_mode} ===")
    u, out_dir, reported_minimum = build_unfolder(channel, grooming_mode)

    x_nom = np.asarray(u.y_unf, dtype=float).copy()
    stat_frac = [
        np.asarray(u.normalized_results[i]["stat_unc_frac"], dtype=float)
        for i in range(len(u.normalized_results))
    ]

    resp_nom = np.asarray(u.mosaic_dict["nominal"], dtype=float)
    misses_nom = np.asarray(u.misses_2d, dtype=float)
    truth_nom = resp_nom.sum(axis=0) + misses_nom

    # Per-gen-bin weights matching the prior to the nominal unfolded data.
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(truth_nom > 0, np.clip(x_nom, 0.0, None) / truth_nom, 1.0)
    # A weight of zero would empty a response column; keep a small floor so
    # the data-prior response stays invertible in bins where the unfolded
    # data fluctuates to non-positive values.
    weights = np.clip(weights, 1e-3, None)

    resp_dp = resp_nom * weights[None, :]
    misses_dp = misses_nom * weights
    truth_dp = truth_nom * weights

    # _perform_unfold reads the misses from the prepared per-systematic dict,
    # so the reweighted misses must be swapped in for the data-prior response
    # (scaling matched and missed yields coherently keeps the efficiency of
    # the reweighted simulation).
    misses_saved = u.misses_2d_dict["nominal"]
    u.misses_2d_dict["nominal"] = misses_dp
    try:
        u._perform_unfold(
            systematic="nominal",
            resp_np=resp_dp,
            true_flat_override=truth_dp,
        )
    finally:
        u.misses_2d_dict["nominal"] = misses_saved
    x_dp = np.asarray(u.y_unf, dtype=float).copy()

    norm_nom = normalize(u, x_nom)
    norm_dp = normalize(u, x_dp)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(norm_nom != 0, norm_dp / norm_nom, 1.0)

    mask_blocks = []
    for edges in u.gen_edges_by_pt:
        edges = np.asarray(edges, dtype=float)
        mask_blocks.append(edges[:-1] >= reported_minimum - 1e-9)

    metrics_path = out_dir / "data_prior_metrics.txt"
    ratio_by_pt = unflatten_gen_by_pt(ratio, u.gen_edges_by_pt)
    with open(metrics_path, "w") as f:
        f.write(
            f"# data-prior test ({channel}, {grooming_mode}); shown-window bins only\n"
            "# pt_slice  mean|ratio-1|  max|ratio-1|  max_stat_frac\n"
        )
        for i in u._reported_pt_indices():
            m = mask_blocks[i]
            dep = np.abs(ratio_by_pt[i][m] - 1.0)
            f.write(
                f"pt{i} {np.mean(dep):.3e} {np.max(dep):.3e} "
                f"{np.max(np.asarray(stat_frac[i])[m]):.6f}\n"
            )
            print(
                f"  pt{i}: mean |ratio-1| = {np.mean(dep):.3e}, "
                f"max = {np.max(dep):.3e}, stat band max = {np.max(np.asarray(stat_frac[i])[m]):.4f}"
            )
    print(f"  wrote {metrics_path}")

    _plot(u, out_dir, norm_nom, norm_dp, ratio, stat_frac, reported_minimum)
    print(f"  figures -> {out_dir}/data_prior_pt*.pdf")


def _plot(u, out_dir, norm_nom, norm_dp, ratio, stat_frac, reported_minimum):
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")

    nom_by_pt = unflatten_gen_by_pt(norm_nom, u.gen_edges_by_pt)
    dp_by_pt = unflatten_gen_by_pt(norm_dp, u.gen_edges_by_pt)
    ratio_by_pt = unflatten_gen_by_pt(ratio, u.gen_edges_by_pt)

    for i in u._reported_pt_indices():
        edges = np.asarray(u.gen_edges_by_pt[i], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xlo, xhi = u._observable_xlim(i)
        visible = edges[:-1] >= xlo

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, height_ratios=[3, 1],
            figsize=(9, 9), gridspec_kw={"hspace": 0.06},
        )
        hep.histplot(
            nom_by_pt[i], edges, ax=ax_top, color="black",
            label="Nominal prior", lw=2.0,
        )
        ax_top.plot(
            centers, dp_by_pt[i], "s", mfc="none", mec="#2ca02c", mew=2.0,
            ms=9, label="Data prior",
        )
        ax_top.set_ylabel(r"$\frac{1}{N}\frac{dN}{d\log_{10}\rho^2}$")
        ax_top.legend(fontsize=15)
        hep.cms.label(
            u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(),
            fontsize=19, ax=ax_top,
        )
        top_vals = np.concatenate([nom_by_pt[i][visible], dp_by_pt[i][visible]])
        if np.any(top_vals > 0):
            ax_top.set_ylim(0, 1.35 * float(np.max(top_vals)))

        band = np.asarray(stat_frac[i], dtype=float)
        band_step = np.append(band, band[-1])
        ax_bot.fill_between(
            edges, 1.0 - band_step, 1.0 + band_step, step="post",
            color="0.8", alpha=0.6, lw=0, label="Stat. unc.",
        )
        ax_bot.plot(centers, ratio_by_pt[i], "s", mfc="none", mec="#2ca02c", mew=2.0, ms=9)
        ax_bot.axhline(1.0, color="gray", lw=1)
        vis_dev = np.abs(ratio_by_pt[i][visible] - 1.0)
        vis_band = band[visible]
        span = max(0.02, 1.4 * float(np.max(np.concatenate([vis_dev, vis_band]))))
        ax_bot.set_ylim(1.0 - span, 1.0 + span)
        ax_bot.set_xlim(xlo, xhi)
        ax_bot.set_xlabel(u._observable_label())
        ax_bot.set_ylabel("Data prior / Nominal", fontsize=15)

        lo = u.pt_edges[i]
        hi = u.pt_edges[i + 1] if i + 2 < len(u.pt_edges) else float("inf")
        hi_s = r"$\infty$" if not np.isfinite(hi) else f"{hi:.0f}"
        ax_top.text(
            0.05, 0.95, f"$p_T$  {lo:.0f}–{hi_s} GeV",
            transform=ax_top.transAxes, va="top", fontsize=15,
        )
        for ext in ("pdf", "png"):
            fig.savefig(
                out_dir / f"data_prior_pt{i}.{ext}",
                bbox_inches="tight", pad_inches=0.1,
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", nargs="+", default=["dijet", "trijet"])
    parser.add_argument("--grooming-mode", nargs="+", default=["groomed", "ungroomed"])
    args = parser.parse_args()
    for channel in args.channel:
        for mode in args.grooming_mode:
            run_mode(channel, mode)


if __name__ == "__main__":
    main()
