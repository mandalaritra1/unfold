#!/usr/bin/env python3
"""Compare Combine rho unfolding with TUnfold and generator truth overlays."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

from unfold.utils.cms_plot import (  # noqa: E402
    PUB_ANNOTATION_FONTSIZE,
    PUB_LABEL_FONTSIZE,
    PUB_LEGEND_FONTSIZE,
    PUB_TICK_FONTSIZE,
    stamp_figure,
)

LUMI_RLABEL = r"138 fb$^{-1}$ (13 TeV)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="original_jacobian_reg")
    parser.add_argument("--mode", choices=("groomed", "ungroomed"), default="groomed")
    parser.add_argument(
        "--combine-dir",
        type=Path,
        default=REPO_ROOT / "outputs/zjet/rho/combine_full/original_jacobian_reg_groomed_all",
    )
    parser.add_argument(
        "--tunfold-npz",
        type=Path,
        default=REPO_ROOT / "outputs/zjet/rho/original_jacobian_reg/data/normalized_covariance_groomed.npz",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def normalize_density_by_pt(values: np.ndarray, edges_by_pt: list[list[float]]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    normalized = np.zeros_like(values)
    offset = 0
    for edges in edges_by_pt:
        widths = np.diff(np.asarray(edges, dtype=float))
        count = len(widths)
        sl = slice(offset, offset + count)
        total = float(np.sum(values[sl]))
        if total > 0.0:
            normalized[sl] = values[sl] / total / widths
        offset += count
    return normalized


def build_nominal_unfolder(tag: str, groomed: bool):
    import matplotlib

    matplotlib.use("Agg")
    from unfold.tools.unfolder_core import Unfolder, get_spec

    spec = get_spec("zjet", "rho", tag)
    spec = replace(
        spec,
        output_dir=str(REPO_ROOT / "outputs/zjet/rho/combine_full/comparison_inputs") + "/",
    )
    return Unfolder(spec, groomed=groomed, do_syst=False, compute_jackknife_stat=False)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use(hep.style.CMS)

    mode = args.mode
    groomed = mode == "groomed"
    combine_dir = args.combine_dir.resolve()
    stem = f"zjet_rho_{args.tag}_{mode}_combine"
    combine_npz = combine_dir / f"{stem}_combine_unfolded.npz"
    combine_meta = combine_dir / f"{stem}_meta.npz"
    if args.output is None:
        output = combine_dir / f"{stem}_comparison_gen_tunfold.png"
    else:
        output = args.output.resolve()

    combine = np.load(combine_npz, allow_pickle=True)
    meta = np.load(combine_meta, allow_pickle=True)
    tunfold = np.load(args.tunfold_npz, allow_pickle=True)

    gen_edges_by_pt = json.loads(str(combine["gen_edges_json"].item()))
    pt_edges = np.asarray(combine["pt_edges"], dtype=float)
    gen_counts = np.asarray([len(edges) - 1 for edges in gen_edges_by_pt], dtype=int)
    starts = np.concatenate([[0], np.cumsum(gen_counts)[:-1]]).astype(int)

    combine_norm = np.asarray(combine["normalized"], dtype=float)
    combine_stat = np.sqrt(np.clip(np.diag(combine["cov_norm_stat"]), 0.0, None))
    combine_total = np.sqrt(np.clip(np.diag(combine["cov_norm_total"]), 0.0, None))

    tunfold_norm = np.asarray(tunfold["normalized"], dtype=float)
    tunfold_stat = np.sqrt(np.clip(np.diag(tunfold["cov_stat"]), 0.0, None))
    tunfold_total = np.sqrt(np.clip(np.diag(tunfold["cov_total"]), 0.0, None))

    pythia_norm = normalize_density_by_pt(np.asarray(meta["full_truth"], dtype=float), gen_edges_by_pt)

    unfolder = build_nominal_unfolder(args.tag, groomed)
    herwig_norm = normalize_density_by_pt(np.asarray(unfolder.herwig_gen_val_flat, dtype=float), gen_edges_by_pt)

    n_panels = len(gen_edges_by_pt) - 1
    # 2 x n_panels grid -> keep the ~10x10 CMS footprint per column, with the
    # ratio row taking a third of the column height.
    fig, axes = plt.subplots(
        2, n_panels, figsize=(10.5 * n_panels, 14.0), sharex="col",
        layout="constrained", gridspec_kw={"height_ratios": [3, 1]},
    )
    for panel, ipt in enumerate(range(1, len(gen_edges_by_pt))):
        ax = axes[0, panel]
        rax = axes[1, panel]
        start = starts[ipt]
        count = gen_counts[ipt]
        idx = np.arange(start, start + count)
        edges = np.asarray(gen_edges_by_pt[ipt], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_widths = 0.5 * np.diff(edges)

        ax.fill_between(
            centers,
            combine_norm[idx] - combine_total[idx],
            combine_norm[idx] + combine_total[idx],
            step="mid",
            color="#5790fc",
            alpha=0.24,
            label="Combine stat+syst" if panel == 0 else None,
        )
        ax.errorbar(
            centers,
            combine_norm[idx],
            yerr=combine_stat[idx],
            xerr=half_widths,
            fmt="o",
            color="#111827",
            markersize=4,
            label="Combine" if panel == 0 else None,
        )
        ax.fill_between(
            centers,
            tunfold_norm[idx] - tunfold_total[idx],
            tunfold_norm[idx] + tunfold_total[idx],
            step="mid",
            color="#f89c20",
            alpha=0.20,
            label="TUnfold stat+syst" if panel == 0 else None,
        )
        ax.errorbar(
            centers,
            tunfold_norm[idx],
            yerr=tunfold_stat[idx],
            xerr=half_widths,
            fmt="s",
            color="#9a3412",
            markersize=3.5,
            label="TUnfold" if panel == 0 else None,
        )
        ax.step(edges, np.r_[pythia_norm[idx], pythia_norm[idx][-1]], where="post", color="#008060", lw=1.8, label="PYTHIA gen" if panel == 0 else None)
        ax.step(edges, np.r_[herwig_norm[idx], herwig_norm[idx][-1]], where="post", color="#7c3aed", lw=1.8, ls="--", label="HERWIG gen" if panel == 0 else None)

        denom = np.where(tunfold_norm[idx] != 0.0, tunfold_norm[idx], np.nan)
        rax.axhline(1.0, color="#6b7280", lw=1)
        rax.errorbar(
            centers,
            combine_norm[idx] / denom,
            yerr=combine_total[idx] / denom,
            xerr=half_widths,
            fmt="o",
            color="#111827",
            markersize=4,
        )
        rax.step(edges, np.r_[pythia_norm[idx] / denom, (pythia_norm[idx] / denom)[-1]], where="post", color="#008060", lw=1.5)
        rax.step(edges, np.r_[herwig_norm[idx] / denom, (herwig_norm[idx] / denom)[-1]], where="post", color="#7c3aed", lw=1.5, ls="--")
        # Keep the conventional window but never clip: widen if any curve runs out.
        shown = np.r_[combine_norm[idx] / denom, pythia_norm[idx] / denom,
                      herwig_norm[idx] / denom,
                      (combine_norm[idx] + combine_total[idx]) / denom,
                      (combine_norm[idx] - combine_total[idx]) / denom]
        shown = np.abs(shown - 1.0)
        shown = shown[np.isfinite(shown)]
        half = max(0.55, float(shown.max()) * 1.1) if shown.size else 0.55
        rax.set_ylim(1.0 - half, 1.0 + half)
        rax.grid(True, alpha=0.25)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$", fontsize=PUB_LABEL_FONTSIZE)
        rax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        if panel == 0:
            rax.set_ylabel("ratio to TUnfold", fontsize=PUB_LABEL_FONTSIZE)

        hi = int(pt_edges[ipt + 1]) if ipt + 1 < len(pt_edges) - 1 else None
        pt_range = f"{int(pt_edges[ipt])}" + (f"-{hi}" if hi is not None else "+") + " GeV"
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        if panel == 0:
            ax.set_ylabel(unfolder.spec.normalized_ylabel, fontsize=PUB_LABEL_FONTSIZE)
        # Headroom so the upper-left CMS block clears the curves.
        top = np.nanmax(np.r_[combine_norm[idx] + combine_total[idx],
                              tunfold_norm[idx] + tunfold_total[idx],
                              pythia_norm[idx], herwig_norm[idx]])
        if np.isfinite(top) and top > 0:
            ax.set_ylim(0.0, top * 1.6)
        # rlabel stays lumi+com (skill rule 4); the pT range rides just under
        # the CMS block so each panel still self-identifies.
        hep.cms.label("Preliminary", data=True, loc=2, ax=ax, rlabel=LUMI_RLABEL)
        ax.text(0.04, 0.80, pt_range, transform=ax.transAxes, va="top", ha="left",
                fontsize=PUB_ANNOTATION_FONTSIZE)

    axes[0, 0].legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE, ncol=2,
                      loc="upper right")
    stamp_figure(fig, inputs=f"{stem} @ {combine_dir.name} vs {args.tunfold_npz.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
