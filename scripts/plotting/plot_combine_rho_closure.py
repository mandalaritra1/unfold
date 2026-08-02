#!/usr/bin/env python3
"""Plot Combine self-closure against the nominal PYTHIA truth."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unfold.utils.cms_plot import (  # noqa: E402
    PUB_ANNOTATION_FONTSIZE,
    PUB_LABEL_FONTSIZE,
    PUB_LEGEND_FONTSIZE,
    PUB_TICK_FONTSIZE,
    stamp_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="original_jacobian_reg")
    parser.add_argument("--mode", choices=("groomed", "ungroomed"), default="groomed")
    parser.add_argument(
        "--closure-dir",
        type=Path,
        default=REPO_ROOT / "outputs/zjet/rho/combine_full/original_jacobian_reg_groomed_self_closure",
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


def offsets(edges_by_pt: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(edges) - 1 for edges in edges_by_pt], dtype=int)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    return starts, counts


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use(hep.style.CMS)

    closure_dir = args.closure_dir.resolve()
    stem = f"zjet_rho_{args.tag}_{args.mode}_combine"
    result = np.load(closure_dir / f"{stem}_combine_unfolded.npz", allow_pickle=True)
    meta = np.load(closure_dir / f"{stem}_meta.npz", allow_pickle=True)
    gen_edges_by_pt = json.loads(str(result["gen_edges_json"].item()))
    pt_edges = np.asarray(result["pt_edges"], dtype=float)
    starts, counts = offsets(gen_edges_by_pt)

    expected = normalize_density_by_pt(np.asarray(meta["full_truth"], dtype=float), gen_edges_by_pt)
    closure = np.asarray(result["normalized"], dtype=float)
    closure_stat = np.sqrt(np.clip(np.diag(result["cov_norm_stat"]), 0.0, None))
    closure_total = np.sqrt(np.clip(np.diag(result["cov_norm_total"]), 0.0, None))
    poi = np.asarray(result["poi_values"], dtype=float)
    max_poi_deviation = float(np.max(np.abs(poi - 1.0)))

    output = args.output
    if output is None:
        output = closure_dir / f"{stem}_self_closure.png"
    else:
        output = output.resolve()

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
        idx = np.arange(starts[ipt], starts[ipt] + counts[ipt])
        edges = np.asarray(gen_edges_by_pt[ipt], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_widths = 0.5 * np.diff(edges)

        ax.step(
            edges,
            np.r_[expected[idx], expected[idx][-1]],
            where="post",
            color="#008060",
            lw=1.8,
            label="PYTHIA truth" if panel == 0 else None,
        )
        ax.fill_between(
            centers,
            closure[idx] - closure_total[idx],
            closure[idx] + closure_total[idx],
            step="mid",
            color="#5790fc",
            alpha=0.25,
            label="closure stat+syst" if panel == 0 else None,
        )
        ax.errorbar(
            centers,
            closure[idx],
            yerr=closure_stat[idx],
            xerr=half_widths,
            fmt="o",
            color="#111827",
            markersize=4,
            label="Combine closure" if panel == 0 else None,
        )

        denom = np.where(expected[idx] != 0.0, expected[idx], np.nan)
        ratio = closure[idx] / denom
        ratio_total = closure_total[idx] / denom
        rax.axhline(1.0, color="#6b7280", lw=1)
        rax.fill_between(
            centers,
            ratio - ratio_total,
            ratio + ratio_total,
            step="mid",
            color="#5790fc",
            alpha=0.25,
        )
        rax.errorbar(centers, ratio, yerr=closure_stat[idx] / denom, xerr=half_widths, fmt="o", color="#111827", markersize=4)
        # Keep the conventional +-20% window but never clip: widen it if the
        # closure or its band actually runs outside.
        span = np.nanmax(np.abs(np.r_[ratio - ratio_total, ratio + ratio_total] - 1.0))
        half = max(0.2, float(span) * 1.1) if np.isfinite(span) else 0.2
        rax.set_ylim(1.0 - half, 1.0 + half)
        rax.grid(True, alpha=0.25)
        rax.set_xlabel(r"$\log_{10}(\rho^2)$", fontsize=PUB_LABEL_FONTSIZE)
        rax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        if panel == 0:
            rax.set_ylabel("closure / truth", fontsize=PUB_LABEL_FONTSIZE)

        hi = int(pt_edges[ipt + 1]) if ipt + 1 < len(pt_edges) - 1 else None
        pt_range = f"{int(pt_edges[ipt])}" + (f"-{hi}" if hi is not None else "+") + " GeV"
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        if panel == 0:
            ax.set_ylabel(r"$\frac{1}{d\sigma/dp_T}\frac{d\sigma}{d\log_{10}(\rho^2)\,dp_T}$",
                          fontsize=PUB_LABEL_FONTSIZE)
        # Headroom so the upper-left CMS block clears the curves.
        top = np.nanmax(np.r_[expected[idx], closure[idx] + closure_total[idx]])
        if np.isfinite(top) and top > 0:
            ax.set_ylim(0.0, top * 1.5)
        # data=False already prepends "Simulation" -- the first arg is the flavor.
        hep.cms.label("Internal", data=False, loc=2, ax=ax, rlabel=pt_range)

    axes[0, 0].legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE, loc="upper right")
    axes[0, 0].text(
        0.03, 0.72, f"self-closure ({args.mode})\nmax $|r-1|$ = {max_poi_deviation:.3g}",
        transform=axes[0, 0].transAxes, va="top", ha="left",
        fontsize=PUB_ANNOTATION_FONTSIZE,
    )
    stamp_figure(fig, inputs=f"{stem} @ {closure_dir.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
