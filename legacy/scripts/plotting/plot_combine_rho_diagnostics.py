#!/usr/bin/env python3
"""Bottom-line and uncertainty diagnostics for the Combine rho unfolding."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

from unfold.utils.cms_plot import (  # noqa: E402
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
    return parser.parse_args()


def chi2_from_covariance(delta: np.ndarray, covariance: np.ndarray, *, rcond: float = 1e-12) -> dict:
    delta = np.asarray(delta, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    symmetric = 0.5 * (covariance + covariance.T)
    vals, vecs = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(vals))), 1.0)
    keep = vals > rcond * scale
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return {"chi2": float("nan"), "ndof": 0, "pvalue": float("nan")}
    projected = vecs[:, keep].T @ delta
    chi2 = float(np.sum(projected * projected / vals[keep]))
    try:
        from scipy.stats import chi2 as scipy_chi2

        pvalue = float(scipy_chi2.sf(chi2, rank))
    except Exception:
        pvalue = float("nan")
    return {"chi2": chi2, "ndof": rank, "pvalue": pvalue}


def offsets(edges_by_pt: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(edges) - 1 for edges in edges_by_pt], dtype=int)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    return starts, counts


def build_unfolder(tag: str, groomed: bool):
    import matplotlib

    matplotlib.use("Agg")
    from unfold.tools.unfolder_core import Unfolder, get_spec

    spec = get_spec("zjet", "rho", tag)
    spec = replace(
        spec,
        output_dir=str(REPO_ROOT / "outputs/zjet/rho/combine_full/diagnostic_inputs") + "/",
    )
    return Unfolder(spec, groomed=groomed, do_syst=False, compute_jackknife_stat=False)


def bottom_line_rows(unfolder, combine: np.lib.npyio.NpzFile) -> tuple[list[dict], dict]:
    measured = np.asarray(unfolder.mosaic_2d, dtype=float)
    survival = 1.0 - np.asarray(unfolder.fake_fraction_2d, dtype=float)
    corrected_measured = measured * survival
    measured_variance = np.asarray(unfolder.measured_variances, dtype=float) * survival * survival
    reco_nominal = np.asarray(unfolder.mosaic.sum(axis=1), dtype=float)

    truth_nominal = np.asarray(unfolder.mosaic.sum(axis=0), dtype=float) + np.asarray(unfolder.misses_2d, dtype=float)
    combine_truth = np.asarray(combine["central"], dtype=float)
    cov_abs_stat = np.asarray(combine["cov_abs_stat"], dtype=float)
    cov_abs_total = np.asarray(combine["cov_abs_total"], dtype=float)

    reco_starts, reco_counts = offsets(unfolder.reco_edges_by_pt)
    gen_starts, gen_counts = offsets(unfolder.gen_edges_by_pt)

    rows = []
    global_reco = []
    global_gen = []
    for ipt in range(1, len(unfolder.pt_edges) - 1):
        ridx = np.arange(reco_starts[ipt], reco_starts[ipt] + reco_counts[ipt])
        gidx = np.arange(gen_starts[ipt], gen_starts[ipt] + gen_counts[ipt])
        reco_cov = np.diag(np.clip(measured_variance[ridx], 0.0, None))
        smeared = chi2_from_covariance(corrected_measured[ridx] - reco_nominal[ridx], reco_cov)
        combine_stat = chi2_from_covariance(
            combine_truth[gidx] - truth_nominal[gidx],
            cov_abs_stat[np.ix_(gidx, gidx)],
        )
        combine_total = chi2_from_covariance(
            combine_truth[gidx] - truth_nominal[gidx],
            cov_abs_total[np.ix_(gidx, gidx)],
        )
        hi = int(unfolder.pt_edges[ipt + 1]) if ipt + 1 < len(unfolder.pt_edges) - 1 else None
        rows.append(
            {
                "pt": f"{int(unfolder.pt_edges[ipt])}" + (f"-{hi}" if hi is not None else "+"),
                "smeared": smeared,
                "combine_unfolded_stat": combine_stat,
                "combine_unfolded_total": combine_total,
            }
        )
        global_reco.extend(ridx.tolist())
        global_gen.extend(gidx.tolist())

    global_reco = np.asarray(global_reco, dtype=int)
    global_gen = np.asarray(global_gen, dtype=int)
    global_row = {
        "pt": "global",
        "smeared": chi2_from_covariance(
            corrected_measured[global_reco] - reco_nominal[global_reco],
            np.diag(np.clip(measured_variance[global_reco], 0.0, None)),
        ),
        "combine_unfolded_stat": chi2_from_covariance(
            combine_truth[global_gen] - truth_nominal[global_gen],
            cov_abs_stat[np.ix_(global_gen, global_gen)],
        ),
        "combine_unfolded_total": chi2_from_covariance(
            combine_truth[global_gen] - truth_nominal[global_gen],
            cov_abs_total[np.ix_(global_gen, global_gen)],
        ),
    }
    return rows, global_row


def plot_bottom_line(rows: list[dict], global_row: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use(hep.style.CMS)

    plot_rows = rows + [global_row]
    labels = [row["pt"] for row in plot_rows]
    x = np.arange(len(labels))
    width = 0.25

    def reduced(row: dict, key: str) -> float:
        metric = row[key]
        return metric["chi2"] / metric["ndof"] if metric["ndof"] else np.nan

    # Single squarish panel -> keep the CMS style default figsize (skill rule 2).
    fig, ax = plt.subplots(layout="constrained")
    heights = np.array([
        [reduced(row, "smeared") for row in plot_rows],
        [reduced(row, "combine_unfolded_stat") for row in plot_rows],
        [reduced(row, "combine_unfolded_total") for row in plot_rows],
    ], dtype=float)
    ax.bar(x - width, heights[0], width, color="#5790fc", label="Reco smeared")
    ax.bar(x, heights[1], width, color="#f89c20", label="Combine unfolded stat")
    ax.bar(x + width, heights[2], width, color="#e42536", label="Combine unfolded total")
    ax.axhline(1.0, color="#6b7280", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$\chi^2 / \mathrm{ndof}$", fontsize=PUB_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.25)
    # Headroom from the actual bar maximum -- never a hard cap.
    top = np.nanmax(heights)
    if np.isfinite(top) and top > 0:
        ax.set_ylim(0.0, top * 1.5)
    ax.legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE, loc="upper right")
    hep.cms.label("Preliminary", data=True, loc=2, ax=ax, rlabel=LUMI_RLABEL)
    stamp_figure(fig, inputs=out.parent.name)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_uncertainty_breakdown(combine, tunfold, gen_edges_by_pt: list[list[float]], pt_edges: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use(hep.style.CMS)

    starts, counts = offsets(gen_edges_by_pt)
    n_panels = len(gen_edges_by_pt) - 1
    # 1 x n_panels -> uniform ~10x10-per-axes CMS footprint (skill rule 2).
    fig, axes = plt.subplots(1, n_panels, figsize=(10.5 * n_panels, 10.6),
                             sharey=True, layout="constrained")
    if n_panels == 1:
        axes = [axes]

    combine_y = np.asarray(combine["normalized"], dtype=float)
    combine_stat = np.sqrt(np.clip(np.diag(combine["cov_norm_stat"]), 0.0, None))
    combine_syst = np.sqrt(np.clip(np.diag(combine["cov_norm_syst"]), 0.0, None))
    combine_total = np.sqrt(np.clip(np.diag(combine["cov_norm_total"]), 0.0, None))

    tunfold_y = np.asarray(tunfold["normalized"], dtype=float)
    tunfold_stat = np.sqrt(np.clip(np.diag(tunfold["cov_stat"]), 0.0, None))
    tunfold_syst = np.sqrt(np.clip(np.diag(tunfold["cov_syst"]), 0.0, None))
    tunfold_total = np.sqrt(np.clip(np.diag(tunfold["cov_total"]), 0.0, None))

    shown: list[np.ndarray] = []
    for ax, ipt in zip(axes, range(1, len(gen_edges_by_pt))):
        idx = np.arange(starts[ipt], starts[ipt] + counts[ipt])
        edges = np.asarray(gen_edges_by_pt[ipt], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Plain fractions on the axis, not percent (skill rule 7).
        with np.errstate(divide="ignore", invalid="ignore"):
            c_stat = combine_stat[idx] / np.abs(combine_y[idx])
            c_syst = combine_syst[idx] / np.abs(combine_y[idx])
            c_total = combine_total[idx] / np.abs(combine_y[idx])
            t_stat = tunfold_stat[idx] / np.abs(tunfold_y[idx])
            t_syst = tunfold_syst[idx] / np.abs(tunfold_y[idx])
            t_total = tunfold_total[idx] / np.abs(tunfold_y[idx])

        ax.plot(centers, c_stat, "o-", color="#111827", label="Combine stat")
        ax.plot(centers, c_syst, "o-", color="#5790fc", label="Combine syst")
        ax.plot(centers, c_total, "o-", color="#e42536", label="Combine total")
        ax.plot(centers, t_stat, "s--", color="#6b7280", label="TUnfold stat")
        ax.plot(centers, t_syst, "s--", color="#f89c20", label="TUnfold syst")
        ax.plot(centers, t_total, "s--", color="#964a8b", label="TUnfold total")
        hi = int(pt_edges[ipt + 1]) if ipt + 1 < len(pt_edges) - 1 else None
        pt_range = f"{int(pt_edges[ipt])}" + (f"-{hi}" if hi is not None else "+") + " GeV"
        ax.set_xlabel(r"$\log_{10}(\rho^2)$", fontsize=PUB_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        ax.grid(True, alpha=0.25)
        shown.append(np.r_[c_total, t_total])
        hep.cms.label("Preliminary", data=True, loc=2, ax=ax, rlabel=pt_range)
    axes[0].set_ylabel("Fractional uncertainty", fontsize=PUB_LABEL_FONTSIZE)
    # Shared y (sharey=True): one headroom from the global max, no hard cap.
    # Empty bins divide by zero above, so drop non-finite values rather than
    # let a single inf disable the headroom entirely.
    finite = np.concatenate(shown) if shown else np.array([])
    finite = finite[np.isfinite(finite)]
    if finite.size and finite.max() > 0:
        axes[0].set_ylim(0.0, float(finite.max()) * 1.6)
    axes[0].legend(frameon=False, fontsize=PUB_LEGEND_FONTSIZE, ncol=2,
                   loc="upper right")
    stamp_figure(fig, inputs=out.parent.name)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")

    mode = args.mode
    groomed = mode == "groomed"
    combine_dir = args.combine_dir.resolve()
    stem = f"zjet_rho_{args.tag}_{mode}_combine"
    combine = np.load(combine_dir / f"{stem}_combine_unfolded.npz", allow_pickle=True)
    tunfold = np.load(args.tunfold_npz, allow_pickle=True)
    gen_edges_by_pt = json.loads(str(combine["gen_edges_json"].item()))
    pt_edges = np.asarray(combine["pt_edges"], dtype=float)

    unfolder = build_unfolder(args.tag, groomed)
    rows, global_row = bottom_line_rows(unfolder, combine)

    bottom_plot = combine_dir / f"{stem}_bottom_line_chi2.png"
    uncertainty_plot = combine_dir / f"{stem}_uncertainty_breakdown.png"
    summary_json = combine_dir / f"{stem}_diagnostics.json"
    plot_bottom_line(rows, global_row, bottom_plot)
    plot_uncertainty_breakdown(combine, tunfold, gen_edges_by_pt, pt_edges, uncertainty_plot)
    summary = {
        "bottom_line_scope": (
            "Reco uses fake-corrected data vs matched PYTHIA. Combine unfolded uses "
            "fitted full-truth spectrum vs nominal PYTHIA full truth. Stat covariance "
            "is the frozen-nuisance POI covariance; total covariance profiles all "
            "shape nuisances."
        ),
        "rows": rows,
        "global": global_row,
        "outputs": {
            "bottom_line": bottom_plot.name,
            "uncertainty_breakdown": uncertainty_plot.name,
        },
        "caveat": (
            "Combine source-by-source nuisance decomposition is not included here; "
            "this plot shows stat, profiled-syst, and total. Source breakdown needs "
            "grouped nuisance refits or impacts."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(bottom_plot)
    print(uncertainty_plot)
    print(summary_json)


if __name__ == "__main__":
    main()
