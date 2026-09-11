#!/usr/bin/env python3
"""Compare saved analytic and jackknife covariance matrices with fixed fakes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm
import mplhep as hep
import numpy as np


def correlation(covariance):
    sigma = np.sqrt(np.clip(np.diag(covariance), 0, None))
    denominator = np.outer(sigma, sigma)
    return np.divide(covariance, denominator, out=np.zeros_like(covariance),
                     where=denominator > 0)


def matrix_metrics(analytic, jackknife, blocks):
    ca, cj = correlation(analytic), correlation(jackknife)
    off_diagonal = ~np.eye(len(analytic), dtype=bool)
    between_pt = blocks[:, None] != blocks[None, :]
    return {
        "bins": len(analytic),
        "rank_analytic": int(np.linalg.matrix_rank(analytic)),
        "rank_jackknife": int(np.linalg.matrix_rank(jackknife)),
        "offdiagonal_correlation_RMS_difference": float(
            np.sqrt(np.mean((cj[off_diagonal] - ca[off_diagonal])**2))),
        "between_pt_median_abs_correlation_analytic": float(np.median(abs(ca[between_pt]))),
        "between_pt_median_abs_correlation_jackknife": float(np.median(abs(cj[between_pt]))),
        "median_error_ratio": float(np.median(np.sqrt(np.diag(jackknife) / np.diag(analytic)))),
        "relative_covariance_frobenius_difference": float(
            np.linalg.norm(jackknife - analytic) / np.linalg.norm(analytic)),
    }


def draw_matrices(analytic, jackknife, blocks, pt_edges, descriptor, provenance,
                  *, correlations_only=False):
    ca, cj = correlation(analytic), correlation(jackknife)
    matrices = [(ca, cj)] if correlations_only else [(analytic, jackknife), (ca, cj)]
    nrows = len(matrices)
    fig, axes = plt.subplots(nrows, 2, squeeze=False,
                             figsize=(21.2, 10.6*nrows), layout="constrained")
    footer = .105 if correlations_only else .055
    fig.get_layout_engine().set(rect=(0, footer, 1, 1-footer-.015))
    vmax = max(abs(analytic).max(), abs(jackknife).max())
    norm = SymLogNorm(linthresh=vmax/1000, vmin=-vmax, vmax=vmax)
    boundaries = np.flatnonzero(np.diff(blocks)) + 1
    edges = np.arange(len(analytic) + 1)
    groups = sorted(set(blocks))
    ticks = [np.flatnonzero(blocks == group).mean() + .5 for group in groups]
    labels = [f"{pt_edges[i]:g}–{pt_edges[i+1]:g}" if i < len(pt_edges)-2
              else f">{pt_edges[i]:g}" for i in groups]
    for row, pair in enumerate(matrices):
        is_correlation = correlations_only or row == 1
        for col, matrix in enumerate(pair):
            ax = axes[row, col]
            style = {"cmin": -1, "cmax": 1} if is_correlation else {"norm": norm}
            # mplhep provides the histogram mesh; a shared colorbar per row
            # uses matplotlib's constrained-layout support instead of the
            # per-axes divider used by hist2dplot's default colorbar.
            artist = hep.hist2dplot(matrix, edges, edges, ax=ax, cmap="RdBu_r",
                                    cbar=False, **style)
            for boundary in boundaries:
                ax.axvline(boundary, color="black", lw=1.3)
                ax.axhline(boundary, color="black", lw=1.3)
            ax.set_xticks(ticks, labels, rotation=25, ha="right", fontsize=23)
            ax.set_yticks(ticks, labels, fontsize=23)
            ax.set_xlabel(r"GEN bins grouped by $p_T$ [GeV]", fontsize=27)
            ax.set_ylabel(r"GEN bins grouped by $p_T$ [GeV]" if col == 0 else "", fontsize=27)
        colorbar = fig.colorbar(artist.pcolormesh, ax=axes[row, :], fraction=.035, pad=.025)
        colorbar.set_label("Correlation coefficient" if is_correlation
                           else "Covariance (symmetric log scale)", fontsize=25)
        if not is_correlation:
            decades = 10.0**np.arange(np.ceil(np.log10(norm.linthresh)),
                                      np.floor(np.log10(vmax)) + 1)
            colorbar.set_ticks(np.r_[-decades[::-1], 0, decades])
        colorbar.ax.tick_params(labelsize=22)
    fig.text(.5, footer*.66, descriptor, ha="center", fontsize=24)
    fig.text(.5, footer*.35,
             "10 replicas; analytic data term uses new diagonal sumw2; nominal fit weights retained",
             ha="center", fontsize=17)
    fig.text(.99, footer*.08, provenance, ha="right", fontsize=13)
    # Position the axes before mplhep computes CMS/status text spacing.
    fig.canvas.draw()
    for row in range(nrows):
        for col in range(2):
            # Dense matrices occupy the whole panel, so keep CMS above it.
            hep.cms.label("Internal", data=True, loc=0, ax=axes[row, col],
                          rlabel="Analytic" if col == 0 else "Jackknife")
    fig.canvas.draw()
    # Preserve the resolved constrained layout across PNG and PDF renderers.
    fig.set_layout_engine(None)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    destination = args.comparison / "covariance_comparison"
    destination.mkdir(exist_ok=True)
    report = json.loads((args.comparison / "comparison.json").read_text())
    provenance = f"2026-09-10 | unfold {report['git_head'][:7]} | hadronic JK Run 2, 20260908"
    hep.style.use(hep.style.CMS)
    output_metrics = []
    with PdfPages(destination / "covariance_comparison.pdf") as book:
        for run in report["runs"]:
            name = f"{run['channel']}_{run['mode']}"
            with np.load(args.comparison / f"{name}.npz") as arrays:
                mask = arrays["reported_mask"]
                n_bins = len(arrays["gen_edges"]) - 1
                blocks = np.repeat(np.arange(len(arrays["pt_edges"])-1), n_bins)[mask]
                # Data and MC samples are independent. Sum their covariance
                # estimates, not replicas carrying the same group label.
                total_a = arrays["analytic_data"] + arrays["analytic_response"]
                total_j = arrays["jackknife_data"] + arrays["jackknife_response"]
                row = {
                    "name": name,
                    "full_rank_analytic": int(np.linalg.matrix_rank(total_a)),
                    "full_rank_jackknife": int(np.linalg.matrix_rank(total_j)),
                    "components": {},
                }
                for component in ("total", "data", "response"):
                    analytic, jackknife = ((total_a, total_j) if component == "total" else
                                            (arrays[f"analytic_{component}"], arrays[f"jackknife_{component}"]))
                    analytic = analytic[np.ix_(mask, mask)]
                    jackknife = jackknife[np.ix_(mask, mask)]
                    row["components"][component] = matrix_metrics(analytic, jackknife, blocks)
                analytic = total_a[np.ix_(mask, mask)]
                jackknife = total_j[np.ix_(mask, mask)]
                descriptor = (f"{run['channel'].capitalize()}, {run['mode']} | data + response statistics | "
                              "fake correction fixed")
                for correlations_only in (False, True):
                    fig = draw_matrices(analytic, jackknife, blocks, arrays["pt_edges"],
                                        descriptor, provenance, correlations_only=correlations_only)
                    suffix = "_correlation" if correlations_only else ""
                    fig.savefig(destination / f"{name}{suffix}.png", dpi=110)
                    fig.savefig(destination / f"{name}{suffix}.pdf")
                    if not correlations_only:
                        book.savefig(fig)
                    plt.close(fig)
                output_metrics.append(row)
    (destination / "matrix_metrics.json").write_text(json.dumps(output_metrics, indent=2) + "\n")
    print(destination / "covariance_comparison.pdf")


if __name__ == "__main__":
    main()
