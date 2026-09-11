#!/usr/bin/env python3
"""Plot saved experimental jackknife covariance diagonals with mplhep."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    destination = args.comparison / "plots"
    destination.mkdir(exist_ok=False)
    report = json.loads((args.comparison / "comparison.json").read_text())
    hep.style.use(hep.style.CMS)
    # These are uncertainty magnitudes derived from full covariance matrices,
    # not event-count histograms. Plot the precomputed fractions with histplot;
    # a data/model histogram ratio would have different statistical semantics.
    with PdfPages(destination / "comparison.pdf") as book:
        for run in report["runs"]:
            channel, mode = run["channel"], run["mode"]
            with np.load(args.comparison / f"{channel}_{mode}.npz") as arrays:
                edges = arrays["gen_edges"]
                n_bins = len(edges) - 1
                for index, low in enumerate(arrays["pt_edges"][:-1]):
                    sl = slice(index*n_bins, (index+1)*n_bins)
                    mask = arrays["reported_mask"][sl]
                    selected = np.flatnonzero(mask)
                    assert np.array_equal(selected, np.arange(selected[0], selected[-1]+1))
                    shown_edges = edges[selected[0]:selected[-1]+2]
                    center = np.abs(arrays["center"][sl][mask])
                    assert np.all(center > 0)
                    high = arrays["pt_edges"][index+1]
                    pt_text = f"{low:g}–{high:g} GeV" if index < len(arrays["pt_edges"])-2 else f">{low:g} GeV"
                    for component in ("data", "mc"):
                        items = (
                            [("analytic_data", "Analytic, diagonal input sumw2", "#5790fc", "--"),
                             ("jackknife_data", "Jackknife, data", "#f89c20", "-")]
                            if component == "data" else
                            [("analytic_response", "TUnfold, response sumw2", "#5790fc", "--"),
                             ("jackknife_response", "Jackknife, response + misses", "#f89c20", "-"),
                             ("jackknife_full_mc", "Jackknife, also vary fake correction", "#964a8b", ":")]
                        )
                        fig, ax = plt.subplots(layout="constrained")
                        fig.get_layout_engine().set(rect=(0, .055, 1, .945))
                        largest = 0
                        for key, label, color, style in items:
                            fraction = np.sqrt(np.clip(np.diag(arrays[key])[sl][mask], 0, None))/center
                            largest = max(largest, fraction.max())
                            hep.histplot(fraction, shown_edges, yerr=False, ax=ax, label=label,
                                         color=color, linestyle=style, linewidth=2.8)
                        ax.set(xlim=(shown_edges[0], shown_edges[-1]), ylim=(0, largest*1.65),
                               xlabel=rf"$\log_{{10}}(\rho^2)$, {mode}", ylabel="Fractional statistical uncertainty")
                        ax.legend(loc="upper left", fontsize=19, title_fontsize=21,
                                  title=f"{channel.capitalize()}, {pt_text}")
                        hep.cms.label("Internal", data=True, loc=0, ax=ax, lumi=138, com=13)
                        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                        fig.text(.99, .022, "10 replicas; fixed nominal fit weights; selected comparison window",
                                 ha="right", fontsize=11)
                        fig.text(.99, .005, f"2026-09-10 | unfold {report['git_head'][:7]} | hadronic JK Run 2, 20260908",
                                 ha="right", fontsize=10)
                        name = f"{channel}_{mode}_pt{index}_{component}"
                        fig.savefig(destination / f"{name}.png", dpi=110)
                        fig.savefig(destination / f"{name}.pdf")
                        book.savefig(fig)
                        plt.close(fig)
    print(destination / "comparison.pdf")


if __name__ == "__main__":
    main()
