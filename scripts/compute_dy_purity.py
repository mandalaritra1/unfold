#!/usr/bin/env python3
"""Compute the DY composition purity from validation MC pickle inputs.

Purity here means:

    DY / (DY + WW + WZ + ZZ + ttjets + single-top)

using the nominal systematic variation. This is a sample-composition purity,
not the response-matrix bin purity used for unfolding diagnostics.
"""

from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path


ERA_DATASET_TAGS = {
    "2016": "UL16NanoAODv9",
    "2016APV": "UL16NanoAODAPVv9",
    "2017": "UL17NanoAODv9",
    "2018": "UL18NanoAODv9",
}

BACKGROUND_PROCESSES = ("ww", "wz", "zz", "ttjets")


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def weighted_sum_value(histogram) -> float:
    summed = histogram.sum()
    return float(getattr(summed, "value", summed))


def nominal_yield(histograms, hist_name: str, dataset_name: str) -> float:
    histogram = histograms[hist_name]
    selection = {"dataset": dataset_name}
    if "systematic" in histogram.axes.name:
        selection["systematic"] = "nominal"
    return weighted_sum_value(histogram[selection])


def compute_era_yields(input_dir: Path, hist_name: str, era: str) -> dict[str, float]:
    era_tag = ERA_DATASET_TAGS[era]
    pythia = load_pickle(input_dir / f"validation_pythia_{era}.pkl")
    backgrounds = load_pickle(input_dir / "validation_backgrounds_all.pkl")
    single_top = load_pickle(input_dir / "validation_st_all.pkl")

    dy_yield = nominal_yield(pythia, hist_name, f"pythia_{era_tag}")
    component_yields = {
        process: nominal_yield(backgrounds, hist_name, f"{process}_{era_tag}")
        for process in BACKGROUND_PROCESSES
    }
    component_yields["ST"] = nominal_yield(single_top, hist_name, f"ST_{era_tag}")

    background_yield = sum(component_yields.values())
    total_yield = dy_yield + background_yield
    purity = dy_yield / total_yield if total_yield > 0 else float("nan")

    return {
        "dy": dy_yield,
        "background": background_yield,
        "total": total_yield,
        "purity": purity,
        **component_yields,
    }


def format_float(value: float) -> str:
    return f"{value:,.6g}"


def print_text_table(args, all_yields, combined_dy, combined_background, combined_total, combined_purity):
    print(f"Histogram: {args.hist}")
    print(f"Inputs: {args.input_dir}")
    print()
    print(f"{'era':<8} {'DY':>14} {'non-DY MC':>14} {'total MC':>14} {'DY purity':>12}")
    print("-" * 68)
    for era, yields in all_yields.items():
        print(
            f"{era:<8} "
            f"{format_float(yields['dy']):>14} "
            f"{format_float(yields['background']):>14} "
            f"{format_float(yields['total']):>14} "
            f"{100.0 * yields['purity']:>11.4f}%"
        )
    print("-" * 68)
    print(
        f"{'Run2':<8} "
        f"{format_float(combined_dy):>14} "
        f"{format_float(combined_background):>14} "
        f"{format_float(combined_total):>14} "
        f"{100.0 * combined_purity:>11.4f}%"
    )


def print_latex_table(args, all_yields, combined_dy, combined_background, combined_total, combined_purity):
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{DY sample purity in the validation selection.}")
    print(r"  \label{tab:dy-purity-validation}")
    print(r"  \begin{tabular}{lr}")
    print(r"    \hline")
    print(r"    Data taking period & Purity [\%] \\")
    print(r"    \hline")
    for era, yields in all_yields.items():
        print(
            f"    {era} & {100.0 * yields['purity']:.4f} \\\\"
        )
    print(r"    \hline")
    print(
        f"    Run 2 & {100.0 * combined_purity:.4f} \\\\"
    )
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DY / total MC purity from validation pickle inputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs/validation"),
        help="Directory containing validation_*.pkl inputs.",
    )
    parser.add_argument(
        "--hist",
        default="pt_Z",
        help="Histogram key to integrate. Use a one-entry-per-event histogram for event purity.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a paste-ready LaTeX table instead of the plain text table.",
    )
    args = parser.parse_args()

    all_yields = {
        era: compute_era_yields(args.input_dir, args.hist, era)
        for era in ERA_DATASET_TAGS
    }

    combined_dy = sum(yields["dy"] for yields in all_yields.values())
    combined_background = sum(yields["background"] for yields in all_yields.values())
    combined_total = combined_dy + combined_background
    combined_purity = combined_dy / combined_total if combined_total > 0 else float("nan")

    if args.latex:
        print_latex_table(args, all_yields, combined_dy, combined_background, combined_total, combined_purity)
    else:
        print_text_table(args, all_yields, combined_dy, combined_background, combined_total, combined_purity)


if __name__ == "__main__":
    main()
