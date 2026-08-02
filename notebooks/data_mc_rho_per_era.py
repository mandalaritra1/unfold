#!/usr/bin/env python3
"""Per-era detector-level data/MC validation plots in rho.

Same style, stack, and two-leg PS/HAD modelling band as the combined Run 2 plots
made by ``data_mc_rho_fancy.py`` -- this script only changes what is summed:

  * one plot per data-taking era instead of the Run 2 sum, and
  * integrated over the reported jet-pT bins instead of showing a single one.

The 185-200 GeV bin is a buffer in pT (unfolded, not reported) and is excluded
from the integral. The model samples are produced inclusively over the eras, so
every era gets the same Run 2 modelling shape band; that is intended, since the
parton-shower and hadronization shape difference is a property of the generator
rather than of the data-taking conditions.

Run: source .venv/bin/activate && python notebooks/data_mc_rho_per_era.py
Outputs: outputs/zjet/rho/data_mc_per_era/data_mc_rho_<groomed|ungroomed>_<era>.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_mc_rho_fancy import (  # noqa: E402
    ERA_DATA_DATASETS,
    ERA_DATASET_TAGS,
    ERA_LUMI_FBINV,
    ERAS,
    PLOT_CONFIGS,
    PROCESS_DATASETS,
    REPORTED_PT_INDICES,
    ROOT,
    STACK_ORDER,
    combine_dy_systematics,
    histogram_arrays,
    load_pickle,
    make_plot,
    model_ps_had_band,
    model_reco_shapes,
    total_mc_uncertainty,
    unfold_reco_display_edges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "inputs" / "zjet" / "validation",
        help="Directory containing validation_*.pkl inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "zjet" / "rho" / "data_mc_per_era",
        help="Directory for the CMS Internal plots.",
    )
    parser.add_argument(
        "--eras",
        nargs="+",
        default=list(ERAS),
        choices=list(ERAS),
        help="Subset of data-taking eras to plot.",
    )
    return parser.parse_args()


def render_era(era, mode, config, data, backgrounds, single_top, pythia_inputs,
               output_dir):
    """Build one (era, grooming) plot and return the written path."""
    histogram_name = config["histogram"]
    xmin = config["xmin"]
    groomed = mode == "groomed"
    pt_indices = REPORTED_PT_INDICES
    target_edges = unfold_reco_display_edges(groomed, xmin)

    data_values, data_variances, rho_edges = histogram_arrays(
        data[histogram_name],
        ERA_DATA_DATASETS[era],
        xmin,
        target_edges=target_edges,
        pt_indices=pt_indices,
    )

    process_values = {}
    process_variances = {}
    tag = ERA_DATASET_TAGS[era]

    for process in ("WW", "WZ", "ZZ", "tt+jets"):
        values, variances, _ = histogram_arrays(
            backgrounds[histogram_name],
            [PROCESS_DATASETS[process](tag)],
            xmin,
            target_edges=target_edges,
            pt_indices=pt_indices,
        )
        process_values[process] = values
        process_variances[process] = variances

    values, variances, _ = histogram_arrays(
        single_top[histogram_name],
        [PROCESS_DATASETS["Single top"](tag)],
        xmin,
        target_edges=target_edges,
        pt_indices=pt_indices,
    )
    process_values["Single top"] = values
    process_variances["Single top"] = variances

    dy_systematics, dy_variance, dy_edges = combine_dy_systematics(
        pythia_inputs,
        histogram_name,
        xmin,
        target_edges=target_edges,
        eras=(era,),
        pt_indices=pt_indices,
    )
    if not np.array_equal(rho_edges, dy_edges):
        raise ValueError(f"Inconsistent rho binning for {mode} in {era}")
    process_values["DY signal"] = dy_systematics["nominal"]
    process_variances["DY signal"] = dy_variance

    total_mc_variance = np.sum(
        [process_variances[process] for process in STACK_ORDER],
        axis=0,
    )
    uncertainty_up, uncertainty_down = total_mc_uncertainty(
        dy_systematics,
        total_mc_variance,
    )

    model_shapes = model_reco_shapes(
        histogram_name, xmin, target_edges, pt_indices=pt_indices
    )
    model_band, _, _ = model_ps_had_band(dy_systematics["nominal"], model_shapes)
    uncertainty_up = np.sqrt(uncertainty_up ** 2 + model_band ** 2)
    uncertainty_down = np.sqrt(uncertainty_down ** 2 + model_band ** 2)

    # Shape comparison in the reported window, as for the Run 2 plots.
    mc_integral = float(sum(values.sum() for values in process_values.values()))
    data_integral = float(data_values.sum())
    norm = data_integral / mc_integral if mc_integral > 0 else 1.0
    process_values = {name: values * norm for name, values in process_values.items()}
    uncertainty_up = uncertainty_up * norm
    uncertainty_down = uncertainty_down * norm

    output_path = output_dir / f"data_mc_rho_{mode}_{era}.pdf"
    make_plot(
        data_values,
        data_variances,
        process_values,
        uncertainty_up,
        uncertainty_down,
        rho_edges,
        "Internal",
        groomed,
        ERA_LUMI_FBINV[era],
        output_path,
        pt_range=None,
        era_label=era,
    )

    total_data = data_values.sum()
    total_mc = sum(values.sum() for values in process_values.values())
    print(
        f"{era} {mode}: data={total_data:.0f}, MC={total_mc:.1f}, "
        f"Data/MC={total_data / total_mc:.4f}"
    )
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [
        args.input_dir / "validation_backgrounds_all.pkl",
        args.input_dir / "validation_st_all.pkl",
        args.input_dir / "validation_data.pkl",
        *[args.input_dir / f"validation_pythia_{era}.pkl" for era in args.eras],
    ]
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{listing}")

    backgrounds = load_pickle(input_paths[0])
    single_top = load_pickle(input_paths[1])
    data = load_pickle(input_paths[2])
    pythia_inputs = {
        era: load_pickle(args.input_dir / f"validation_pythia_{era}.pkl")
        for era in args.eras
    }

    output_paths = []
    for era in args.eras:
        for mode, config in PLOT_CONFIGS.items():
            output_paths.append(
                render_era(
                    era, mode, config, data, backgrounds, single_top,
                    pythia_inputs, args.output_dir,
                )
            )

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
