#!/usr/bin/env python3
"""Make combined Run 2 detector-level data/MC validation plots in rho.

The input histograms already contain the nominal Z+jet event selection and
luminosity-normalized MC weights. The displayed result uses the detector-level
200--290 GeV jet-pT bin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle as pkl
import shlex
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ERAS = ("2016APV", "2016", "2017", "2018")
ERA_DATASET_TAGS = {
    "2016APV": "UL16NanoAODAPVv9",
    "2016": "UL16NanoAODv9",
    "2017": "UL17NanoAODv9",
    "2018": "UL18NanoAODv9",
}
DATASETS = (
    "SingleElectron_UL2016APV",
    "SingleMuon_UL2016APV",
    "SingleElectron_UL2016",
    "SingleMuon_UL2016",
    "SingleElectron_UL2017",
    "SingleMuon_UL2017",
    "EGamma_UL2018",
    "SingleMuon_UL2018",
)
PROCESS_DATASETS = {
    "WW": lambda tag: f"ww_{tag}",
    "WZ": lambda tag: f"wz_{tag}",
    "ZZ": lambda tag: f"zz_{tag}",
    "tt+jets": lambda tag: f"ttjets_{tag}",
    "Single top": lambda tag: f"ST_{tag}",
    "DY signal": lambda tag: f"pythia_{tag}",
}
PLOT_CONFIGS = {
    "ungroomed": {
        "histogram": "ptjet_rhojet_u_reco",
        "xmin": -2.5,
        "output_name": "data_mc_rho_ungroomed_run2.pdf",
    },
    "groomed": {
        "histogram": "ptjet_rhojet_g_reco",
        "xmin": -4.5,
        "output_name": "data_mc_rho_groomed_run2.pdf",
    },
}
PT_BIN_INDEX = 1
PT_RANGE_GEV = (200, 290)
STACK_ORDER = ("WW", "WZ", "ZZ", "tt+jets", "Single top", "DY signal")
STACK_LABELS = {
    "WW": "WW",
    "WZ": "WZ",
    "ZZ": "ZZ",
    "tt+jets": r"$t\bar{t}+\mathrm{jets}$",
    "Single top": "Single t",
    "DY signal": "DYJets",
}
STACK_COLORS = {
    "WW": "blue",
    "WZ": "green",
    "ZZ": "orange",
    "tt+jets": "violet",
    "Single top": "pink",
    "DY signal": "red",
}


hep.style.use("CMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create combined Run 2 detector-level rho data/MC plots."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "inputs" / "zjet" / "validation",
        help="Directory containing validation_*.pkl inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "zjet" / "rho" / "data_mc",
        help="Directory for the CMS Internal plots and provenance JSON.",
    )
    parser.add_argument(
        "--input-production-tag",
        default="validation",
        help="Production tag recorded in the provenance JSON.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=138.0,
        help="Integrated luminosity displayed on the plots in fb^-1.",
    )
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Do not refresh <output-dir>/index.html and cached PNG previews.",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def histogram_arrays(histogram, dataset_names, xmin, systematic="nominal"):
    """Sum selected datasets and reported reco-pT bins into displayed rho bins."""
    rho_edges = np.asarray(histogram.axes["mpt_reco"].edges, dtype=float)
    rho_mask = (rho_edges[:-1] >= xmin) & (rho_edges[1:] <= 0.0)
    selected_edges = np.concatenate((rho_edges[:-1][rho_mask], [rho_edges[1:][rho_mask][-1]]))

    dataset_indices = [histogram.axes["dataset"].index(name) for name in dataset_names]
    systematic_index = histogram.axes["systematic"].index(systematic)

    values = histogram.values()[dataset_indices, PT_BIN_INDEX, :, systematic_index]
    variances = histogram.variances()[dataset_indices, PT_BIN_INDEX, :, systematic_index]
    values = values.sum(axis=0)[rho_mask]
    variances = variances.sum(axis=0)[rho_mask]
    return values, variances, selected_edges


def combine_dy_systematics(pythia_inputs, histogram_name, xmin):
    combined = {}
    combined_variance = None
    rho_edges = None

    for era in ERAS:
        histogram = pythia_inputs[era][histogram_name]
        dataset_name = PROCESS_DATASETS["DY signal"](ERA_DATASET_TAGS[era])
        for systematic in histogram.axes["systematic"]:
            values, variances, current_edges = histogram_arrays(
                histogram,
                [dataset_name],
                xmin,
                systematic=systematic,
            )
            combined[systematic] = combined.get(systematic, np.zeros_like(values)) + values
            if systematic == "nominal":
                if combined_variance is None:
                    combined_variance = np.zeros_like(variances)
                combined_variance += variances
                rho_edges = current_edges

    return combined, combined_variance, rho_edges


def total_mc_uncertainty(dy_systematics, total_mc_variance):
    """Match the existing plot convention: quadrature sums of Up/Down shifts."""
    nominal = dy_systematics["nominal"]
    uncertainty_up_sq = np.array(total_mc_variance, copy=True)
    uncertainty_down_sq = np.array(total_mc_variance, copy=True)

    for systematic, varied in dy_systematics.items():
        shift_sq = np.square(varied - nominal)
        if systematic.endswith("Up"):
            uncertainty_up_sq += shift_sq
        elif systematic.endswith("Down"):
            uncertainty_down_sq += shift_sq

    return np.sqrt(uncertainty_up_sq), np.sqrt(uncertainty_down_sq)


def make_plot(
    data_values,
    data_variances,
    process_values,
    uncertainty_up,
    uncertainty_down,
    rho_edges,
    cms_label,
    groomed,
    lumi,
    output_path,
):
    fig, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )

    stack_values = [process_values[process] for process in STACK_ORDER]
    hep.histplot(
        stack_values,
        bins=rho_edges,
        stack=True,
        histtype="fill",
        label=[STACK_LABELS[process] for process in STACK_ORDER],
        color=[STACK_COLORS[process] for process in STACK_ORDER],
        ax=axis,
    )

    total_mc = np.sum(stack_values, axis=0)
    data_errors = np.sqrt(data_variances)
    centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])

    axis.errorbar(
        centers,
        data_values,
        yerr=data_errors,
        fmt="o",
        color="black",
        markersize=5,
        label="Data",
        zorder=10,
    )
    axis.stairs(
        total_mc + uncertainty_up,
        baseline=np.clip(total_mc - uncertainty_down, 0.0, None),
        edges=rho_edges,
        label="Total Unc.",
        hatch="///",
        edgecolor="black",
        facecolor="none",
        fill=True,
        alpha=1.0,
        zorder=9,
    )

    axis.set_yscale("log")
    axis.set_ylim(0.1, max(1.0, float(data_values.max()) * 100.0))
    axis.set_ylabel("#Events")
    axis.legend(ncol=4, fontsize=17)
    displayed_lumi = int(lumi) if float(lumi).is_integer() else lumi
    hep.cms.label(cms_label, data=True, lumi=displayed_lumi, com=13, ax=axis)
    axis.text(
        0.98,
        0.76,
        rf"${PT_RANGE_GEV[0]} < p_T < {PT_RANGE_GEV[1]}$ GeV",
        transform=axis.transAxes,
        horizontalalignment="right",
        fontsize=17,
    )

    ratio = np.divide(
        data_values,
        total_mc,
        out=np.full_like(data_values, np.nan, dtype=float),
        where=total_mc > 0,
    )
    ratio_error = np.divide(
        data_errors,
        total_mc,
        out=np.zeros_like(data_errors),
        where=total_mc > 0,
    )
    relative_up = np.divide(
        uncertainty_up,
        total_mc,
        out=np.zeros_like(uncertainty_up),
        where=total_mc > 0,
    )
    relative_down = np.divide(
        uncertainty_down,
        total_mc,
        out=np.zeros_like(uncertainty_down),
        where=total_mc > 0,
    )

    ratio_axis.errorbar(
        centers,
        ratio,
        yerr=ratio_error,
        fmt="o",
        color="black",
        markersize=5,
    )
    ratio_axis.stairs(
        1.0 + relative_up,
        baseline=1.0 - relative_down,
        edges=rho_edges,
        hatch="///",
        edgecolor="black",
        facecolor="none",
        fill=True,
        alpha=1.0,
    )
    ratio_axis.axhline(1.0, color="red", linestyle="--")
    ratio_axis.set_ylim(0.0, 2.0)
    ratio_axis.set_ylabel("Data/MC")
    ratio_axis.set_xlabel(
        r"$\log_{10}(\rho^2)$, Groomed"
        if groomed
        else r"$\log_{10}(\rho^2)$, Ungroomed"
    )
    ratio_axis.set_xlim(rho_edges[0], rho_edges[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(args, input_paths, output_paths):
    try:
        input_dir = args.input_dir.resolve().relative_to(ROOT)
    except ValueError:
        input_dir = args.input_dir.resolve()

    provenance = {
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "input_production_tag": args.input_production_tag,
        "input_directory": str(input_dir),
        "input_files": {
            path.name: {"sha256": file_sha256(path)}
            for path in input_paths
        },
        "configuration": {
            "eras": list(ERAS),
            "integrated_luminosity_fb-1": args.lumi,
            "center_of_mass_energy_TeV": 13,
            "reco_pt_range_GeV": list(PT_RANGE_GEV),
            "rho_ranges": {
                name: [config["xmin"], 0.0]
                for name, config in PLOT_CONFIGS.items()
            },
            "mc_normalization": "absolute luminosity normalization from input histograms",
            "total_mc_uncertainty": (
                "DY systematic Up/Down shifts added in quadrature with statistical "
                "variances from all stacked MC processes"
            ),
            "background_systematics": (
                "not available in validation background inputs; nominal MC statistical "
                "uncertainties are included"
            ),
        },
        "outputs": [str(path.resolve().relative_to(ROOT)) for path in output_paths],
    }

    for directory in (args.output_dir, args.output_dir / "Preliminary"):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "run2_plot_config.json").open("w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2)
            handle.write("\n")


def build_gallery(output_dir: Path) -> Path:
    """Build the same static PDF-preview gallery used by the rho outputs."""
    gallery_path = output_dir / "index.html"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "outputs" / "build_rho_gallery.py"),
            "--root",
            str(output_dir),
            "--title",
            "Z+jet data/MC validation gallery",
        ],
        check=True,
    )
    return gallery_path


def main() -> None:
    args = parse_args()
    input_paths = [
        args.input_dir / "validation_backgrounds_all.pkl",
        args.input_dir / "validation_st_all.pkl",
        args.input_dir / "validation_data.pkl",
        *[
            args.input_dir / f"validation_pythia_{era}.pkl"
            for era in ERAS
        ],
    ]
    missing_inputs = [path for path in input_paths if not path.is_file()]
    if missing_inputs:
        missing = "\n".join(f"  {path}" for path in missing_inputs)
        raise FileNotFoundError(f"Missing required input files:\n{missing}")

    backgrounds = load_pickle(input_paths[0])
    single_top = load_pickle(input_paths[1])
    data = load_pickle(input_paths[2])
    pythia_inputs = {
        era: load_pickle(args.input_dir / f"validation_pythia_{era}.pkl")
        for era in ERAS
    }

    output_paths = []
    for mode, config in PLOT_CONFIGS.items():
        histogram_name = config["histogram"]
        xmin = config["xmin"]

        data_values, data_variances, rho_edges = histogram_arrays(
            data[histogram_name],
            DATASETS,
            xmin,
        )
        process_values = {}
        process_variances = {}

        for process in ("WW", "WZ", "ZZ", "tt+jets"):
            dataset_names = [
                PROCESS_DATASETS[process](ERA_DATASET_TAGS[era])
                for era in ERAS
            ]
            values, variances, _ = histogram_arrays(
                backgrounds[histogram_name],
                dataset_names,
                xmin,
            )
            process_values[process] = values
            process_variances[process] = variances

        single_top_datasets = [
            PROCESS_DATASETS["Single top"](ERA_DATASET_TAGS[era])
            for era in ERAS
        ]
        values, variances, _ = histogram_arrays(
            single_top[histogram_name],
            single_top_datasets,
            xmin,
        )
        process_values["Single top"] = values
        process_variances["Single top"] = variances

        dy_systematics, dy_variance, dy_edges = combine_dy_systematics(
            pythia_inputs,
            histogram_name,
            xmin,
        )
        if not np.array_equal(rho_edges, dy_edges):
            raise ValueError(f"Inconsistent rho binning for {mode}")
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

        for cms_label, subdirectory in (("Internal", Path()), ("Preliminary", Path("Preliminary"))):
            output_path = args.output_dir / subdirectory / config["output_name"]
            make_plot(
                data_values,
                data_variances,
                process_values,
                uncertainty_up,
                uncertainty_down,
                rho_edges,
                cms_label,
                mode == "groomed",
                args.lumi,
                output_path,
            )
            output_paths.append(output_path)

        total_data = data_values.sum()
        total_mc = sum(values.sum() for values in process_values.values())
        print(
            f"{mode}: data={total_data:.0f}, MC={total_mc:.1f}, "
            f"Data/MC={total_data / total_mc:.4f}"
        )

    write_provenance(args, input_paths, output_paths)
    if not args.no_gallery:
        gallery_path = build_gallery(args.output_dir)
        print(f"Gallery: {gallery_path}")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
