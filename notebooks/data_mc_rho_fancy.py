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
sys.path.insert(0, str(ROOT / "src"))

from unfold.tools import binning
from unfold.utils.cms_plot import save_cms_label_flavors
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
        # ARC round-2: the groomed result is not shown below the display floor,
        # so trim the data/MC comparison to the same reported window.
        "xmin": -3.5,
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
    "DY signal": "DY+jets",
}
# CMS CVD-friendly Petroff 6-color scheme (arXiv:2107.02270) -- ARC round-2
# asked whether the stack follows the official palette.
STACK_COLORS = {
    "WW": "#5790fc",
    "WZ": "#7a21dd",
    "ZZ": "#f89c20",
    "tt+jets": "#964a8b",
    "Single top": "#9c9ca1",
    "DY signal": "#e42536",
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


def unfold_reco_display_edges(groomed, xmin):
    """The unfolding's reco rho edges (arc_r2 axes, displayed pT bin) clipped
    to the display window — the data/MC comparison uses the same binning as
    the response matrix. Groomed ends in the 0.5-wide [-1,-0.5], [-0.5,0]
    bins; ungroomed reco stays uniform 0.25 (only its GEN axis merges [-1,0])."""
    edges = np.asarray(
        binning.bin_edges(groomed).reco_rho_edges_by_pt_arcr2[PT_BIN_INDEX],
        dtype=float,
    )
    return edges[(edges >= xmin - 1e-9) & (edges <= 1e-9)]


def rebin_sum(values, edges, target_edges):
    """Merge fine bins into target bins (target edges must nest in edges)."""
    idx = np.searchsorted(edges, target_edges)
    if not np.allclose(np.asarray(edges)[idx], target_edges):
        raise ValueError("target edges are not a subset of the histogram edges")
    return np.add.reduceat(values, idx[:-1])


def histogram_arrays(histogram, dataset_names, xmin, systematic="nominal",
                     target_edges=None):
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
    if target_edges is not None:
        values = rebin_sum(values, selected_edges, target_edges)
        variances = rebin_sum(variances, selected_edges, target_edges)
        selected_edges = np.asarray(target_edges, dtype=float)
    return values, variances, selected_edges


def combine_dy_systematics(pythia_inputs, histogram_name, xmin, target_edges=None):
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
                target_edges=target_edges,
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


# Two-leg parton-shower / hadronization modelling uncertainty (ARC round-2,
# plan WS4.1): PS = Vincia shower swap; HAD = envelope over CR modes 1/2 and
# Lund fragmentation hard/soft. Reco-level gen-reweighted Pythia spectra staged
# from the reweight_pythia_rho production. Combined in quadrature (independent
# legs), then added to the detector+ME band.
MODEL_DIR = ROOT / "inputs" / "zjet" / "rho" / "model_reco"
MODEL_PS_SOURCE = "vincia"
MODEL_HAD_SOURCES = ("cr1", "cr2", "fraghard", "fragsoft")


def model_reco_shapes(histogram_name, xmin, target_edges):
    """Reco DY shape for each model source at PT_BIN_INDEX, summed over eras
    and rebinned from the fine model axis onto the display edges."""
    shapes = {}
    for source in (MODEL_PS_SOURCE, *MODEL_HAD_SOURCES):
        histogram = load_pickle(MODEL_DIR / f"{source}_all.pkl")[histogram_name]
        edges = np.asarray(histogram.axes["mpt_reco"].edges, dtype=float)
        rho_mask = (edges[:-1] >= xmin - 1e-9) & (edges[1:] <= 1e-9)
        selected_edges = np.concatenate((edges[:-1][rho_mask], [edges[1:][rho_mask][-1]]))
        syst_index = histogram.axes["systematic"].index("nominal")
        values = histogram.values()[:, PT_BIN_INDEX, :, syst_index].sum(axis=0)[rho_mask]
        shapes[source] = rebin_sum(values, selected_edges, target_edges)
    return shapes


def model_ps_had_band(dy_nominal, shapes):
    """|dy_nominal - source| after normalizing each source to the DY integral so
    only the SHAPE enters. PS = Vincia; HAD = envelope over CR/frag; quadrature."""
    dy_sum = dy_nominal.sum()

    def shape_diff(alt):
        a_sum = alt.sum()
        if a_sum <= 0 or dy_sum <= 0:
            return np.zeros_like(dy_nominal)
        return np.abs(dy_nominal - alt * (dy_sum / a_sum))

    ps = shape_diff(shapes[MODEL_PS_SOURCE])
    had = np.max([shape_diff(shapes[s]) for s in MODEL_HAD_SOURCES], axis=0)
    return np.sqrt(ps ** 2 + had ** 2), ps, had


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
        # ARC round-2: ratio panel close to the main panel
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.07},
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
    # ARC round-2: extra y-headroom so the legend clears the stack and the
    # pT range fits below it, left-aligned to the legend's left edge.
    axis.set_ylim(0.1, max(1.0, float(data_values.max()) * 1000.0))
    axis.set_ylabel("Events")
    axis.legend(ncol=4, fontsize=17, loc="upper left")
    displayed_lumi = int(lumi) if float(lumi).is_integer() else lumi
    hep.cms.label(cms_label, data=True, lumi=displayed_lumi, com=13, ax=axis)
    axis.text(
        0.02,
        0.72,
        rf"${PT_RANGE_GEV[0]} < p_{{\mathrm{{T}}}} < {PT_RANGE_GEV[1]}$ GeV",
        transform=axis.transAxes,
        horizontalalignment="left",
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
    # No tick label at 0 or 2: the corner "0" collides with the first x tick
    # label, and the top "2" crowds the main panel above.
    ratio_axis.set_yticks([0.5, 1.0, 1.5])
    ratio_axis.set_ylabel("Data/MC")
    # ARC round-2: lowercase groomed/ungroomed
    ratio_axis.set_xlabel(
        r"$\log_{10}(\rho^2)$, groomed"
        if groomed
        else r"$\log_{10}(\rho^2)$, ungroomed"
    )
    ratio_axis.set_xlim(rho_edges[0], rho_edges[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # One draw, all label flavors (Internal at top level + <Flavor>/ siblings).
    save_cms_label_flavors(fig, output_path, cms_label)
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def write_provenance(args, input_paths, output_paths):
    input_dir = _repo_relative(args.input_dir)

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
        "outputs": [_repo_relative(path) for path in output_paths],
    }

    for directory in (args.output_dir, args.output_dir / "Preliminary",
                      args.output_dir / "PrivateWork"):
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
        # Display in the unfolding's reco binning (e.g. groomed [-1,-0.5,0]
        # right-most bins), not the raw uniform validation axis.
        target_edges = unfold_reco_display_edges(mode == "groomed", xmin)

        data_values, data_variances, rho_edges = histogram_arrays(
            data[histogram_name],
            DATASETS,
            xmin,
            target_edges=target_edges,
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
                target_edges=target_edges,
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
            target_edges=target_edges,
        )
        process_values["Single top"] = values
        process_variances["Single top"] = variances

        dy_systematics, dy_variance, dy_edges = combine_dy_systematics(
            pythia_inputs,
            histogram_name,
            xmin,
            target_edges=target_edges,
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

        # ARC round-2 (WS4.1): add the two-leg PS/HAD modelling uncertainty of
        # the DY shape in quadrature with the detector+ME band, so the data/MC
        # ratio is covered by a band that carries the modelling difference.
        model_shapes = model_reco_shapes(histogram_name, xmin, target_edges)
        model_band, model_ps, model_had = model_ps_had_band(
            dy_systematics["nominal"], model_shapes
        )
        uncertainty_up = np.sqrt(uncertainty_up ** 2 + model_band ** 2)
        uncertainty_down = np.sqrt(uncertainty_down ** 2 + model_band ** 2)

        # Normalize the total MC to the data integral over the shown range only
        # (ARC round-2: the comparison is a shape test in the reported window, so
        # the overall luminosity/k-factor offset is divided out). The same scale
        # multiplies every stack component and the uncertainty band, so the
        # fractional band is preserved and the ratio panel centers on 1.
        mc_integral = float(sum(values.sum() for values in process_values.values()))
        data_integral = float(data_values.sum())
        norm = data_integral / mc_integral if mc_integral > 0 else 1.0
        process_values = {name: values * norm for name, values in process_values.items()}
        uncertainty_up = uncertainty_up * norm
        uncertainty_down = uncertainty_down * norm

        # Drawn once with the Internal label; save_cms_label_flavors inside
        # make_plot writes the Preliminary/PrivateWork siblings.
        output_path = args.output_dir / config["output_name"]
        make_plot(
            data_values,
            data_variances,
            process_values,
            uncertainty_up,
            uncertainty_down,
            rho_edges,
            "Internal",
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
