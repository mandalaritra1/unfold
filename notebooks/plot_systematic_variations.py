#!/usr/bin/env python3
"""Plot Run 2 DY systematic variations for selected validation variables.

The validation inputs already contain the nominal Z+jet selection and
luminosity-normalized event weights. This script combines the four Run 2 eras
and compares one or more Up/Down pairs with the nominal DY prediction.
"""

from __future__ import annotations

import argparse
import pickle as pkl
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
VARIABLE_CONFIGS = {
    "ptjet_rhojet_u_reco": {
        "xlabel": r"$\log_{10}(\rho^2)$, Ungroomed",
        "xlim": (-2.5, 0.0),
        "plot_axis": "mpt_reco",
        "axis_indices": {"ptreco": 1},
        "selection_label": r"$200 < p_T^\mathrm{reco} < 290$ GeV",
    },
    "ptjet_rhojet_g_reco": {
        "xlabel": r"$\log_{10}(\rho^2)$, Groomed",
        "xlim": (-4.5, 0.0),
        "plot_axis": "mpt_reco",
        "axis_indices": {"ptreco": 1},
        "selection_label": r"$200 < p_T^\mathrm{reco} < 290$ GeV",
    },
    "mass_jet0": {
        "xlabel": r"Leading jet mass [GeV]",
        "xlim": (0.0, 500.0),
    },
    "pt_jet0": {
        "xlabel": r"Leading jet $p_T$ [GeV]",
        "xlim": (200.0, 1000.0),
    },
    "eta_jet0": {
        "xlabel": r"Leading jet $\eta$",
        "xlim": (-2.5, 2.5),
    },
    "y_jet0": {
        "xlabel": r"Leading jet $y$",
        "xlim": (-2.5, 2.5),
    },
    "phi_jet0": {
        "xlabel": r"Leading jet $\phi$",
        "xlim": (-3.2, 3.2),
    },
    "nJets": {
        "xlabel": "Number of jets",
        "xlim": (0.0, 5.0),
    },
    "mass_Z": {
        "xlabel": r"$m_{\ell\ell}$ [GeV]",
        "xlim": (70.0, 110.0),
    },
    "pt_Z": {
        "xlabel": r"Z candidate $p_T$ [GeV]",
        "xlim": (90.0, 1000.0),
    },
    "eta_Z": {
        "xlabel": r"Z candidate $\eta$",
        "xlim": (-2.5, 2.5),
    },
    "phi_Z": {
        "xlabel": r"Z candidate $\phi$",
        "xlim": (-3.2, 3.2),
    },
    "dphi": {
        "xlabel": r"$\Delta\phi(Z,\mathrm{jet})$",
        "xlim": (0.0, 3.2),
    },
    "dr": {
        "xlabel": r"$\Delta R(Z,\mathrm{jet})$",
        "xlim": None,
    },
    "ptasym": {
        "xlabel": r"$p_T$ asymmetry",
        "xlim": None,
    },
}
SYSTEMATIC_DISPLAY_NAMES = {
    "eleid": "Electron ID",
    "elereco": "Electron reconstruction",
    "eletrig": "Electron trigger",
    "fsr": "FSR",
    "isr": "ISR",
    "l1prefiring": "L1 prefiring",
    "muid": "Muon ID",
    "muiso": "Muon isolation",
    "mureco": "Muon reconstruction",
    "mutrig": "Muon trigger",
    "pdf": "PDF",
    "pu": "Pileup",
    "q2": r"$Q^2$ scale",
}
PETROFF_6_COLORS = (
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
)
PETROFF_10_COLORS = (
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
)


hep.style.use("CMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot nominal, Up, and Down DY predictions for systematics "
            "for one or more one-dimensional validation variables."
        )
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        help=(
            "Validation histogram names to plot, for example "
            "'mass_jet0 pt_jet0'."
        ),
    )
    parser.add_argument(
        "--systematic",
        help=(
            "One systematic base name without Up/Down. Kept as a convenient "
            "alias for a single entry in --systematics."
        ),
    )
    parser.add_argument(
        "--systematics",
        nargs="+",
        help=(
            "Systematic base names without Up/Down, for example "
            "'JMS JMR JER'."
        ),
    )
    parser.add_argument(
        "--list-systematics",
        action="store_true",
        help="Print systematic base names available in all eligible variables.",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="Print eligible one-dimensional validation variables and exit.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "inputs" / "validation",
        help="Directory containing validation_pythia_<era>.pkl inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "validation" / "systematic_variations",
        help="Directory for output PDFs.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=138.0,
        help="Integrated luminosity displayed on the plots in fb^-1.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear y-axis instead of the default logarithmic scale.",
    )
    args = parser.parse_args()

    listing = args.list_systematics or args.list_variables
    if not listing and not args.variables:
        parser.error("--variables is required unless a --list option is used")
    if args.systematic and args.systematics:
        parser.error("use either --systematic or --systematics, not both")
    if not listing and not (args.systematic or args.systematics):
        parser.error(
            "--systematic or --systematics is required unless a --list option is used"
        )

    selected_systematics = args.systematics or (
        [args.systematic] if args.systematic else []
    )
    invalid_names = [
        name
        for name in selected_systematics
        if name.endswith("Up") or name.endswith("Down")
    ]
    if invalid_names:
        parser.error(
            "systematic names must not include Up/Down suffixes: "
            + ", ".join(invalid_names)
        )
    args.systematics = list(dict.fromkeys(selected_systematics))

    return args


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def load_inputs(input_dir: Path):
    input_paths = [
        input_dir / f"validation_pythia_{era}.pkl"
        for era in ERAS
    ]
    missing_inputs = [path for path in input_paths if not path.is_file()]
    if missing_inputs:
        missing = "\n".join(f"  {path}" for path in missing_inputs)
        raise FileNotFoundError(f"Missing required input files:\n{missing}")

    return {
        era: load_pickle(input_dir / f"validation_pythia_{era}.pkl")
        for era in ERAS
    }


def eligible_variables(pythia_inputs) -> list[str]:
    """Return supported histograms with dataset and systematic axes."""
    common_names = set.intersection(
        *(set(pythia_inputs[era]) for era in ERAS)
    )
    eligible = []
    for name in sorted(common_names):
        is_eligible = True
        for era in ERAS:
            histogram = pythia_inputs[era][name]
            axis_names = [axis.name for axis in getattr(histogram, "axes", ())]
            physics_axes = [
                axis_name
                for axis_name in axis_names
                if axis_name not in {"dataset", "systematic"}
            ]
            config = VARIABLE_CONFIGS.get(name, {})
            configured_plot_axis = config.get("plot_axis")
            configured_indices = config.get("axis_indices", {})
            supported_physics_axes = (
                len(physics_axes) == 1
                or (
                    configured_plot_axis in physics_axes
                    and set(physics_axes)
                    == {configured_plot_axis, *configured_indices}
                )
            )
            if (
                "dataset" not in axis_names
                or "systematic" not in axis_names
                or not supported_physics_axes
            ):
                is_eligible = False
                break
        if is_eligible:
            eligible.append(name)
    return eligible


def paired_systematic_bases(pythia_inputs, variable_names) -> list[str]:
    """Return Up/Down pairs shared by the selected variables and all eras."""
    common_labels = None

    for era in ERAS:
        for variable_name in variable_names:
            histogram = pythia_inputs[era][variable_name]
            labels = set(histogram.axes["systematic"])
            common_labels = labels if common_labels is None else common_labels & labels

    up_bases = {
        label.removesuffix("Up")
        for label in common_labels
        if label.endswith("Up")
    }
    down_bases = {
        label.removesuffix("Down")
        for label in common_labels
        if label.endswith("Down")
    }
    return sorted(up_bases & down_bases)


def histogram_values(histogram, dataset_name, systematic, variable_name):
    """Extract one dataset/systematic and reduce to the configured plot axis."""
    physics_axes = [
        axis
        for axis in histogram.axes
        if axis.name not in {"dataset", "systematic"}
    ]
    config = VARIABLE_CONFIGS.get(variable_name, {})
    plot_axis_name = config.get(
        "plot_axis",
        physics_axes[0].name if len(physics_axes) == 1 else None,
    )
    axis_indices = config.get("axis_indices", {})
    if plot_axis_name is None:
        raise ValueError(
            f"No plot-axis configuration for multi-dimensional variable "
            f"'{variable_name}' with axes {[axis.name for axis in physics_axes]}"
        )
    physics_axis = histogram.axes[plot_axis_name]
    dataset_index = histogram.axes["dataset"].index(dataset_name)
    systematic_index = histogram.axes["systematic"].index(systematic)
    selectors = []
    for axis in histogram.axes:
        if axis.name == "dataset":
            selectors.append(dataset_index)
        elif axis.name == "systematic":
            selectors.append(systematic_index)
        elif axis.name in axis_indices:
            selectors.append(axis_indices[axis.name])
        elif axis.name == plot_axis_name:
            selectors.append(slice(None))
        else:
            raise ValueError(
                f"No index configured for axis '{axis.name}' in '{variable_name}'"
            )

    values = histogram.values()[tuple(selectors)]
    return (
        np.asarray(values, dtype=float),
        np.asarray(physics_axis.edges, dtype=float),
        physics_axis.name,
    )


def combine_eras(pythia_inputs, variable_name, systematic_labels):
    combined = {
        systematic: None
        for systematic in systematic_labels
    }
    combined_edges = None
    combined_axis_name = None

    for era in ERAS:
        histogram = pythia_inputs[era][variable_name]
        dataset_name = f"pythia_{ERA_DATASET_TAGS[era]}"

        for systematic in systematic_labels:
            values, edges, axis_name = histogram_values(
                histogram,
                dataset_name,
                systematic,
                variable_name,
            )
            if combined[systematic] is None:
                combined[systematic] = np.zeros_like(values)
            combined[systematic] += values

            if combined_edges is None:
                combined_edges = edges
                combined_axis_name = axis_name
            elif not np.array_equal(combined_edges, edges):
                raise ValueError(
                    f"Inconsistent binning in {variable_name} for era {era}"
                )
            elif combined_axis_name != axis_name:
                raise ValueError(
                    f"Inconsistent axis name in {variable_name} for era {era}"
                )

    return combined, combined_edges, combined_axis_name


def ratio_to_nominal(varied, nominal):
    return np.divide(
        varied,
        nominal,
        out=np.full_like(varied, np.nan, dtype=float),
        where=nominal != 0,
    )


def systematic_display_name(systematic_base):
    return SYSTEMATIC_DISPLAY_NAMES.get(systematic_base, systematic_base)


def systematic_colors(number_of_systematics):
    """Use the accessible color cycles recommended in arXiv:2107.02270."""
    if number_of_systematics > len(PETROFF_10_COLORS):
        raise ValueError(
            "At most 10 systematics can be shown without repeating colors "
            "from the Petroff accessible palette."
        )
    palette = (
        PETROFF_6_COLORS
        if number_of_systematics <= len(PETROFF_6_COLORS)
        else PETROFF_10_COLORS
    )
    return list(palette[:number_of_systematics])


def make_plot(
    variations,
    edges,
    systematic_bases,
    xlabel,
    xlim,
    selection_label,
    use_log_scale,
    lumi,
    output_path,
):
    nominal_label = "nominal"

    fig, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )

    axis.stairs(
        variations[nominal_label],
        edges=edges,
        label="Nominal",
        color="black",
        linestyle="-",
        linewidth=2,
    )

    colors = systematic_colors(len(systematic_bases))
    variation_styles = []
    for index, systematic_base in enumerate(systematic_bases):
        color = colors[index]
        display_name = systematic_display_name(systematic_base)
        variation_styles.extend(
            (
                (
                    f"{systematic_base}Up",
                    f"{display_name} Up",
                    color,
                    "--",
                ),
                (
                    f"{systematic_base}Down",
                    f"{display_name} Down",
                    color,
                    ":",
                ),
            )
        )

    for variation, label, color, linestyle in variation_styles:
        axis.stairs(
            variations[variation],
            edges=edges,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )

    positive_values = np.concatenate(
        [values[values > 0] for values in variations.values()]
    )
    if positive_values.size:
        if use_log_scale:
            axis.set_yscale("log")
            axis.set_ylim(
                max(0.1, float(positive_values.min()) * 0.5),
                float(positive_values.max()) * 10.0,
            )
        else:
            axis.set_ylim(0.0, float(positive_values.max()) * 1.35)
    axis.set_ylabel("#Events")
    legend_title = "DYJets, combined Run 2"
    if selection_label:
        legend_title += f"\n{selection_label}"
    axis.legend(
        title=legend_title,
        title_fontsize=14,
        fontsize=14,
        ncol=2 if len(systematic_bases) > 1 else 1,
    )

    displayed_lumi = int(lumi) if float(lumi).is_integer() else lumi
    hep.cms.label("Internal", data=False, lumi=displayed_lumi, com=13, ax=axis)

    ratio_axis.stairs(
        np.ones_like(variations[nominal_label]),
        edges=edges,
        color="black",
        linewidth=1.5,
    )
    varied_ratios = []
    for variation, _, color, linestyle in variation_styles:
        ratio = ratio_to_nominal(
            variations[variation],
            variations[nominal_label],
        )
        varied_ratios.append(ratio)
        ratio_axis.stairs(
            ratio,
            edges=edges,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )

    ratio_axis.axhline(1.0, color="black", linewidth=1, alpha=0.6)
    finite_ratios = np.concatenate(
        [ratio[np.isfinite(ratio)] for ratio in varied_ratios]
    )
    if finite_ratios.size:
        ratio_min = min(0.8, float(finite_ratios.min()) - 0.05)
        ratio_max = max(1.2, float(finite_ratios.max()) + 0.05)
        ratio_axis.set_ylim(ratio_min, ratio_max)
    else:
        ratio_axis.set_ylim(0.8, 1.2)
    ratio_axis.set_ylabel("Var./Nom.")
    ratio_axis.set_xlabel(xlabel)
    ratio_axis.set_xlim(*(xlim if xlim is not None else (edges[0], edges[-1])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pythia_inputs = load_inputs(args.input_dir)
    available_variables = eligible_variables(pythia_inputs)

    if args.list_variables:
        print("\n".join(available_variables))
        return

    selected_variables = args.variables or available_variables
    unknown_variables = sorted(set(selected_variables) - set(available_variables))
    if unknown_variables:
        unknown = ", ".join(unknown_variables)
        available = "\n".join(f"  {name}" for name in available_variables)
        raise ValueError(
            f"Unknown or non-1D validation variable(s): {unknown}\n"
            f"Available variables:\n{available}"
        )

    available_systematics = paired_systematic_bases(
        pythia_inputs,
        selected_variables,
    )

    if args.list_systematics:
        print("\n".join(available_systematics))
        return

    unavailable_systematics = [
        name
        for name in args.systematics
        if name not in available_systematics
    ]
    if unavailable_systematics:
        available = "\n".join(f"  {name}" for name in available_systematics)
        raise ValueError(
            "The following systematics do not have complete Up/Down pairs in "
            f"all selected variables and eras: {', '.join(unavailable_systematics)}\n"
            f"Available pairs:\n{available}"
        )

    systematic_labels = ["nominal"]
    for systematic_base in args.systematics:
        systematic_labels.extend(
            (f"{systematic_base}Up", f"{systematic_base}Down")
        )

    output_systematics = "_".join(args.systematics)
    scale_suffix = "_linear" if args.linear else ""
    for variable_name in selected_variables:
        variations, edges, axis_name = combine_eras(
            pythia_inputs,
            variable_name,
            systematic_labels,
        )
        config = VARIABLE_CONFIGS.get(
            variable_name,
            {
                "xlabel": axis_name,
                "xlim": None,
                "selection_label": None,
            },
        )
        output_path = args.output_dir / (
            f"{variable_name}_{output_systematics}_run2{scale_suffix}.pdf"
        )
        make_plot(
            variations,
            edges,
            args.systematics,
            config["xlabel"],
            config["xlim"],
            config.get("selection_label"),
            not args.linear,
            args.lumi,
            output_path,
        )
        print(output_path)


if __name__ == "__main__":
    main()
