#!/usr/bin/env python3
"""Plot ISR/FSR (or any weight) variations of gen-level jet observables split by
initiating-parton flavor.

The flavor-tagged validation inputs carry gen-level jet histograms with a
``parton_flavor`` (quark/gluon/other) or integer ``n`` flavor axis *and* a
``systematic`` axis. For each requested observable and systematic this script
draws, per flavor, the nominal prediction together with its Up/Down variation
and the Var./Nom. ratio, so the flavor dependence of the variation is visible.

The flavor histograms only carry theory weight variations (ISR/FSR/q2/PDF move
the gen-level prediction); detector systematics leave the gen weight nominal and
therefore overlap the nominal curve exactly.
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

# Flavor-tagged validation inputs, keyed by era. Only the eras present on disk
# are combined; missing eras are skipped with a warning.
ERAS = ("2016APV", "2016", "2017", "2018")
ERA_DATASET_TAGS = {
    "2016APV": "UL16NanoAODAPVv9",
    "2016": "UL16NanoAODv9",
    "2017": "UL17NanoAODv9",
    "2018": "UL18NanoAODv9",
}

# Observables with a flavor axis. ``flavor_axis`` names the categorical axis and
# ``flavor_labels`` maps each displayed flavor to the index/category used to
# select it from that axis.
FLAVOR_VARIABLE_CONFIGS = {
    "mass_flavor_jet0_gen": {
        "xlabel": r"Leading gen jet mass [GeV]",
        "xlim": (0.0, 500.0),
        "plot_axis": "mass",
        "flavor_axis": "parton_flavor",
        "flavor_labels": {"quark": "quark", "gluon": "gluon", "other": "other"},
    },
    "mpt_flavor_jet0_gen": {
        "xlabel": r"$\log_{10}(\rho^2)$ (gen)",
        "xlim": None,
        "plot_axis": "mpt_gen",
        "flavor_axis": "parton_flavor",
        "flavor_labels": {"quark": "quark", "gluon": "gluon", "other": "other"},
    },
    "pt_flavor_jet0_gen": {
        "xlabel": r"Leading gen jet $p_T$ [GeV]",
        "xlim": (200.0, 1000.0),
        "plot_axis": "pt",
        "flavor_axis": "n",
        # Integer flavor encoding from the processor: quark=1, gluon=2, other=0.
        "flavor_labels": {"quark": 1, "gluon": 2, "other": 0},
    },
    "y_flavor_jet0_gen": {
        "xlabel": r"Leading gen jet $y$",
        "xlim": (-2.5, 2.5),
        "plot_axis": "y",
        "flavor_axis": "n",
        "flavor_labels": {"quark": 1, "gluon": 2, "other": 0},
    },
}
SYSTEMATIC_DISPLAY_NAMES = {
    "isr": "ISR",
    "fsr": "FSR",
    "q2": r"$Q^2$ scale",
    "pdf": "PDF",
}
# One color per flavor (Petroff accessible palette, arXiv:2107.02270).
FLAVOR_COLORS = {
    "quark": "#5790fc",
    "gluon": "#e42536",
    "other": "#9c9ca1",
}


hep.style.use("CMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot nominal/Up/Down gen-level predictions split by parton flavor "
            "for one or more systematics."
        )
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["mass_flavor_jet0_gen", "mpt_flavor_jet0_gen"],
        help="Flavor histogram names to plot.",
    )
    parser.add_argument(
        "--systematics",
        nargs="+",
        default=["isr", "fsr"],
        help="Systematic base names without Up/Down, for example 'isr fsr'.",
    )
    parser.add_argument(
        "--flavors",
        nargs="+",
        default=["quark", "gluon"],
        help="Flavors to overlay in each plot, for example 'quark gluon'.",
    )
    parser.add_argument(
        "--ratio-ylim",
        nargs=2,
        type=float,
        default=(0.5, 1.5),
        metavar=("LOW", "HIGH"),
        help="Fixed y-range for the Var./Nom. ratio panel.",
    )
    parser.add_argument(
        "--list-systematics",
        action="store_true",
        help="Print Up/Down systematic base names available in the inputs and exit.",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="Print flavor histograms available in the inputs and exit.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "inputs" / "zjet" / "validation",
        help="Directory containing validation_pythia_<era>_flavortagged.pkl inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "validation" / "flavor_systematic_variations",
        help="Directory for output PDFs.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=138.0,
        help="Integrated luminosity displayed on the plots in fb^-1.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a logarithmic y-axis instead of the default linear scale.",
    )
    args = parser.parse_args()

    invalid_names = [
        name
        for name in args.systematics
        if name.endswith("Up") or name.endswith("Down")
    ]
    if invalid_names:
        parser.error(
            "systematic names must not include Up/Down suffixes: "
            + ", ".join(invalid_names)
        )
    args.systematics = list(dict.fromkeys(args.systematics))
    return args


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def load_inputs(input_dir: Path):
    """Load every flavor-tagged era found on disk; require at least one."""
    inputs = {}
    for era in ERAS:
        path = input_dir / f"validation_pythia_{era}_flavortagged.pkl"
        if path.is_file():
            inputs[era] = load_pickle(path)
        else:
            print(f"[skip] missing flavor-tagged input for {era}: {path}")
    if not inputs:
        raise FileNotFoundError(
            f"No validation_pythia_<era>_flavortagged.pkl files found in {input_dir}"
        )
    return inputs


def flavor_variables(inputs) -> list[str]:
    """Flavor histograms present in all loaded eras with dataset+systematic axes."""
    common_names = set.intersection(*(set(value) for value in inputs.values()))
    available = []
    for name, config in FLAVOR_VARIABLE_CONFIGS.items():
        if name not in common_names:
            continue
        ok = True
        for histogram in (inputs[era][name] for era in inputs):
            axis_names = {axis.name for axis in histogram.axes}
            if not {"dataset", "systematic", config["flavor_axis"]} <= axis_names:
                ok = False
                break
        if ok:
            available.append(name)
    return available


def paired_systematic_bases(inputs, variable_names) -> list[str]:
    """Up/Down systematic bases shared by every selected variable and era."""
    common_labels = None
    for histogram in (
        inputs[era][name] for era in inputs for name in variable_names
    ):
        labels = set(histogram.axes["systematic"])
        common_labels = labels if common_labels is None else common_labels & labels

    up_bases = {l.removesuffix("Up") for l in common_labels if l.endswith("Up")}
    down_bases = {
        l.removesuffix("Down") for l in common_labels if l.endswith("Down")
    }
    return sorted(up_bases & down_bases)


def flavor_values(histogram, dataset_name, systematic, flavor_selector, config):
    """Reduce one dataset/systematic/flavor selection to the 1D plot axis."""
    plot_axis = histogram.axes[config["plot_axis"]]
    selectors = []
    for axis in histogram.axes:
        if axis.name == "dataset":
            selectors.append(axis.index(dataset_name))
        elif axis.name == "systematic":
            selectors.append(axis.index(systematic))
        elif axis.name == config["flavor_axis"]:
            selectors.append(axis.index(flavor_selector))
        elif axis.name == config["plot_axis"]:
            selectors.append(slice(None))
        else:
            raise ValueError(
                f"Unexpected axis '{axis.name}' in flavor histogram"
            )
    values = histogram.values()[tuple(selectors)]
    return (
        np.asarray(values, dtype=float),
        np.asarray(plot_axis.edges, dtype=float),
    )


def combine_eras(inputs, variable_name, systematic_labels, flavor_name, config):
    """Sum the selected flavor over all loaded eras for each systematic label."""
    combined = {label: None for label in systematic_labels}
    combined_edges = None
    flavor_selector = config["flavor_labels"][flavor_name]

    for era, payload in inputs.items():
        histogram = payload[variable_name]
        dataset_name = f"pythia_{ERA_DATASET_TAGS[era]}"
        for label in systematic_labels:
            values, edges = flavor_values(
                histogram, dataset_name, label, flavor_selector, config
            )
            combined[label] = (
                values if combined[label] is None else combined[label] + values
            )
            if combined_edges is None:
                combined_edges = edges
            elif not np.array_equal(combined_edges, edges):
                raise ValueError(
                    f"Inconsistent binning for {variable_name} in era {era}"
                )
    return combined, combined_edges


def ratio_to_nominal(varied, nominal):
    return np.divide(
        varied,
        nominal,
        out=np.full_like(varied, np.nan, dtype=float),
        where=nominal != 0,
    )


def systematic_display_name(systematic_base):
    return SYSTEMATIC_DISPLAY_NAMES.get(systematic_base, systematic_base)


def make_plot(
    per_flavor,
    edges,
    flavor_names,
    systematic_base,
    variable_name,
    config,
    use_log_scale,
    ratio_ylim,
    lumi,
    output_path,
):
    """One figure per (variable, systematic): all flavors overlaid in one panel.

    Flavor is encoded by color, the nominal/Up/Down variation by line style, so
    the per-flavor response to the systematic can be compared directly.
    """
    display_name = systematic_display_name(systematic_base)

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )

    bottom.axhline(1.0, color="black", linewidth=1, alpha=0.6)
    positive = []
    for flavor_name in flavor_names:
        variations = per_flavor[flavor_name]
        color = FLAVOR_COLORS.get(flavor_name, "black")
        nominal = variations["nominal"]
        up = variations[f"{systematic_base}Up"]
        down = variations[f"{systematic_base}Down"]

        top.stairs(nominal, edges=edges, color=color, linewidth=2, label=flavor_name)
        top.stairs(up, edges=edges, color=color, linestyle="--", linewidth=2)
        top.stairs(down, edges=edges, color=color, linestyle=":", linewidth=2)
        positive.append(
            np.concatenate(
                [v[v > 0] for v in (nominal, up, down) if np.any(v > 0)]
                or [np.array([])]
            )
        )

        bottom.stairs(
            ratio_to_nominal(up, nominal),
            edges=edges,
            color=color,
            linestyle="--",
            linewidth=2,
        )
        bottom.stairs(
            ratio_to_nominal(down, nominal),
            edges=edges,
            color=color,
            linestyle=":",
            linewidth=2,
        )

    # Line-style proxies so the legend explains nominal/Up/Down independently
    # of flavor color.
    for style, style_label in (("-", "Nominal"), ("--", "Up"), (":", "Down")):
        top.plot([], [], color="black", linestyle=style, linewidth=2, label=style_label)

    positive = np.concatenate(positive) if positive else np.array([])
    if positive.size:
        if use_log_scale:
            top.set_yscale("log")
            top.set_ylim(
                max(0.1, float(positive.min()) * 0.5),
                float(positive.max()) * 10.0,
            )
        else:
            top.set_ylim(0.0, float(positive.max()) * 1.35)
    top.set_ylabel("#Events")
    top.legend(fontsize=13, ncol=2, title=display_name, title_fontsize=13)

    bottom.set_ylim(*ratio_ylim)
    bottom.set_ylabel("Var./Nom.")
    bottom.set_xlabel(config["xlabel"])
    xlim = config.get("xlim")
    bottom.set_xlim(*(xlim if xlim is not None else (edges[0], edges[-1])))

    displayed_lumi = int(lumi) if float(lumi).is_integer() else lumi
    hep.cms.label("Internal", data=False, lumi=displayed_lumi, com=13, ax=top)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = load_inputs(args.input_dir)
    available_variables = flavor_variables(inputs)

    if args.list_variables:
        print("\n".join(available_variables) or "(none)")
        return

    unknown = sorted(set(args.variables) - set(available_variables))
    if unknown:
        available = "\n".join(f"  {name}" for name in available_variables)
        raise ValueError(
            f"Unknown or non-flavor variable(s): {', '.join(unknown)}\n"
            f"Available flavor variables (with dataset+systematic axes):\n{available}"
        )

    available_systematics = paired_systematic_bases(inputs, args.variables)
    if args.list_systematics:
        print("\n".join(available_systematics) or "(none)")
        return

    unavailable = [s for s in args.systematics if s not in available_systematics]
    if unavailable:
        available = "\n".join(f"  {name}" for name in available_systematics)
        raise ValueError(
            "These systematics lack complete Up/Down pairs in the flavor "
            f"histograms: {', '.join(unavailable)}\nAvailable pairs:\n{available}"
        )

    scale_suffix = "_log" if args.log else ""
    for variable_name in args.variables:
        config = FLAVOR_VARIABLE_CONFIGS[variable_name]
        flavor_names = [f for f in args.flavors if f in config["flavor_labels"]]
        unknown_flavors = sorted(set(args.flavors) - set(config["flavor_labels"]))
        if unknown_flavors:
            raise ValueError(
                f"Unknown flavor(s) for {variable_name}: {', '.join(unknown_flavors)}. "
                f"Available: {', '.join(config['flavor_labels'])}"
            )
        for systematic_base in args.systematics:
            systematic_labels = [
                "nominal",
                f"{systematic_base}Up",
                f"{systematic_base}Down",
            ]
            per_flavor = {}
            edges = None
            for flavor_name in flavor_names:
                variations, edges = combine_eras(
                    inputs, variable_name, systematic_labels, flavor_name, config
                )
                per_flavor[flavor_name] = variations
            output_path = args.output_dir / (
                f"{variable_name}_{systematic_base}_byflavor{scale_suffix}.pdf"
            )
            make_plot(
                per_flavor,
                edges,
                flavor_names,
                systematic_base,
                variable_name,
                config,
                args.log,
                tuple(args.ratio_ylim),
                args.lumi,
                output_path,
            )
            print(output_path)


if __name__ == "__main__":
    main()
