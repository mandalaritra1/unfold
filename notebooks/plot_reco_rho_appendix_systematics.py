#!/usr/bin/env python3
"""Per-systematic reco-level log10(rho^2) variation plots for the AN appendix.

One PDF per (grooming, systematic, ptreco bin): the nominal DY prediction with
the Up/Down pair of a single systematic overlaid, plus a Var./Nom. ratio
sub-panel. Each curve is normalized to unit integral and divided by the bin
width, so pure normalization systematics (lepton SFs) show a flat ratio and
only shape changes survive — matching the mass-binned appendix plots these
replace.

File names follow the AN appendix convention exactly:

    input_<grooming>_<systematic>_<i>.pdf   (i = 0,1,2 for the three pt bins)

so the LaTeX only needs its figure directory switched. All Up/Down bases in
the validation inputs are plotted, including the individual JES_* sources.
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
        "grooming": "ungroomed",
    },
    "ptjet_rhojet_g_reco": {
        "xlabel": r"$\log_{10}(\rho^2)$, Groomed",
        "xlim": (-3.5, 0.0),
        "grooming": "groomed",
    },
}
PLOT_AXIS = "mpt_reco"
PT_AXIS = "ptreco"

# ptreco bin index -> (file suffix, legend label). Index 0 (0-200 GeV) is
# below the selection and not plotted.
PT_BINS = {
    1: (0, r"$200 < p_T^\mathrm{reco} < 290$ GeV"),
    2: (1, r"$290 < p_T^\mathrm{reco} < 400$ GeV"),
    3: (2, r"$p_T^\mathrm{reco} > 400$ GeV"),
}

# pkl systematic base -> stem used in the AN appendix file names.
FILENAME_ALIASES = {"l1prefiring": "prefiring"}

# Petroff 6-color scheme members: variations are always blue Up / red Down.
UP_COLOR = "#5790fc"
DOWN_COLOR = "#e42536"

hep.style.use("CMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-systematic nominal/Up/Down reco log10(rho^2) "
            "distributions with a ratio sub-panel, one PDF per grooming, "
            "systematic, and ptreco bin, named for the AN appendix."
        )
    )
    parser.add_argument(
        "--systematics",
        nargs="+",
        help="Systematic base names without Up/Down (default: all pairs).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "inputs" / "zjet" / "validation",
        help="Directory containing validation_pythia_<era>.pkl inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "validation" / "reco_rho_appendix_systematics",
        help="Directory for output PDFs.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=138.0,
        help="Integrated luminosity displayed on the plots in fb^-1.",
    )
    return parser.parse_args()


def load_inputs(input_dir: Path):
    input_paths = {era: input_dir / f"validation_pythia_{era}.pkl" for era in ERAS}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(f"  {p}" for p in missing)
        )
    inputs = {}
    for era, path in input_paths.items():
        with path.open("rb") as handle:
            inputs[era] = pkl.load(handle)
    return inputs


def paired_systematic_bases(pythia_inputs, variable_name) -> list[str]:
    """Up/Down systematic bases present in the variable across all eras."""
    common_labels = None
    for era in ERAS:
        labels = set(pythia_inputs[era][variable_name].axes["systematic"])
        common_labels = labels if common_labels is None else common_labels & labels
    up_bases = {l.removesuffix("Up") for l in common_labels if l.endswith("Up")}
    down_bases = {l.removesuffix("Down") for l in common_labels if l.endswith("Down")}
    return sorted(up_bases & down_bases)


def histogram_values(histogram, dataset_name, systematic, pt_index):
    """Select one dataset/systematic/ptreco bin, reduce to the mpt_reco axis."""
    plot_axis = histogram.axes[PLOT_AXIS]
    selectors = []
    for axis in histogram.axes:
        if axis.name == "dataset":
            selectors.append(axis.index(dataset_name))
        elif axis.name == "systematic":
            selectors.append(axis.index(systematic))
        elif axis.name == PT_AXIS:
            selectors.append(pt_index)
        elif axis.name == PLOT_AXIS:
            selectors.append(slice(None))
        else:
            raise ValueError(f"Unexpected axis '{axis.name}' in {histogram}")
    values = histogram.values()[tuple(selectors)]
    return np.asarray(values, dtype=float), np.asarray(plot_axis.edges, dtype=float)


def combine_eras(pythia_inputs, variable_name, systematic_labels, pt_index):
    """Sum the four eras for each systematic label at a fixed ptreco bin."""
    combined = {label: None for label in systematic_labels}
    combined_edges = None
    for era in ERAS:
        histogram = pythia_inputs[era][variable_name]
        dataset_name = f"pythia_{ERA_DATASET_TAGS[era]}"
        for label in systematic_labels:
            values, edges = histogram_values(histogram, dataset_name, label, pt_index)
            combined[label] = values if combined[label] is None else combined[label] + values
            if combined_edges is None:
                combined_edges = edges
            elif not np.array_equal(combined_edges, edges):
                raise ValueError(f"Inconsistent binning for {variable_name} in {era}")
    return combined, combined_edges


def normalized_density(values, edges):
    """Unit-integral distribution divided by the bin width."""
    widths = np.diff(edges)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values)
    return values / total / widths


def ratio_to_nominal(varied, nominal):
    return np.divide(
        varied,
        nominal,
        out=np.full_like(varied, np.nan, dtype=float),
        where=nominal != 0,
    )


def make_plot(nominal, up, down, edges, config, base, pt_label, lumi, output_path):
    fig, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
        layout="constrained",
    )

    axis.stairs(nominal, edges=edges, color="black", linewidth=2.5, label="Nominal")
    axis.stairs(up, edges=edges, color=UP_COLOR, linestyle="--", linewidth=2,
                label=f"{base}Up")
    axis.stairs(down, edges=edges, color=DOWN_COLOR, linestyle=":", linewidth=2,
                label=f"{base}Down")

    in_window = (edges[:-1] >= config["xlim"][0]) & (edges[1:] <= config["xlim"][1])
    visible_max = max(
        (float(v[in_window].max()) for v in (nominal, up, down) if v[in_window].size),
        default=0.0,
    )
    if visible_max > 0:
        axis.set_ylim(0.0, visible_max * 1.5)
    axis.set_ylabel(r"$1/N\;dN/d\log_{10}(\rho^2)$")
    axis.legend(
        title=f"DYJets, combined Run 2\n{pt_label}",
        title_fontsize=16,
        fontsize=16,
        loc="upper right",
    )
    displayed_lumi = int(lumi) if float(lumi).is_integer() else lumi
    hep.cms.label("Preliminary", data=False, lumi=displayed_lumi, com=13, ax=axis)

    ratio_axis.axhline(1.0, color="black", linewidth=1.5, alpha=0.7)
    lo, hi = 0.9, 1.1
    for varied, color, style in ((up, UP_COLOR, "--"), (down, DOWN_COLOR, ":")):
        ratio = ratio_to_nominal(varied, nominal)
        ratio_axis.stairs(ratio, edges=edges, color=color, linestyle=style, linewidth=2)
        finite = ratio[in_window][np.isfinite(ratio[in_window])]
        if finite.size:
            lo = min(lo, float(finite.min()) - 0.02)
            hi = max(hi, float(finite.max()) + 0.02)
    ratio_axis.set_ylim(lo, hi)
    ratio_axis.set_ylabel("Var./Nom.")
    ratio_axis.set_xlabel(config["xlabel"])
    ratio_axis.set_xlim(*config["xlim"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=100)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pythia_inputs = load_inputs(args.input_dir)

    for variable_name, config in VARIABLE_CONFIGS.items():
        all_bases = paired_systematic_bases(pythia_inputs, variable_name)
        bases = args.systematics or all_bases
        unknown = sorted(set(bases) - set(all_bases))
        if unknown:
            raise ValueError(f"No Up/Down pair for: {', '.join(unknown)}")

        systematic_labels = ["nominal"]
        for base in bases:
            systematic_labels.extend((f"{base}Up", f"{base}Down"))

        for pt_index, (file_suffix, pt_label) in PT_BINS.items():
            combined, edges = combine_eras(
                pythia_inputs, variable_name, systematic_labels, pt_index
            )
            densities = {
                label: normalized_density(values, edges)
                for label, values in combined.items()
            }
            for base in bases:
                stem = FILENAME_ALIASES.get(base, base)
                output_path = (
                    args.output_dir
                    / f"input_{config['grooming']}_{stem}_{file_suffix}.pdf"
                )
                make_plot(
                    densities["nominal"],
                    densities[f"{base}Up"],
                    densities[f"{base}Down"],
                    edges,
                    config,
                    base,
                    pt_label,
                    args.lumi,
                    output_path,
                )
                print(output_path)


if __name__ == "__main__":
    main()
