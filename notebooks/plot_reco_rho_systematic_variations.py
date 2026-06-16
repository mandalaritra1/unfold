#!/usr/bin/env python3
"""Plot reco-level log10(rho^2) systematic variations, groomed and ungroomed.

For each grooming and each ptreco bin this writes two separate files:

  * a distribution plot (nominal + every Up/Down variation overlaid), and
  * a full-size Var./Nom. ratio plot (no thin sub-panel, so every variation is
    readable).

All systematics are shown together. The many per-source JES variations are
combined in quadrature into a single "JES" band:

    JES Up   = nominal + sqrt( sum_i (JES_iUp   - nominal)^2 )
    JES Down = nominal - sqrt( sum_i (JES_iDown - nominal)^2 )

while JER / JMS / JMR are kept as separate systematics. The four Run 2 eras are
combined.
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
        "xlim": (-4.5, 0.0),
        "grooming": "groomed",
    },
}
PLOT_AXIS = "mpt_reco"
PT_AXIS = "ptreco"

# ptreco bin index -> (low, high, label). Index 0 (0-200 GeV) is below the
# selection and not plotted.
PT_BINS = {
    1: (200, 290, r"$200 < p_T^\mathrm{reco} < 290$ GeV"),
    2: (290, 400, r"$290 < p_T^\mathrm{reco} < 400$ GeV"),
    3: (400, 13000, r"$p_T^\mathrm{reco} > 400$ GeV"),
}

SYSTEMATIC_DISPLAY_NAMES = {
    "eleid": "Electron ID",
    "elereco": "Electron reco",
    "eletrig": "Electron trigger",
    "fsr": "FSR",
    "isr": "ISR",
    "l1prefiring": "L1 prefiring",
    "muid": "Muon ID",
    "muiso": "Muon isolation",
    "mureco": "Muon reco",
    "mutrig": "Muon trigger",
    "pdf": "PDF",
    "pu": "Pileup",
    "q2": r"$Q^2$ scale",
    "JER": "JER",
    "JMS": "JMS",
    "JMR": "JMR",
}

# Systematics combined in quadrature into a single labelled band. JES_* sources
# are grouped by prefix; the rest are grouped by the explicit member lists below.
JES_PREFIX = "JES"
JES_LABEL = "JES (quad.)"
SYSTEMATIC_GROUPS = {
    "Lepton eff.": ["eleid", "elereco", "eletrig", "muid", "muiso", "mureco", "mutrig"],
    "Other": ["l1prefiring", "pdf", "pu", "q2"],
}
# Legend / draw order for the kept-individual systematics and the grouped bands.
DISPLAY_ORDER = [
    "JER", "JMR", "JMS", "FSR", "ISR", JES_LABEL, "Lepton eff.", "Other",
]


hep.style.use("CMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot reco-level log10(rho^2) systematic variations (groomed and "
            "ungroomed), one distribution and one full-size ratio file per "
            "ptreco bin, with JES sources combined in quadrature."
        )
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(VARIABLE_CONFIGS),
        choices=list(VARIABLE_CONFIGS),
        help="Reco rho histograms to plot (default: groomed and ungroomed).",
    )
    parser.add_argument(
        "--pt-bins",
        nargs="+",
        type=int,
        default=list(PT_BINS),
        choices=list(PT_BINS),
        help="ptreco bin indices to plot (1:200-290, 2:290-400, 3:>400).",
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
        default=ROOT / "outputs" / "validation" / "reco_rho_systematic_variations",
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
        help="Logarithmic y-axis for the distribution plot (default: linear).",
    )
    parser.add_argument(
        "--ratio-range",
        type=float,
        default=0.3,
        help="Half-width of the symmetric ratio y-range (0.3 -> 0.7 to 1.3).",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def load_inputs(input_dir: Path):
    input_paths = {era: input_dir / f"validation_pythia_{era}.pkl" for era in ERAS}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(f"  {p}" for p in missing)
        )
    return {era: load_pickle(path) for era, path in input_paths.items()}


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


def quad_combine(combined, bases, nominal):
    """Quadrature-sum the Up and Down deviations of several systematics.

    Up   = nominal + sqrt( sum_i (base_iUp   - nominal)^2 )
    Down = nominal - sqrt( sum_i (base_iDown - nominal)^2 )
    """
    up_dev = np.sqrt(
        np.sum([(combined[f"{b}Up"] - nominal) ** 2 for b in bases], axis=0)
    )
    down_dev = np.sqrt(
        np.sum([(combined[f"{b}Down"] - nominal) ** 2 for b in bases], axis=0)
    )
    return nominal + up_dev, nominal - down_dev


def build_variations(combined, all_bases):
    """Return {label: (up, down)} with the configured groups summed in quadrature.

    JES_* sources and the named SYSTEMATIC_GROUPS are each collapsed into one
    band; every other systematic is kept individually.
    """
    nominal = combined["nominal"]
    variations = {}
    assigned = set()

    jes_bases = [b for b in all_bases if b.startswith(JES_PREFIX)]
    if jes_bases:
        variations[JES_LABEL] = quad_combine(combined, jes_bases, nominal)
        assigned.update(jes_bases)

    for label, members in SYSTEMATIC_GROUPS.items():
        present = [b for b in members if b in all_bases]
        if present:
            variations[label] = quad_combine(combined, present, nominal)
            assigned.update(present)

    for base in all_bases:
        if base in assigned:
            continue
        variations[display_name(base)] = (combined[f"{base}Up"], combined[f"{base}Down"])

    ordered = [label for label in DISPLAY_ORDER if label in variations]
    ordered += [label for label in variations if label not in ordered]
    return nominal, variations, ordered


def ratio_to_nominal(varied, nominal):
    return np.divide(
        varied,
        nominal,
        out=np.full_like(varied, np.nan, dtype=float),
        where=nominal != 0,
    )


def display_name(base):
    return SYSTEMATIC_DISPLAY_NAMES.get(base, base)


def systematic_colors(n):
    """n distinct, saturated colors (legible as both lines and text labels).

    tab10 + Dark2 gives 18 well-separated colors with no pale tints, so the
    overflow annotations stay readable in the systematic's own color.
    """
    palette = [
        *matplotlib.colormaps["tab10"].colors,
        *matplotlib.colormaps["Dark2"].colors,
    ]
    if n > len(palette):
        raise ValueError(f"Need {n} colors but only {len(palette)} are defined.")
    return palette[:n]


def save_figure(fig, output_path):
    """Write the PDF and a PNG preview (for slides) side by side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)


def cms_label(ax, lumi):
    displayed_lumi = int(lumi) if float(lumi).is_integer() else lumi
    hep.cms.label("Internal", data=False, lumi=displayed_lumi, com=13, ax=ax)


def legend_title(config, pt_label):
    return f"DYJets, combined Run 2\n{config['xlabel']}\n{pt_label}"


def add_style_proxies(ax):
    """Line-style key for the variation direction (color encodes the source)."""
    for style, label in (("--", "Up"), (":", "Down")):
        ax.plot([], [], color="black", linestyle=style, linewidth=2, label=label)


def place_legend(ax, config, pt_label):
    ax.legend(
        title=legend_title(config, pt_label),
        title_fontsize=14,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
    )


def make_distribution_plot(
    nominal, variations, ordered, edges, config, pt_label, colors,
    use_log_scale, lumi, output_path,
):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.stairs(nominal, edges=edges, color="black", linewidth=2.5, label="Nominal")
    for base, color in zip(ordered, colors):
        up, down = variations[base]
        ax.stairs(up, edges=edges, color=color, linestyle="--", linewidth=1.5,
                  label=display_name(base))
        ax.stairs(down, edges=edges, color=color, linestyle=":", linewidth=1.5)
    add_style_proxies(ax)

    positive = nominal[nominal > 0]
    if positive.size:
        if use_log_scale:
            ax.set_yscale("log")
            ax.set_ylim(max(0.1, float(positive.min()) * 0.5),
                        float(positive.max()) * 10.0)
        else:
            ax.set_ylim(0.0, float(positive.max()) * 1.35)
    ax.set_ylabel("#Events")
    ax.set_xlabel(config["xlabel"])
    ax.set_xlim(*config["xlim"])
    cms_label(ax, lumi)
    place_legend(ax, config, pt_label)

    save_figure(fig, output_path)
    plt.close(fig)


def annotate_overflow(ax, overflow, lo, hi):
    """Write the value of each ratio bin that runs off the fixed y-range, at the
    bin's x-position and in the systematic's color. Vertical text and a per-slot
    nudge keep collisions readable."""
    slots: dict = {}
    for x, value, color, position in sorted(overflow, key=lambda t: (t[3], t[0])):
        key = (position, round(float(x) * 4) / 4)
        k = slots.get(key, 0)
        slots[key] = k + 1
        if position == "top":
            y, va, base_off = hi, "top", -2
            y_off = base_off - k * 11
        else:
            y, va, base_off = lo, "bottom", 2
            y_off = base_off + k * 11
        ax.annotate(
            f"{value:.2f}",
            xy=(x, y),
            xytext=(0, y_off),
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va=va,
            rotation=90,
            clip_on=False,
        )


def make_ratio_plot(
    nominal, variations, ordered, edges, config, pt_label, colors,
    ratio_range, lumi, output_path,
):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(1.0, color="black", linewidth=1.5, alpha=0.7)
    lo, hi = 1.0 - ratio_range, 1.0 + ratio_range

    # Find ratio bins inside the visible x-window that overflow the fixed range
    # so they can be annotated rather than silently clipped.
    centers = 0.5 * (edges[:-1] + edges[1:])
    visible = (centers >= config["xlim"][0]) & (centers <= config["xlim"][1])
    vis_centers = centers[visible]

    overflow = []
    for base, color in zip(ordered, colors):
        up, down = variations[base]
        for ratio, style in (
            (ratio_to_nominal(up, nominal), "--"),
            (ratio_to_nominal(down, nominal), ":"),
        ):
            ax.stairs(
                ratio, edges=edges, color=color, linestyle=style, linewidth=1.8,
                label=display_name(base) if style == "--" else None,
            )
            vis = ratio[visible]
            finite = np.isfinite(vis)
            above = finite & (vis > hi)
            below = finite & (vis < lo)
            if np.any(above):
                idx = int(np.argmax(np.where(above, vis, -np.inf)))
                overflow.append((vis_centers[idx], float(vis[idx]), color, "top"))
            if np.any(below):
                idx = int(np.argmin(np.where(below, vis, np.inf)))
                overflow.append((vis_centers[idx], float(vis[idx]), color, "bottom"))
    add_style_proxies(ax)

    ax.set_ylim(lo, hi)
    annotate_overflow(ax, overflow, lo, hi)
    ax.set_ylabel("Var. / Nom.")
    ax.set_xlabel(config["xlabel"])
    ax.set_xlim(*config["xlim"])
    cms_label(ax, lumi)
    place_legend(ax, config, pt_label)

    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pythia_inputs = load_inputs(args.input_dir)

    for variable_name in args.variables:
        config = VARIABLE_CONFIGS[variable_name]
        all_bases = paired_systematic_bases(pythia_inputs, variable_name)
        systematic_labels = ["nominal"]
        for base in all_bases:
            systematic_labels.extend((f"{base}Up", f"{base}Down"))

        for pt_index in args.pt_bins:
            low, high, pt_label = PT_BINS[pt_index]
            combined, edges = combine_eras(
                pythia_inputs, variable_name, systematic_labels, pt_index
            )
            nominal, variations, ordered = build_variations(combined, all_bases)
            colors = systematic_colors(len(ordered))

            stem = f"{variable_name}_pt{low}_{high}"
            dist_path = args.output_dir / f"{stem}_distribution_run2.pdf"
            ratio_path = args.output_dir / f"{stem}_ratio_run2.pdf"

            make_distribution_plot(
                nominal, variations, ordered, edges, config, pt_label, colors,
                args.log, args.lumi, dist_path,
            )
            make_ratio_plot(
                nominal, variations, ordered, edges, config, pt_label, colors,
                args.ratio_range, args.lumi, ratio_path,
            )
            print(dist_path)
            print(ratio_path)


if __name__ == "__main__":
    main()
