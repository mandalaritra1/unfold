#!/usr/bin/env python3
"""Render the saved covariance-variant diagnostics in CMS House style."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

PETROFF_BLUE = "#5790fc"
PETROFF_ORANGE = "#f89c20"


def render(input_dir: Path, mode: str):
    path = input_dir / f"covariance_variants_{mode}.json"
    results = json.loads(path.read_text())
    names = [name for name, values in results.items() if "chi2_per_ndof" in values]
    values = [results[name]["chi2_per_ndof"] for name in names]
    label_map = {
        "pT-block stat only": "Block-diagonal stat.",
        "full stat only": "Full stat.",
        "pT-block stat + per-pT PS/HAD": "Block stat. + per-pT model",
        "pT-block stat + global PS/HAD": "Block stat. + global model",
        "full stat + per-pT PS/HAD": "Full stat. + per-pT model",
        "full stat + global PS/HAD": "Full stat. + global model",
    }
    labels = [label_map[name] for name in names]
    colors = [
        PETROFF_ORANGE if name.startswith("pT-block") else PETROFF_BLUE
        for name in names
    ]
    reference = 330.99 if mode == "groomed" else 364.22

    fig, ax = plt.subplots(layout="constrained")
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(
        reference,
        color="black",
        linestyle="--",
        linewidth=2.2,
    )
    ax.bar_label(bars, fmt="%.1f", padding=7, fontsize=20)
    ax.set_xlabel("Unfolded χ² / ndof")
    ax.set_xlim(0, max(max(values), reference) * 1.28)
    ax.margins(y=0.13)
    hep.cms.label(
        "Internal",
        data=True,
        loc=0,
        rlabel="138 fb$^{-1}$ (13 TeV)",
        ax=ax,
    )
    ax.text(
        reference + 0.012 * ax.get_xlim()[1],
        0.50,
        "Smeared reference",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="center",
        rotation=90,
        fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    ax.text(
        0.02,
        0.98,
        f"{mode.capitalize()}, shown phase space",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )

    output_stem = input_dir / f"covariance_variants_{mode}_cms"
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white", transparent=False)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=160,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    return output_stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/zjet/rho/model_covariance_diagnostics"),
    )
    args = parser.parse_args()
    for mode in ("ungroomed", "groomed"):
        print(render(args.input_dir, mode))


if __name__ == "__main__":
    main()
