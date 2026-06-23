#!/usr/bin/env python3
"""Run the Z+jet rho data-prior response test.

Prior-dependence check using already-produced weighted PYTHIA histograms:
unfold the same data once with the nominal PYTHIA rho response and once with a
PYTHIA response whose gen-level rho prior has been reweighted toward the
unfolded data. The weighted input is a full producer pickle, not an
event-level correction.

The weighted MC pickle is a one-off research input and is not committed; pass
it with --weighted-mc. The small per-mode npz artifacts written under
outputs/zjet/rho/<tag>_data_prior_test/artifacts/ let
scripts/plot_data_prior_unfolded_comparison.py redraw the comparison figures
later without ROOT or the heavy pickle.

    source scripts/setup_root.sh
    python scripts/study_data_prior.py --weighted-mc /path/to/weighted_pythia.pkl
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle as pkl
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import mplhep as hep
import numpy as np

from unfold.tools import binning


RHO_KEYS = {
    "ungroomed": {
        "response": "response_matrix_rho_u",
        "reco": "ptjet_rhojet_u_reco",
        "gen": "ptjet_rhojet_u_gen",
    },
    "groomed": {
        "response": "response_matrix_rho_g",
        "reco": "ptjet_rhojet_g_reco",
        "gen": "ptjet_rhojet_g_gen",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weighted-mc",
        type=Path,
        required=True,
        help=(
            "Producer pickle with the data-prior-weighted PYTHIA rho histograms "
            "(gen-level rho prior reweighted toward the unfolded data). Not "
            "committed; see outputs/zjet/rho/<tag>_data_prior_test/README.md."
        ),
    )
    parser.add_argument(
        "--tag",
        default="original",
        help="Z+jet rho input tag used for the nominal MC and data inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/zjet/rho/<tag>_data_prior_test.",
    )
    parser.add_argument("--cms-label", default="Internal")
    parser.add_argument("--lumi", type=float, default=138.0)
    parser.add_argument("--com", type=float, default=13.0)
    parser.add_argument(
        "--jacobian",
        action="store_true",
        help="Use Jacobian propagation for normalized statistical errors.",
    )
    parser.add_argument(
        "--regularization",
        choices=("none", "ratio_curvature"),
        default="none",
        help="Optional ratio-curvature regularization for both unfolds.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Fixed regularization strength; requires --regularization ratio_curvature.",
    )
    args = parser.parse_args()
    if args.tau is not None and args.regularization == "none":
        parser.error("--tau requires --regularization ratio_curvature")
    return args


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pkl.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def validate_payload(payload, path: Path) -> None:
    required = {
        key
        for mode_keys in RHO_KEYS.values()
        for key in mode_keys.values()
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"{path} is missing required rho histograms: {missing}")
    for mode, keys in RHO_KEYS.items():
        response = payload[keys["response"]]
        if "systematic" not in response.axes.name:
            raise ValueError(f"{path} {mode} response has no systematic axis")
        if "nominal" not in list(response.axes["systematic"]):
            raise ValueError(f"{path} {mode} response has no nominal systematic")


def safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return np.divide(
        numer,
        denom,
        out=np.full_like(numer, np.nan, dtype=float),
        where=np.asarray(denom, dtype=float) != 0,
    )


def summarize_shift(nominal: np.ndarray, weighted: np.ndarray) -> dict[str, float]:
    ratio = safe_ratio(weighted, nominal)
    delta = ratio - 1.0
    finite = np.isfinite(delta)
    if not np.any(finite):
        return {"mean_abs": float("nan"), "max_abs": float("nan")}
    return {
        "mean_abs": float(np.mean(np.abs(delta[finite]))),
        "max_abs": float(np.max(np.abs(delta[finite]))),
    }


def summarize_visible_normalized_shift_by_pt(nominal: Unfolder, weighted: Unfolder):
    nominal_norm, _ = normalized_arrays(nominal)
    weighted_norm, _ = normalized_arrays(weighted)
    summaries = []
    for i, (nominal_values, weighted_values) in enumerate(zip(nominal_norm, weighted_norm)):
        edges = np.asarray(nominal.gen_edges_by_pt[i], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xlo, xhi = nominal._observable_xlim(i)
        visible = (centers >= xlo) & (centers <= xhi)
        ratio = safe_ratio(weighted_values, nominal_values)
        finite = visible & np.isfinite(ratio)
        delta = np.abs(ratio[finite] - 1.0)
        hi = nominal.pt_edges[i + 1]
        summaries.append(
            {
                "pt_bin": [
                    float(nominal.pt_edges[i]),
                    None if i + 1 == len(nominal.pt_edges) - 1 else float(hi),
                ],
                "mean_abs": float(np.mean(delta)) if delta.size else float("nan"),
                "max_abs": float(np.max(delta)) if delta.size else float("nan"),
            }
        )
    return summaries


def unfold_once(
    spec,
    *,
    groomed: bool,
    mc_inputs,
    data_inputs,
    output_dir: Path,
    label: str,
    cms_label: str,
    lumi: float,
    com: float,
) -> Unfolder:
    from unfold.tools.unfolder_core import Unfolder

    run_spec = replace(
        spec,
        output_dir=os.path.relpath(output_dir / label, REPO_ROOT) + "/",
    )
    return Unfolder.from_prepared_inputs(
        run_spec,
        groomed,
        mc_inputs=mc_inputs,
        data_inputs=data_inputs,
        analysis_binning=binning.bin_edges(groomed),
        systematics=["nominal"],
        cms_label=cms_label,
        lumi=lumi,
        com=com,
    )


def normalized_arrays(unfolder: Unfolder) -> tuple[list[np.ndarray], list[np.ndarray]]:
    values = [np.asarray(item["unfolded"], dtype=float) for item in unfolder.normalized_results]
    errors = [np.asarray(item["unfolded_err"], dtype=float) for item in unfolder.normalized_results]
    return values, errors


def normalized_field(unfolder: Unfolder, key: str, err_key: str | None = None):
    values = [np.asarray(item[key], dtype=float) for item in unfolder.normalized_results]
    errors = None
    if err_key is not None:
        errors = [np.asarray(item[err_key], dtype=float) for item in unfolder.normalized_results]
    return values, errors


def pt_title(unfolder: Unfolder, i: int) -> str:
    lo = unfolder.pt_edges[i]
    hi = unfolder.pt_edges[i + 1]
    hi_label = r"$\infty$" if i + 1 == len(unfolder.pt_edges) - 1 else f"{hi:g}"
    return rf"{lo:g} < $p_T$ < {hi_label} GeV"


def save_figure_pair(fig, output_dir: Path, stem: str) -> dict[str, Path]:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=240)
    return {"pdf": pdf_path, "png": png_path}


def add_labeled_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: int = 15,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="black",
        linespacing=1.15,
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "0.25") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def make_explainer_slide(output_dir: Path) -> dict:
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(15.5, 8.8), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.04, 0.93, "Data-prior test", fontsize=28, weight="bold", ha="left", va="top")
    ax.text(
        0.04,
        0.86,
        "What the first comparison slide means",
        fontsize=16,
        color="0.25",
        ha="left",
        va="top",
    )

    add_labeled_box(
        ax,
        (0.06, 0.58),
        0.23,
        0.14,
        "PYTHIA truth\nnominal gen rho prior",
        facecolor="#f3f4f6",
        edgecolor="black",
    )
    add_labeled_box(
        ax,
        (0.38, 0.58),
        0.23,
        0.14,
        "nominal unfolded data\nunfold data with nominal response",
        facecolor="#e8f1fb",
        edgecolor="#1f77b4",
        fontsize=14,
    )
    add_labeled_box(
        ax,
        (0.70, 0.58),
        0.23,
        0.14,
        "reweighted PYTHIA truth\nmatched to unfolded data",
        facecolor="#eaf6eb",
        edgecolor="#2ca02c",
        fontsize=14,
    )
    add_labeled_box(
        ax,
        (0.58, 0.27),
        0.34,
        0.14,
        "new gen / reco / response\nfrom weighted PYTHIA histograms",
        facecolor="#fff3e1",
        edgecolor="#d98817",
        fontsize=14,
    )

    add_arrow(ax, (0.29, 0.65), (0.38, 0.65), color="#1f77b4")
    add_arrow(ax, (0.61, 0.65), (0.70, 0.65), color="#2ca02c")
    add_arrow(ax, (0.82, 0.58), (0.75, 0.41), color="#d98817")
    add_arrow(ax, (0.75, 0.27), (0.75, 0.17), color="0.25")

    ax.text(
        0.50,
        0.10,
        "Then unfold the same data again and compare nominal vs data-prior response.",
        fontsize=16,
        ha="center",
        va="center",
    )

    key_rows = [
        ("nominal unfolded", "nominal unfolded data"),
        ("nominal truth", "PYTHIA truth"),
        ("data-prior truth", "reweighted PYTHIA truth"),
    ]
    y0 = 0.43
    ax.text(0.06, y0 + 0.08, "Legend translation", fontsize=18, weight="bold", ha="left")
    for i, (old, new) in enumerate(key_rows):
        y = y0 - 0.06 * i
        ax.text(0.07, y, old, fontsize=14, ha="left", va="center", color="0.25")
        ax.text(0.275, y, "->", fontsize=14, ha="center", va="center", color="0.45")
        ax.text(0.31, y, new, fontsize=14, ha="left", va="center", color="black")

    paths = save_figure_pair(fig, output_dir, "data_prior_explainer")
    return {"title": "Data-prior test", "mode": "overview", "figure": fig, **paths}


def make_pt_shape_figure(
    reference_unfolder: Unfolder,
    mode: str,
    title: str,
    main_series: list[dict],
    ratio_series: list[dict],
    *,
    ratio_ylabel: str,
    cms_label: str,
    lumi: float,
    com: float,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    axis: str = "gen",
):
    hep.style.use("CMS")
    n_pt = len(reference_unfolder.pt_edges) - 1
    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.12)
    panel_axes = []

    for i in range(n_pt):
        inner = outer[i // 2, i % 2].subgridspec(
            2,
            1,
            height_ratios=[3.0, 1.0],
            hspace=0.04,
        )
        main_ax = fig.add_subplot(inner[0])
        ratio_ax = fig.add_subplot(inner[1], sharex=main_ax)
        panel_axes.append((main_ax, ratio_ax))

    for i, (main_ax, ratio_ax) in enumerate(panel_axes):
        if axis == "reco":
            edges = np.asarray(reference_unfolder.reco_edges_by_pt[i], dtype=float)
            lower = (
                reference_unfolder.spec.xlim_lower_groomed
                if reference_unfolder.groomed
                else reference_unfolder.spec.xlim_lower_ungroomed
            )
            xlim = (lower, float(edges[-1]))
        else:
            edges = np.asarray(reference_unfolder.gen_edges_by_pt[i], dtype=float)
            xlim = reference_unfolder._observable_xlim(i)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xerr = 0.5 * np.diff(edges)

        for series in main_series:
            values = np.asarray(series["values"][i], dtype=float)
            errors = series.get("errors")
            yerr = None if errors is None else np.asarray(errors[i], dtype=float)
            main_ax.errorbar(
                centers,
                values,
                yerr=yerr,
                xerr=xerr,
                fmt=series.get("marker", "o"),
                ms=3,
                lw=1,
                color=series["color"],
                label=series["label"],
            )

        for series in ratio_series:
            numerator = np.asarray(series["numerator"][i], dtype=float)
            denominator = np.asarray(series["denominator"][i], dtype=float)
            ratio = safe_ratio(numerator, denominator)
            ratio_ax.errorbar(
                centers,
                ratio,
                xerr=xerr,
                fmt=series.get("marker", "o"),
                ms=3,
                lw=1,
                color=series["color"],
                label=series["label"],
            )

        ratio_ax.axhline(1.0, color="black", lw=1, ls=":")
        main_ax.set_title(pt_title(reference_unfolder, i), fontsize=14)
        main_ax.set_xlim(xlim)
        ratio_ax.set_xlim(xlim)
        ratio_ax.set_ylim(*ratio_ylim)
        main_ax.set_ylabel(reference_unfolder._normalized_ylabel(), fontsize=11)
        ratio_ax.set_ylabel(ratio_ylabel, fontsize=11)
        ratio_ax.set_xlabel(reference_unfolder._observable_short_label(), fontsize=12)
        main_ax.tick_params(axis="both", labelsize=10)
        ratio_ax.tick_params(axis="both", labelsize=10)
        main_ax.tick_params(labelbottom=False)
        if i == 0:
            main_ax.legend(fontsize=10)

    hep.cms.label(
        cms_label,
        data=True,
        lumi=reference_unfolder._as_int_when_whole(lumi),
        com=reference_unfolder._as_int_when_whole(com),
        fontsize=15,
        ax=panel_axes[0][0],
    )
    fig.suptitle(f"{title} ({mode})", fontsize=16)
    return fig


def response_probability(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    colsum = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, colsum, out=np.zeros_like(matrix), where=colsum > 0)


def make_response_figure(
    mode: str,
    nominal: Unfolder,
    weighted: Unfolder,
    *,
    cms_label: str,
    lumi: float,
    com: float,
):
    hep.style.use("CMS")
    nominal_prob = response_probability(nominal.mosaic)
    weighted_prob = response_probability(weighted.mosaic)
    ratio = safe_ratio(weighted_prob, nominal_prob)
    positive = np.concatenate(
        [nominal_prob[nominal_prob > 0], weighted_prob[weighted_prob > 0]]
    )
    vmin = max(float(np.min(positive)), 1e-5) if positive.size else 1e-5
    vmax = float(np.max(positive)) if positive.size else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    panels = [
        ("nominal", nominal_prob, LogNorm(vmin=vmin, vmax=vmax), "viridis"),
        ("data-prior", weighted_prob, LogNorm(vmin=vmin, vmax=vmax), "viridis"),
        ("data-prior / nominal", ratio, None, "coolwarm"),
    ]

    for ax, (panel_title, values, norm, cmap) in zip(axes, panels):
        kwargs = {"origin": "lower", "aspect": "auto", "cmap": cmap}
        if norm is not None:
            kwargs["norm"] = norm
            plot_values = np.ma.masked_less_equal(values, 0.0)
        else:
            kwargs["vmin"] = 0.5
            kwargs["vmax"] = 1.5
            plot_values = np.ma.masked_invalid(values)
        image = ax.imshow(plot_values, **kwargs)
        ax.set_title(panel_title, fontsize=13)
        ax.set_xlabel("gen bin index", fontsize=11)
        ax.set_ylabel("reco bin index", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    hep.cms.label(
        cms_label,
        data=True,
        lumi=nominal._as_int_when_whole(lumi),
        com=nominal._as_int_when_whole(com),
        fontsize=14,
        ax=axes[0],
    )
    fig.suptitle(f"Response Matrix ({mode})", fontsize=16)
    return fig


def make_full_comparison_plots(
    output_dir: Path,
    mode: str,
    nominal: Unfolder,
    weighted: Unfolder,
    *,
    cms_label: str,
    lumi: float,
    com: float,
):
    nominal_unfolded, nominal_unfolded_err = normalized_field(
        nominal,
        "unfolded",
        "unfolded_err",
    )
    weighted_unfolded, weighted_unfolded_err = normalized_field(
        weighted,
        "unfolded",
        "unfolded_err",
    )
    nominal_truth, _ = normalized_field(nominal, "true")
    weighted_truth, _ = normalized_field(weighted, "true")
    data_reco, _ = normalized_field(nominal, "measured")
    nominal_reco, _ = normalized_field(nominal, "reco_mc")
    weighted_reco, _ = normalized_field(weighted, "reco_mc")

    plot_specs = [
        (
            "gen_level",
            "Gen Level",
            make_pt_shape_figure(
                nominal,
                mode,
                "Gen Level",
                [
                    {"label": "PYTHIA truth", "values": nominal_truth, "color": "black", "marker": "o"},
                    {
                        "label": "reweighted PYTHIA truth",
                        "values": weighted_truth,
                        "color": "tab:green",
                        "marker": "s",
                    },
                    {
                        "label": "nominal unfolded data",
                        "values": nominal_unfolded,
                        "errors": nominal_unfolded_err,
                        "color": "tab:blue",
                        "marker": "^",
                    },
                ],
                [
                    {
                        "label": "PYTHIA truth / unfolded data",
                        "numerator": nominal_truth,
                        "denominator": nominal_unfolded,
                        "color": "black",
                        "marker": "o",
                    },
                    {
                        "label": "reweighted truth / unfolded data",
                        "numerator": weighted_truth,
                        "denominator": nominal_unfolded,
                        "color": "tab:green",
                        "marker": "s",
                    },
                ],
                ratio_ylabel="truth / unfolded",
                cms_label=cms_label,
                lumi=lumi,
                com=com,
            ),
        ),
        (
            "reco_level",
            "Reco Level",
            make_pt_shape_figure(
                nominal,
                mode,
                "Reco Level",
                [
                    {"label": "data reco", "values": data_reco, "color": "black", "marker": "o"},
                    {"label": "nominal reco", "values": nominal_reco, "color": "tab:orange", "marker": "s"},
                    {"label": "data-prior reco", "values": weighted_reco, "color": "tab:green", "marker": "^"},
                ],
                [
                    {
                        "label": "nominal reco / data",
                        "numerator": nominal_reco,
                        "denominator": data_reco,
                        "color": "tab:orange",
                        "marker": "s",
                    },
                    {
                        "label": "data-prior reco / data",
                        "numerator": weighted_reco,
                        "denominator": data_reco,
                        "color": "tab:green",
                        "marker": "^",
                    },
                ],
                ratio_ylabel="MC / data",
                cms_label=cms_label,
                lumi=lumi,
                com=com,
                axis="reco",
            ),
        ),
        (
            "response_matrix",
            "Response Matrix",
            make_response_figure(
                mode,
                nominal,
                weighted,
                cms_label=cms_label,
                lumi=lumi,
                com=com,
            ),
        ),
        (
            "unfolded",
            "Unfolded Result",
            make_pt_shape_figure(
                nominal,
                mode,
                "Unfolded Result",
                [
                    {
                        "label": "nominal response unfolded data",
                        "values": nominal_unfolded,
                        "errors": nominal_unfolded_err,
                        "color": "black",
                        "marker": "o",
                    },
                    {
                        "label": "data-prior response unfolded data",
                        "values": weighted_unfolded,
                        "errors": weighted_unfolded_err,
                        "color": "tab:green",
                        "marker": "s",
                    },
                ],
                [
                    {
                        "label": "data-prior / nominal",
                        "numerator": weighted_unfolded,
                        "denominator": nominal_unfolded,
                        "color": "tab:green",
                        "marker": "s",
                    },
                ],
                ratio_ylabel="data-prior / nominal",
                cms_label=cms_label,
                lumi=lumi,
                com=com,
            ),
        ),
    ]

    outputs = []
    for stem, slide_title, fig in plot_specs:
        paths = save_figure_pair(fig, output_dir, f"{mode}_{stem}_comparison")
        outputs.append({"title": slide_title, "mode": mode, "figure": fig, **paths})
    return outputs


def write_slide_pdf(plot_outputs: list[dict], output_dir: Path) -> Path:
    slide_path = output_dir / "data_prior_comparison_slides.pdf"
    with PdfPages(slide_path) as pdf:
        for item in plot_outputs:
            pdf.savefig(item["figure"])
    for item in plot_outputs:
        plt.close(item["figure"])
    return slide_path


def write_artifact(
    output_dir: Path,
    mode: str,
    nominal: Unfolder,
    weighted: Unfolder,
) -> Path:
    path = output_dir / "artifacts" / f"{mode}_data_prior_test.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    nom_norm, nom_norm_err = normalized_arrays(nominal)
    weighted_norm, weighted_norm_err = normalized_arrays(weighted)
    np.savez_compressed(
        path,
        pt_edges=np.asarray(nominal.pt_edges, dtype=float),
        rho_edges_gen=np.asarray(nominal.edges_gen, dtype=float),
        rho_edges_reco=np.asarray(nominal.edges, dtype=float),
        gen_edges_by_pt=np.asarray(nominal.gen_edges_by_pt, dtype=object),
        nominal_unfolded=nominal.y_unf,
        weighted_unfolded=weighted.y_unf,
        nominal_unfolded_err=nominal.ye_unf,
        weighted_unfolded_err=weighted.ye_unf,
        nominal_unfolded_2dnorm=nominal.unfolded_2dnorm_flat,
        weighted_unfolded_2dnorm=weighted.unfolded_2dnorm_flat,
        nominal_normalized=np.asarray(nom_norm, dtype=object),
        weighted_normalized=np.asarray(weighted_norm, dtype=object),
        nominal_normalized_err=np.asarray(nom_norm_err, dtype=object),
        weighted_normalized_err=np.asarray(weighted_norm_err, dtype=object),
        nominal_prior=nominal.y_true,
        weighted_prior=weighted.y_true,
        nominal_measured=nominal.y_meas,
        weighted_measured=weighted.y_meas,
        nominal_fake_fraction=nominal.fake_fraction_2d,
        weighted_fake_fraction=weighted.fake_fraction_2d,
        nominal_misses=nominal.misses_2d,
        weighted_misses=weighted.misses_2d,
        nominal_response_mosaic=nominal.mosaic,
        weighted_response_mosaic=weighted.mosaic,
    )
    return path


def plot_mode_comparison(
    output_dir: Path,
    mode: str,
    nominal: Unfolder,
    weighted: Unfolder,
    *,
    cms_label: str,
    lumi: float,
    com: float,
) -> Path:
    hep.style.use("CMS")
    nom_norm, nom_err = normalized_arrays(nominal)
    weighted_norm, weighted_err = normalized_arrays(weighted)

    def visible_ratio_for_pt(i):
        edges_i = np.asarray(nominal.gen_edges_by_pt[i], dtype=float)
        centers_i = 0.5 * (edges_i[:-1] + edges_i[1:])
        xlo, xhi = nominal._observable_xlim(i)
        visible = (centers_i >= xlo) & (centers_i <= xhi)
        ratio = safe_ratio(weighted_norm[i], nom_norm[i])
        return ratio, visible

    ratio_min, ratio_max = 0.5, 1.5
    n_pt = len(nominal.pt_edges) - 1
    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    outer = fig.add_gridspec(
        2,
        2,
        wspace=0.08,
        hspace=0.12,
    )
    panel_axes = []
    for i in range(n_pt):
        inner = outer[i // 2, i % 2].subgridspec(
            2,
            1,
            height_ratios=[3.0, 1.0],
            hspace=0.04,
        )
        dist_ax = fig.add_subplot(inner[0])
        ratio_ax = fig.add_subplot(inner[1], sharex=dist_ax)
        panel_axes.append((dist_ax, ratio_ax))

    for i, (dist_ax, ratio_ax) in enumerate(panel_axes):
        edges = np.asarray(nominal.gen_edges_by_pt[i], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xerr = 0.5 * np.diff(edges)

        dist_ax.errorbar(
            centers,
            nom_norm[i],
            yerr=nom_err[i],
            xerr=xerr,
            fmt="o",
            ms=3,
            lw=1,
            color="black",
            label="nominal response",
        )
        dist_ax.errorbar(
            centers,
            weighted_norm[i],
            yerr=weighted_err[i],
            xerr=xerr,
            fmt="s",
            ms=3,
            lw=1,
            color="tab:green",
            label="data-prior response",
        )
        ratio, visible = visible_ratio_for_pt(i)
        ratio_ax.axhline(1.0, color="black", lw=1, ls=":")
        ratio_ax.errorbar(
            centers,
            ratio,
            xerr=xerr,
            fmt="o",
            ms=3,
            lw=1,
            color="tab:green",
        )

        finite_visible = ratio[visible & np.isfinite(ratio)]
        if finite_visible.size:
            delta = np.abs(finite_visible - 1.0)
            ratio_ax.text(
                0.03,
                0.90,
                f"mean |Δ|={np.mean(delta):.1%}, max={np.max(delta):.1%}",
                transform=ratio_ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )

        lo = nominal.pt_edges[i]
        hi = nominal.pt_edges[i + 1]
        hi_label = r"$\infty$" if i + 1 == n_pt else f"{hi:g}"
        title = rf"{lo:g} < $p_T$ < {hi_label} GeV"
        dist_ax.set_title(title, fontsize=14)
        dist_ax.set_xlim(nominal._observable_xlim(i))
        ratio_ax.set_xlim(nominal._observable_xlim(i))
        ratio_ax.set_ylim(ratio_min, ratio_max)
        dist_ax.set_ylabel(nominal._normalized_ylabel(), fontsize=11)
        ratio_ax.set_ylabel("weighted / nominal", fontsize=11)
        ratio_ax.set_xlabel(nominal._observable_short_label(), fontsize=12)
        dist_ax.tick_params(axis="both", labelsize=10)
        ratio_ax.tick_params(axis="both", labelsize=10)
        dist_ax.tick_params(labelbottom=False)
        if i == 0:
            dist_ax.legend(fontsize=11)

    hep.cms.label(
        cms_label,
        data=True,
        lumi=nominal._as_int_when_whole(lumi),
        com=nominal._as_int_when_whole(com),
        fontsize=15,
        ax=panel_axes[0][0],
    )
    fig.suptitle(f"Z+jet rho data-prior response test ({mode})", fontsize=16)
    path = output_dir / f"{mode}_data_prior_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    import ROOT
    from unfold.tools.unfolder_core import get_spec

    weighted_path = args.weighted_mc.expanduser().resolve()
    if not weighted_path.is_file():
        raise FileNotFoundError(weighted_path)

    spec = get_spec("zjet", "rho", args.tag)
    spec = replace(
        spec,
        stat_propagation="jacobian" if args.jacobian else spec.stat_propagation,
        regularization=args.regularization,
        tau=args.tau,
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "outputs" / "zjet" / "rho" / f"{args.tag}_data_prior_test"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nominal_mc_path = (REPO_ROOT / spec.input_dir / spec.mc_file).resolve()
    data_path = (REPO_ROOT / spec.input_dir / spec.data_file).resolve()
    nominal_mc = load_pickle(nominal_mc_path)
    weighted_mc = load_pickle(weighted_path)
    data_inputs = load_pickle(data_path)
    validate_payload(nominal_mc, nominal_mc_path)
    validate_payload(weighted_mc, weighted_path)

    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    summaries = []
    artifacts = []
    plot_outputs = [make_explainer_slide(output_dir)]
    for mode, groomed in (("ungroomed", False), ("groomed", True)):
        print(f"[{mode}] unfolding nominal response")
        nominal = unfold_once(
            spec,
            groomed=groomed,
            mc_inputs=nominal_mc,
            data_inputs=data_inputs,
            output_dir=output_dir / mode,
            label="nominal",
            cms_label=args.cms_label,
            lumi=args.lumi,
            com=args.com,
        )
        print(f"[{mode}] unfolding data-prior response")
        weighted = unfold_once(
            spec,
            groomed=groomed,
            mc_inputs=weighted_mc,
            data_inputs=data_inputs,
            output_dir=output_dir / mode,
            label="data_prior",
            cms_label=args.cms_label,
            lumi=args.lumi,
            com=args.com,
        )

        artifacts.append(write_artifact(output_dir, mode, nominal, weighted))
        plot_outputs.extend(
            make_full_comparison_plots(
                output_dir,
                mode,
                nominal,
                weighted,
                cms_label=args.cms_label,
                lumi=args.lumi,
                com=args.com,
            )
        )
        summaries.append(
            {
                "mode": mode,
                "absolute_unfolded_shift": summarize_shift(nominal.y_unf, weighted.y_unf),
                "two_d_normalized_shift": summarize_shift(
                    nominal.unfolded_2dnorm_flat,
                    weighted.unfolded_2dnorm_flat,
                ),
                "visible_per_pt_normalized_shape_shift": summarize_visible_normalized_shift_by_pt(
                    nominal,
                    weighted,
                ),
                "prior_shift": summarize_shift(nominal.y_true, weighted.y_true),
                "fake_fraction_shift": summarize_shift(
                    1.0 - nominal.fake_fraction_2d,
                    1.0 - weighted.fake_fraction_2d,
                ),
                "negative_fake_bins_nominal": int(np.count_nonzero(nominal.fakes_2d < 0)),
                "negative_fake_bins_weighted": int(np.count_nonzero(weighted.fakes_2d < 0)),
                "negative_miss_bins_nominal": int(np.count_nonzero(nominal.misses_2d < 0)),
                "negative_miss_bins_weighted": int(np.count_nonzero(weighted.misses_2d < 0)),
            }
        )

    slide_pdf = write_slide_pdf(plot_outputs, output_dir)
    plot_files = []
    for item in plot_outputs:
        plot_files.extend([item["pdf"], item["png"]])

    manifest = {
        "workflow": "zjet rho data-prior response test",
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "tag": args.tag,
        "cms_label": args.cms_label,
        "integrated_luminosity_fb-1": args.lumi,
        "center_of_mass_energy_TeV": args.com,
        "stat_propagation": "jacobian" if args.jacobian else spec.stat_propagation,
        "regularization": args.regularization,
        "tau": args.tau,
        "root_version": ROOT.gROOT.GetVersion(),
        "inputs": {
            "nominal_mc": display_path(nominal_mc_path),
            "nominal_mc_sha256": file_sha256(nominal_mc_path),
            "weighted_mc": display_path(weighted_path),
            "weighted_mc_sha256": file_sha256(weighted_path),
            "data": display_path(data_path),
            "data_sha256": file_sha256(data_path),
        },
        "outputs": {
            "artifacts": [display_path(path) for path in artifacts],
            "plots": [display_path(path) for path in plot_files],
            "slide_pdf": display_path(slide_pdf),
        },
        "summaries": summaries,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    summary_path = output_dir / "run_summary.txt"
    with summary_path.open("w") as handle:
        handle.write("Z+jet rho data-prior response test\n")
        handle.write(f"Nominal MC: {display_path(nominal_mc_path)}\n")
        handle.write(f"Weighted MC: {display_path(weighted_path)}\n")
        handle.write(f"Data: {display_path(data_path)}\n\n")
        for item in summaries:
            handle.write(f"{item['mode']}:\n")
            for key in ("absolute_unfolded_shift", "two_d_normalized_shift", "prior_shift"):
                shift = item[key]
                handle.write(
                    f"  {key}: mean |ratio-1| = {shift['mean_abs']:.6f}, "
                    f"max |ratio-1| = {shift['max_abs']:.6f}\n"
                )
            handle.write("  visible per-pT normalized shape shift shown on plot:\n")
            for shift in item["visible_per_pt_normalized_shape_shift"]:
                pt_lo, pt_hi = shift["pt_bin"]
                pt_label = f"{pt_lo:g}-{pt_hi:g}" if pt_hi is not None else f"{pt_lo:g}-inf"
                handle.write(
                    f"    pT {pt_label} GeV: mean |ratio-1| = {shift['mean_abs']:.6f}, "
                    f"max |ratio-1| = {shift['max_abs']:.6f}\n"
                )
            handle.write(
                "  negative fake bins: "
                f"{item['negative_fake_bins_nominal']} nominal, "
                f"{item['negative_fake_bins_weighted']} weighted\n"
            )
            handle.write(
                "  negative miss bins: "
                f"{item['negative_miss_bins_nominal']} nominal, "
                f"{item['negative_miss_bins_weighted']} weighted\n\n"
            )

    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {slide_pdf}")


if __name__ == "__main__":
    main()
