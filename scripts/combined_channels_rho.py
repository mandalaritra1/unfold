#!/usr/bin/env python3
"""Combined three-channel log10(rho^2) comparison on the aligned common grid.

The 2026-08-27 aligned re-production put dijet, trijet, and Z+jet gen edges on
one lattice, so the three unfolded results rebin EXACTLY onto the common grid
(groomed [-3.5,-2.5,-2,-1.5,-1,0]; ungroomed [-2.5,-2,-1.5,-1,0]) and the
common pT slices [200,290], [290,400], [400,inf).  Every channel is
re-normalized here to unit area over the common window, so per-run
normalization-window choices (dijet peak window, zjet per-slice windows) drop
out.

Inputs and treatment:
- dijet/trijet: the aligned run artifacts' RAW unfolded counts, full stat
  covariance, and all per-source systematic results.  pT slices are merged at
  the count level (dijet 400-480/480-570/570-inf -> 400-inf, with cross-slice
  stat covariance), rho bins merged by exact edge nesting, then normalized;
  the systematic band regroups the per-source shifts with the canonical
  combination (detector quadrature excluding model_*/fsr/isr/herwig;
  PS = max(MESS+Vincia, FSR); HAD = max(CR1, CR2, frag-hard, frag-soft)).
- zjet: the published normalized-density payload with its stat and total
  covariances (per-source raws are not stored there); densities are converted
  to per-slice counts, rebinned, and re-normalized with exact Jacobians.
Channels are treated as uncorrelated in the ratio (independent datasets;
JES-source correlations across channels are ignored at this stage).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from unfold.cms_plot import (  # noqa: E402
    PUB_ANNOTATION_FONTSIZE,
    PUB_LABEL_FONTSIZE,
    PUB_LEGEND_FONTSIZE,
    PUB_TICK_FONTSIZE,
    save_cms_label_flavors,
)

hep.style.use(hep.style.CMS)

OUTPUT_ROOT = ROOT / "outputs" / "hadronic"
OUTPUT_DIR = OUTPUT_ROOT / "combined"

# the production runs of `unfold run --channel <channel>` (tag "original")
HADRONIC_ARTIFACTS = {
    (channel, mode): ROOT / "outputs" / channel / "rho" / "original" / mode / "artifacts" / f"{mode}_results.npz"
    for channel in ("dijet", "trijet") for mode in ("groomed", "ungroomed")
}
ZJET_DIR = ROOT / "outputs/zjet/rho/original/data"
ZJET_NPZ = {
    "groomed": ZJET_DIR / "normalized_covariance_groomed.npz",
    "ungroomed": ZJET_DIR / "normalized_covariance_ungroomed.npz",
}
ZJET_PKL = {
    "groomed": ZJET_DIR / "unfolded_2d_groomed.pkl",
    "ungroomed": ZJET_DIR / "unfolded_2d_ungroomed.pkl",
}

COMMON_EDGES = {
    "groomed": np.array([-3.5, -2.5, -2.0, -1.5, -1.0, 0.0]),
    "ungroomed": np.array([-2.5, -2.0, -1.5, -1.0, 0.0]),
}
COMMON_PT_SLICES = ((200.0, 290.0), (290.0, 400.0), (400.0, None))

CHANNEL_STYLE = {
    "dijet": {"color": "#e76300", "band": "#fdd49e", "label": "Dijet", "marker": "o"},
    "trijet": {"color": "#00707f", "band": "#92dadd", "label": "Trijet", "marker": "s"},
    "zjet": {"color": "#1b7837", "band": "#9acd32", "label": "Z+jet", "marker": "^"},
}

MODEL_LEG_SOURCES = ("model_cr1", "model_cr2", "model_fraghard", "model_fragsoft")
DETECTOR_EXCLUDE_PREFIXES = ("model_", "fsr", "isr", "herwig")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nesting_groups(source_edges: np.ndarray, target_edges: np.ndarray) -> list[list[int]]:
    """Source-bin index groups for each target bin; exact edge nesting only."""

    groups: list[list[int]] = []
    for low, high in zip(target_edges[:-1], target_edges[1:]):
        selected = [
            index
            for index in range(len(source_edges) - 1)
            if source_edges[index] >= low - 1e-9 and source_edges[index + 1] <= high + 1e-9
        ]
        if (
            not selected
            or not np.isclose(source_edges[selected[0]], low)
            or not np.isclose(source_edges[selected[-1] + 1], high)
        ):
            raise ValueError(
                f"target bin [{low}, {high}] does not nest in {list(source_edges)}"
            )
        groups.append(selected)
    return groups


def merge_matrix(
    n_pt_source: int,
    n_gen_source: int,
    pt_groups: list[list[int]],
    gen_groups: list[list[int]],
) -> np.ndarray:
    """Counts-summing matrix (n_out_slices*n_common, n_pt_source*n_gen_source)."""

    n_out = len(pt_groups) * len(gen_groups)
    matrix = np.zeros((n_out, n_pt_source * n_gen_source))
    for out_slice, pt_group in enumerate(pt_groups):
        for out_bin, gen_group in enumerate(gen_groups):
            row = out_slice * len(gen_groups) + out_bin
            for pt_index in pt_group:
                for gen_index in gen_group:
                    matrix[row, pt_index * n_gen_source + gen_index] = 1.0
    return matrix


def normalize_slices(
    counts: np.ndarray, widths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-area densities per slice plus the per-slice totals."""

    totals = counts.sum(axis=1)
    density = counts / (totals[:, None] * widths[None, :])
    return density, totals


def normalize_jacobian(counts: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """d(density)/d(counts) for the flattened (slice, bin) layout."""

    n_slices, n_bins = counts.shape
    n = n_slices * n_bins
    jac = np.zeros((n, n))
    for s in range(n_slices):
        y = counts[s]
        total = y.sum()
        block = np.eye(n_bins) / (total * widths[:, None]) - np.outer(
            y / (total**2 * widths), np.ones(n_bins)
        )
        sl = slice(s * n_bins, (s + 1) * n_bins)
        jac[sl, sl] = block
    return jac


def load_hadronic_channel(channel: str, mode: str) -> dict:
    path = HADRONIC_ARTIFACTS[(channel, mode)]
    with np.load(path, allow_pickle=True) as f:
        gen_edges = np.asarray(f["two_log10_rho_gen_edges"], dtype=float)
        pt_edges = np.asarray(f["pt_edges"], dtype=float)
        raw = np.asarray(f["unfolded"], dtype=float)
        stat_cov = np.asarray(f["covariance"], dtype=float)
        names = [str(s) for s in f["systematic_names"]]
        syst_raw = np.asarray(f["systematic_unfolded"], dtype=float)

    n_gen = len(gen_edges) - 1
    n_pt = len(pt_edges) - 1
    assert raw.size == n_pt * n_gen, (channel, mode, raw.shape, n_pt, n_gen)

    common = COMMON_EDGES[mode]
    widths = np.diff(common)
    # rho: drop the [-10, floor] buffer; merge shown bins onto the common grid.
    gen_groups = nesting_groups(gen_edges, common)
    # pT: map channel slices onto the three common slices.
    pt_groups: list[list[int]] = []
    for low, high in COMMON_PT_SLICES:
        group = [
            index
            for index in range(n_pt)
            if pt_edges[index] >= low - 1e-6
            and (high is None or pt_edges[index + 1] <= high + 1e-6)
        ]
        if not group or not np.isclose(pt_edges[group[0]], low):
            raise ValueError(f"{channel} {mode}: no pT slices for [{low}, {high}]")
        pt_groups.append(group)

    G = merge_matrix(n_pt, n_gen, pt_groups, gen_groups)
    merged = (G @ raw).reshape(len(pt_groups), len(widths))
    merged_cov = G @ stat_cov @ G.T

    density, _ = normalize_slices(merged, widths)
    jac = normalize_jacobian(merged, widths)
    stat_err = np.sqrt(np.clip(np.diag(jac @ merged_cov @ jac.T), 0.0, None)).reshape(
        density.shape
    )

    # Per-source shifts on the normalized densities.
    shifts: dict[str, np.ndarray] = {}
    for name, row in zip(names, syst_raw):
        varied = (G @ np.asarray(row, dtype=float)).reshape(merged.shape)
        varied_density, _ = normalize_slices(varied, widths)
        shifts[name] = varied_density - density

    def updown_base(name: str) -> str | None:
        if name.endswith("Up"):
            return name[:-2]
        if name.endswith("Down"):
            return name[:-4]
        return None

    detector_sq = np.zeros_like(density)
    seen_bases = set()
    for name in names:
        base = updown_base(name)
        if base is None or base in seen_bases:
            continue
        if any(base.lower().startswith(p) for p in DETECTOR_EXCLUDE_PREFIXES):
            continue
        seen_bases.add(base)
        up = np.abs(shifts.get(f"{base}Up", np.zeros_like(density)))
        down = np.abs(shifts.get(f"{base}Down", np.zeros_like(density)))
        detector_sq += np.maximum(up, down) ** 2

    fsr = np.maximum(
        np.abs(shifts.get("fsrUp", np.zeros_like(density))),
        np.abs(shifts.get("fsrDown", np.zeros_like(density))),
    )
    ps = np.maximum(np.abs(shifts.get("model_vincia", np.zeros_like(density))), fsr)
    had = np.zeros_like(density)
    for source in MODEL_LEG_SOURCES:
        had = np.maximum(had, np.abs(shifts.get(source, np.zeros_like(density))))

    syst = np.sqrt(detector_sq + ps**2 + had**2)
    total_err = np.sqrt(stat_err**2 + syst**2)
    return {
        "density": density,
        "stat": stat_err,
        "total": total_err,
        "source": path,
    }


def load_zjet(mode: str) -> dict:
    import pickle

    npz_path = ZJET_NPZ[mode]
    pkl_path = ZJET_PKL[mode]
    payload = pickle.load(open(pkl_path, "rb"))
    hist_abs = payload["unfolded_abs"]
    gen_edges = np.asarray(hist_abs.axes[1].edges, dtype=float)
    pt_edges = np.asarray(payload["pt_edges"], dtype=float)
    with np.load(npz_path, allow_pickle=True) as f:
        n_slices = len(pt_edges) - 1
        n_gen = len(gen_edges) - 1
        density = np.asarray(f["normalized"], dtype=float).reshape(n_slices, n_gen)
        cov_stat = np.asarray(f["cov_stat"], dtype=float)
        cov_total = np.asarray(f["cov_total"], dtype=float)

    # The zjet payload stores per-slice densities; convert to per-slice
    # "counts" (arbitrary per-slice scale, dropped again by renormalization).
    source_widths = np.diff(gen_edges)
    counts = density * source_widths[None, :]
    width_transform = np.kron(np.eye(n_slices), np.diag(source_widths))

    common = COMMON_EDGES[mode]
    widths = np.diff(common)
    gen_groups = nesting_groups(gen_edges, common)
    pt_groups = []
    for low, high in COMMON_PT_SLICES:
        group = [
            index
            for index in range(n_slices)
            if pt_edges[index] >= low - 1e-6
            and (high is None or pt_edges[index + 1] <= high + 1e-6)
        ]
        if not group:
            raise ValueError(f"zjet {mode}: no pT slice for [{low}, {high}]")
        pt_groups.append(group)

    G = merge_matrix(n_slices, n_gen, pt_groups, gen_groups)
    M = G @ width_transform
    merged = (M @ density.reshape(-1)).reshape(len(pt_groups), len(widths))
    out_density, _ = normalize_slices(merged, widths)
    jac = normalize_jacobian(merged, widths) @ M
    stat_err = np.sqrt(np.clip(np.diag(jac @ cov_stat @ jac.T), 0.0, None)).reshape(
        out_density.shape
    )
    total_err = np.sqrt(np.clip(np.diag(jac @ cov_total @ jac.T), 0.0, None)).reshape(
        out_density.shape
    )
    return {
        "density": out_density,
        "stat": stat_err,
        "total": total_err,
        "source": npz_path,
    }


def slice_label(index: int) -> str:
    low, high = COMMON_PT_SLICES[index]
    return f"{low:g} < $p_T$ < {high:g} GeV" if high else f"$p_T$ > {low:g} GeV"


# Within a bin the three channels' error bars sit at these fractions of the
# bin width, so no bar ever overlaps another channel's.
BAR_OFFSET = {"dijet": 0.25, "trijet": 0.5, "zjet": 0.75}
RATIO_OFFSET = {"trijet": 0.35, "zjet": 0.65}
RATIO_CAP = 2.0
CHANNEL_ORDER = ("dijet", "trijet", "zjet")


def ratio_to_dijet(channels: dict, slice_index: int):
    """(ratio, error) per channel for trijet and zjet; channels uncorrelated."""
    dijet = channels["dijet"]["density"][slice_index]
    out = {}
    for name in ("trijet", "zjet"):
        payload = channels[name]
        density = payload["density"][slice_index]
        ratio = np.divide(density, dijet, out=np.full_like(density, np.nan), where=dijet > 0)
        rel = np.divide(payload["total"][slice_index], density,
                        out=np.zeros_like(density), where=density > 0)
        out[name] = (ratio, np.abs(ratio) * rel)
    return out


def ratio_limits(channels: dict) -> tuple[float, float]:
    """One ratio range for every pT slice, capped at RATIO_CAP (arrows beyond)."""
    low, high, marker_high = 0.7, 1.3, 1.0
    for slice_index in range(len(COMMON_PT_SLICES)):
        for ratio, error in ratio_to_dijet(channels, slice_index).values():
            finite = np.isfinite(ratio)
            if finite.any():
                low = min(low, float(np.min((ratio - error)[finite])))
                high = max(high, float(np.max((ratio + error)[finite])))
                marker_high = max(marker_high, float(np.max(ratio[finite])))
    # bars may run into the frame; markers must not, so the cap yields to them
    top = min(RATIO_CAP, np.ceil(high * 10) / 10 + 0.05)
    return (max(0.0, np.floor(low * 10) / 10 - 0.05), max(top, marker_high + 0.15))


def draw_slice(axis, ratio_axis, mode: str, slice_index: int, channels: dict, *,
               ymax: float, ratio_lim: tuple[float, float], first: bool, legend: bool):
    """One pT slice: step outlines for the shape, offset error bars for the
    uncertainty (thick = stat, thin with caps = total), ratio to dijet below."""
    edges = COMMON_EDGES[mode]
    widths = np.diff(edges)
    for edge in edges[1:-1]:
        axis.axvline(edge, color="0.85", linewidth=0.8, zorder=0)
        ratio_axis.axvline(edge, color="0.85", linewidth=0.8, zorder=0)
    for name in CHANNEL_ORDER:
        payload, style = channels[name], CHANNEL_STYLE[name]
        x = edges[:-1] + BAR_OFFSET[name] * widths
        density = payload["density"][slice_index]
        axis.stairs(density, edges, color=style["color"], linewidth=2.6, zorder=3)
        axis.errorbar(x, density, yerr=payload["total"][slice_index], fmt="none",
                      ecolor=style["color"], elinewidth=1.2, capsize=4, alpha=0.85, zorder=4)
        axis.errorbar(x, density, yerr=payload["stat"][slice_index], fmt=style["marker"],
                      color=style["color"], markersize=7, elinewidth=2.6, capsize=0, zorder=5)
    axis.set_ylim(-0.04 * ymax, ymax)   # a bin compatible with zero keeps its marker visible
    axis.set_xlim(edges[0], edges[-1])
    axis.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE - 6)
    axis.text(0.96, 0.95, f"{slice_label(slice_index)}\n{mode}", transform=axis.transAxes,
              ha="right", va="top", fontsize=PUB_ANNOTATION_FONTSIZE - 2)
    if first:
        # Single-line form: the stacked \frac gets its numerator clipped at the
        # figure edge on the shorter ungroomed panels.
        axis.set_ylabel(r"$(1/N)\; dN/d\log_{10}(\rho^2)$", fontsize=PUB_LABEL_FONTSIZE - 6)
        ratio_axis.set_ylabel("Ratio to dijet", fontsize=PUB_LABEL_FONTSIZE - 9)
    else:
        axis.tick_params(labelleft=False)
        ratio_axis.tick_params(labelleft=False)
    if legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], color=CHANNEL_STYLE[name]["color"], linewidth=2.6,
                   marker=CHANNEL_STYLE[name]["marker"], markersize=8,
                   label=CHANNEL_STYLE[name]["label"])
            for name in CHANNEL_ORDER
        ]
        axis.legend(handles=handles, fontsize=PUB_LEGEND_FONTSIZE - 1, loc="upper left",
                    frameon=False, borderaxespad=1.2)

    # ratio row: dijet total uncertainty as a grey band, the other two as
    # step outlines (a binned ratio, so no lines between bins) plus offset bars
    dijet = channels["dijet"]
    dijet_density = dijet["density"][slice_index]
    dijet_rel = np.divide(dijet["total"][slice_index], dijet_density,
                          out=np.zeros_like(dijet_density), where=dijet_density > 0)
    ratio_axis.stairs(1.0 + dijet_rel, edges, baseline=1.0 - dijet_rel, fill=True,
                      color="0.82", alpha=0.9, zorder=0)
    ratio_axis.axhline(1.0, color=CHANNEL_STYLE["dijet"]["color"], linewidth=1.2)
    span = ratio_lim[1] - ratio_lim[0]
    for name, (ratio, error) in ratio_to_dijet(channels, slice_index).items():
        style = CHANNEL_STYLE[name]
        x = edges[:-1] + RATIO_OFFSET[name] * widths
        ratio_axis.stairs(np.where(np.isfinite(ratio), ratio, np.nan), edges,
                          color=style["color"], linewidth=2.0, zorder=3)
        ratio_axis.errorbar(x, ratio, yerr=error, fmt=style["marker"], color=style["color"],
                            markersize=7, capsize=3, linewidth=1.6, zorder=4)
        for xx, rr in zip(x, ratio):
            if np.isfinite(rr) and rr > ratio_lim[1]:
                ratio_axis.annotate("", xy=(xx, ratio_lim[1] - 0.02 * span),
                                    xytext=(xx, ratio_lim[1] - 0.2 * span),
                                    arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=1.4))
    ratio_axis.set_ylim(*ratio_lim)
    ratio_axis.set_xticks(edges)
    ratio_axis.set_xticklabels([("" if (not first and k == 0) else f"{e:g}")
                                for k, e in enumerate(edges)])
    ratio_axis.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE - 6)
    ratio_axis.set_xlabel(r"$\log_{10}(\rho^2)$", fontsize=PUB_LABEL_FONTSIZE - 6)


def slice_ymax(channels: dict) -> float:
    return 1.45 * max(float(np.max(channels[name]["density"][i] + channels[name]["total"][i]))
                      for name in CHANNEL_ORDER for i in range(len(COMMON_PT_SLICES)))


def draw_combined_row(mode: str, channels: dict, cms_label: str) -> Path:
    """The paper figure: the three pT slices side by side in one PDF."""
    n = len(COMMON_PT_SLICES)
    fig, axes = plt.subplots(2, n, figsize=(7.2 * n, 9.0), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.05, "wspace": 0.04})
    ymax, ratio_lim = slice_ymax(channels), ratio_limits(channels)
    for i in range(n):
        draw_slice(axes[0, i], axes[1, i], mode, i, channels, ymax=ymax, ratio_lim=ratio_lim,
                   first=(i == 0), legend=(i == 0))
    hep.cms.text(cms_label, ax=axes[0, 0], fontsize=PUB_ANNOTATION_FONTSIZE)
    axes[0, -1].text(1.0, 1.01, r"138 fb$^{-1}$ (13 TeV)", transform=axes[0, -1].transAxes,
                     ha="right", va="bottom", fontsize=PUB_ANNOTATION_FONTSIZE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"combined_{mode}.pdf"
    save_cms_label_flavors(fig, output_path, cms_label, bbox_inches="tight")
    plt.close(fig)
    return output_path


def draw_combined(mode: str, slice_index: int, channels: dict, cms_label: str) -> Path:
    """One pT slice per PDF (talks); same drawing as the paper row."""
    fig, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.05},
    )
    draw_slice(axis, ratio_axis, mode, slice_index, channels, ymax=slice_ymax(channels),
               ratio_lim=ratio_limits(channels), first=True, legend=True)
    hep.cms.label(cms_label, data=True, lumi=138, ax=axis)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"combined_{mode}_pt{slice_index}.pdf"
    save_cms_label_flavors(fig, output_path, cms_label)
    plt.close(fig)
    return output_path


def main() -> None:
    provenance: dict[str, object] = {"inputs": {}, "outputs": []}
    for mode in ("groomed", "ungroomed"):
        channels = {
            "dijet": load_hadronic_channel("dijet", mode),
            "trijet": load_hadronic_channel("trijet", mode),
            "zjet": load_zjet(mode),
        }
        for name, payload in channels.items():
            source = Path(payload["source"])
            provenance["inputs"][f"{name}_{mode}"] = {
                "path": str(source),
                "sha256": sha256_path(source),
            }
        output = draw_combined_row(mode, channels, "Internal")
        provenance["outputs"].append(str(output))
        print("wrote", output)
        for slice_index in range(len(COMMON_PT_SLICES)):
            output = draw_combined(mode, slice_index, channels, "Internal")
            provenance["outputs"].append(str(output))
            print("wrote", output)
    provenance["common_edges"] = {
        mode: list(map(float, edges)) for mode, edges in COMMON_EDGES.items()
    }
    provenance["pt_slices_GeV"] = [list(s) for s in COMMON_PT_SLICES]
    provenance["ratio_convention"] = "channel / dijet; channels uncorrelated"
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(OUTPUT_DIR / "provenance.json")


if __name__ == "__main__":
    main()
