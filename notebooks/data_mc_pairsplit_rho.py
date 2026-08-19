#!/usr/bin/env python3
"""Run-2 detector-level data/MC validation plots for the pair-split channels.

The pair-split sibling of ``data_mc_rho_fancy.py`` (Z+jet): one panel per
reported jet-pT slice for dijet/trijet x groomed/ungroomed, on the candidate
base reco binning actually used by the unfolding.

Two normalization facts make this comparison honest:

* the data carry pT-dependent trigger-prescale weights, so yields are only
  comparable within one pT slice — the MC is therefore normalized to the data
  yield over the displayed bins of each slice (stated on the panel), and no
  cross-slice yield claim is made;
* the displayed window matches the published one ([-4, 0] groomed,
  [-2.5, 0] ungroomed); the hidden low-rho migration buffer is not shown.

The uncertainty band is MC-stat (+) detector systematics (the same resolved
Run-2 virtual JES/JER legs and safe categories the unfolding uses, each
variation shape-normalized to data so only shape enters) (+) the two-leg
PS/HAD model band built from the audited pair-split model-envelope weights
(PS = max(MESS+Vincia, FSR); HAD = max(CR1, CR2, frag-hard, frag-soft)).
Data error bars use the diagonal of the same measured covariance fed to
TUnfold (event-clustered for dijet, sumw2 for trijet).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import LogLocator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unfold.tools.pairsplit_run2_inputs import (
    FULL_SAFE_SYSTEMATIC_REQUEST,
    LEGACY_HISTOGRAM_KEYS,
    PAIR_SPLIT_CHANNELS,
    PHYSICAL_RHO_DEFINITION,
    TRANSFORMED_COORDINATE_DEFINITION,
    load_pairsplit_run2_inputs,
    prepare_pairsplit_inputs,
    resolve_pairsplit_systematics,
)
from unfold.tools.pairsplit_model_envelope import (
    MODEL_SOURCES,
    derive_pairsplit_model_envelope_inputs,
)
from unfold.tools.pairsplit_vincia import load_pairsplit_vincia_source
from unfold.utils.cms_plot import (
    PUB_ANNOTATION_FONTSIZE,
    PUB_LABEL_FONTSIZE,
    PUB_LEGEND_FONTSIZE,
    PUB_TICK_FONTSIZE,
    save_cms_label_flavors,
)

# Mirrors scripts/run_pairsplit_unfolding.py: the delivered candidate per
# channel and the published display windows.
STUDY_RECOMMENDED_VARIANT_BY_CHANNEL = {"dijet": "coarse_tail", "trijet": "two_to_one"}
DISPLAY_WINDOWS = {"groomed": (-4.0, 0.0), "ungroomed": (-2.5, 0.0)}

MC_COLOR = "#f89c20"  # Petroff 6-colour scheme; data is always black
MC_LABEL = "QCD multijet (MG+Pythia8)"

# Model-band legs (same supersession logic as Unfolder._compute_total_systematic):
# fsr/isr/herwig belong to the model prescription, never to the detector
# quadrature; raw isr is excluded from the band entirely.
PS_EXTRA_CATEGORIES = ("fsrUp", "fsrDown")
DETECTOR_EXCLUDED_PREFIXES = ("model_", "fsr", "isr", "herwig")

hep.style.use("CMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", action="append", choices=PAIR_SPLIT_CHANNELS,
                        default=None, help="Repeatable; defaults to both channels.")
    parser.add_argument("--grooming-mode", choices=("groomed", "ungroomed", "both"),
                        default="both")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs" / "pairsplit_run2" / "data_mc")
    parser.add_argument("--cms-label", default="Internal")
    parser.add_argument("--lumi", type=float, default=138.0)
    parser.add_argument("--no-stamp", action="store_true",
                        help="Drop the working-plot provenance stamp (publication use).")
    args = parser.parse_args()
    args.channel = tuple(args.channel or PAIR_SPLIT_CHANNELS)
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_version() -> str:
    return subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()


def display_mask(edges: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    return (edges[:-1] >= low - 1e-9) & (edges[1:] <= high + 1e-9)


def reco_marginals(prepared, mode: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(values, variances) per systematic from the prepared reco histogram."""
    histogram = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS[mode]["reco"]]
    out = {}
    for systematic in prepared.systematics:
        selected = histogram[{"systematic": systematic}]
        out[systematic] = (
            np.asarray(selected.values(flow=False), dtype=float),
            np.asarray(selected.variances(flow=False), dtype=float),
        )
    return out


def scaled_to(values: np.ndarray, target_sum: float) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        return np.zeros_like(values)
    return values * (target_sum / total)


def band_components(
    mc_by_systematic: dict[str, tuple[np.ndarray, np.ndarray]],
    pt_index: int,
    shown: np.ndarray,
    data_sum: float,
) -> dict[str, np.ndarray]:
    """Per-bin absolute uncertainties on the data-normalized nominal MC.

    Every variation is independently shape-normalized to the data yield, so a
    coherent rate change contributes nothing (it is unobservable here).
    """

    def shown_scaled(systematic):
        values = mc_by_systematic[systematic][0][pt_index][shown]
        return scaled_to(values, data_sum)

    nominal = shown_scaled("nominal")
    nominal_variance = mc_by_systematic["nominal"][1][pt_index][shown]
    nominal_raw_sum = float(mc_by_systematic["nominal"][0][pt_index][shown].sum())
    scale = data_sum / nominal_raw_sum if nominal_raw_sum > 0 else 0.0
    mc_stat = np.sqrt(np.clip(nominal_variance, 0.0, None)) * scale

    # Detector systematics: symmetrize Up/Down pairs as the per-bin max shift.
    detector_sq = np.zeros_like(nominal)
    seen: set[str] = set()
    for name in mc_by_systematic:
        if name == "nominal" or name.startswith(DETECTOR_EXCLUDED_PREFIXES):
            continue
        base = name[:-2] if name.endswith("Up") else (
            name[:-4] if name.endswith("Down") else name)
        if base in seen:
            continue
        seen.add(base)
        shifts = [
            np.abs(shown_scaled(variation) - nominal)
            for variation in (f"{base}Up", f"{base}Down")
            if variation in mc_by_systematic
        ]
        if shifts:
            detector_sq += np.max(shifts, axis=0) ** 2
    detector = np.sqrt(detector_sq)

    # Two-leg model band from the model_* response-variation reco marginals.
    def model_shift(source):
        key = f"model_{source}"
        if key not in mc_by_systematic:
            raise KeyError(f"prepared inputs are missing model variation {key!r}")
        return np.abs(shown_scaled(key) - nominal)

    ps_candidates = [model_shift("vincia")]
    for extra in PS_EXTRA_CATEGORIES:
        if extra in mc_by_systematic:
            ps_candidates.append(np.abs(shown_scaled(extra) - nominal))
    ps = np.max(ps_candidates, axis=0)
    had = np.max([model_shift(s) for s in MODEL_SOURCES if s != "vincia"], axis=0)
    model = np.sqrt(ps**2 + had**2)

    return {
        "nominal": nominal,
        "mc_stat": mc_stat,
        "detector": detector,
        "model": model,
        "total": np.sqrt(mc_stat**2 + detector**2 + model**2),
    }


def pt_annotation(pt_edges: tuple[float, ...], index: int) -> str:
    low = int(pt_edges[index])
    if index + 1 < len(pt_edges) - 1:
        return rf"${low} < p_{{\mathrm{{T}}}} < {int(pt_edges[index + 1])}$ GeV"
    return rf"$p_{{\mathrm{{T}}}} > {low}$ GeV"


def make_panel(
    *,
    edges: np.ndarray,
    data_values: np.ndarray,
    data_errors: np.ndarray,
    components: dict[str, np.ndarray],
    mode: str,
    channel: str,
    annotation: str,
    cms_label: str,
    lumi: float,
    output_path: Path,
    stamp: str | None,
) -> None:
    fig, (axis, ratio_axis) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.07},
    )
    mc = components["nominal"]
    total = components["total"]
    hep.histplot(mc, bins=edges, histtype="fill", color=MC_COLOR,
                 label=MC_LABEL, ax=axis)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axis.errorbar(centers, data_values, yerr=data_errors, fmt="o", color="black",
                  markersize=5, label="Data", zorder=10)
    axis.stairs(mc + total, baseline=np.clip(mc - total, 0.0, None), edges=edges,
                label="Total unc.", hatch="///", edgecolor="black",
                facecolor="none", fill=True, alpha=1.0, zorder=9)

    axis.set_yscale("log")
    positive = data_values[data_values > 0]
    floor = float(positive.min()) / 5.0 if positive.size else 0.1
    data_top = float(max(data_values.max(), 1.0))
    # Fraction-based log headroom: the tallest bin sits at ~65% of the axis
    # height in every panel, clearing the legend and annotation block without
    # the decades of empty space a fixed multiplier leaves on peaked panels.
    span = np.log10(max(data_top / floor, 10.0))
    axis.set_ylim(floor, floor * 10 ** (span / 0.65))
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    axis.set_ylabel("Events", fontsize=PUB_LABEL_FONTSIZE)
    axis.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
    handles, labels = axis.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda k: (labels[k] != "Data", labels[k] == "Total unc."))
    axis.legend([handles[k] for k in order], [labels[k] for k in order],
                ncol=2, fontsize=PUB_LEGEND_FONTSIZE, loc="upper left",
                columnspacing=1.2, handletextpad=0.5)
    displayed_lumi = int(lumi) if float(lumi).is_integer() else lumi
    hep.cms.label(cms_label, data=True, lumi=displayed_lumi, com=13, ax=axis)
    axis.text(0.97, 0.96, f"{channel}\n{annotation}\nMC normalized to data",
              transform=axis.transAxes, ha="right", va="top",
              fontsize=PUB_ANNOTATION_FONTSIZE)

    ratio = np.divide(data_values, mc, out=np.full_like(data_values, np.nan),
                      where=mc > 0)
    ratio_error = np.divide(data_errors, mc, out=np.zeros_like(data_errors),
                            where=mc > 0)
    relative = np.divide(total, mc, out=np.zeros_like(total), where=mc > 0)
    ratio_axis.errorbar(centers, ratio, yerr=ratio_error, fmt="o", color="black",
                        markersize=5)
    ratio_axis.stairs(1.0 + relative, baseline=1.0 - relative, edges=edges,
                      hatch="///", edgecolor="black", facecolor="none",
                      fill=True, alpha=1.0)
    ratio_axis.axhline(1.0, color="red", linestyle="--")
    ratio_axis.set_ylim(0.0, 2.0)
    ratio_axis.set_yticks([0.5, 1.0, 1.5])
    ratio_axis.set_ylabel("Data/MC", fontsize=PUB_LABEL_FONTSIZE)
    ratio_axis.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
    ratio_axis.set_xlabel(rf"$\log_{{10}}(\rho^2)$, {mode}",
                          fontsize=PUB_LABEL_FONTSIZE)
    ratio_axis.set_xlim(edges[0], edges[-1])
    if stamp is not None:
        fig.text(0.99, 0.002, stamp, ha="right", va="bottom", fontsize=7,
                 color="0.45", family="monospace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_cms_label_flavors(fig, output_path, cms_label)
    plt.close(fig)


def run_channel_mode(args, channel: str, mode: str, inputs, vincia_source) -> dict:
    variant = STUDY_RECOMMENDED_VARIANT_BY_CHANNEL[channel]
    window = DISPLAY_WINDOWS[mode]
    resolved = resolve_pairsplit_systematics(
        inputs.modes[mode].systematics, FULL_SAFE_SYSTEMATIC_REQUEST
    )
    model_envelope = derive_pairsplit_model_envelope_inputs(
        inputs, vincia_source, variant=variant,
        normalization_window=window, grooming_mode=mode,
    )
    prepared = prepare_pairsplit_inputs(
        inputs, variant, resolved, grooming_mode=mode,
        model_variations=model_envelope.prepared_variations(),
        model_metadata=model_envelope.provenance_payload(),
    )
    reco_key = LEGACY_HISTOGRAM_KEYS[mode]["reco"]
    data_histogram = prepared.data_inputs[reco_key][{"systematic": "nominal"}]
    data_values = np.asarray(data_histogram.values(flow=False), dtype=float)
    covariance_diag = np.diag(prepared.measured_covariance).reshape(data_values.shape)
    mc_by_systematic = reco_marginals(prepared, mode)

    edges = np.asarray(prepared.analysis_binning.two_log10_rho_reco_edges, float)
    shown = display_mask(edges, window)
    shown_edges = np.concatenate((edges[:-1][shown], [edges[1:][shown][-1]]))

    output_dir = args.output_root / channel
    stamp = None if args.no_stamp else (
        f"{datetime.date.today().isoformat()}  |  unfold {repo_version()}  |  "
        f"inputs: pairsplit_run2 {variant} {mode}"
    )
    outputs = []
    for pt_index in range(len(prepared.analysis_binning.pt_edges) - 1):
        data_shown = data_values[pt_index][shown]
        data_sum = float(data_shown.sum())
        components = band_components(mc_by_systematic, pt_index, shown, data_sum)
        output_path = output_dir / f"data_mc_{mode}_pt{pt_index}.pdf"
        make_panel(
            edges=shown_edges,
            data_values=data_shown,
            data_errors=np.sqrt(np.clip(covariance_diag[pt_index][shown], 0.0, None)),
            components=components,
            mode=mode,
            channel=channel,
            annotation=pt_annotation(prepared.analysis_binning.pt_edges, pt_index),
            cms_label=args.cms_label,
            lumi=args.lumi,
            output_path=output_path,
            stamp=stamp,
        )
        outputs.append(str(output_path))
        print(f"  wrote {output_path}")
    return {
        "channel": channel,
        "grooming_mode": mode,
        "candidate": variant,
        "display_window": list(window),
        "normalization": "MC scaled to the data yield over the displayed bins, per pT slice",
        "data_error_source": prepared.metadata["data_covariance_source"],
        "band": (
            "MC stat (+) detector systematics (per-source max|Up/Down|, quadrature, "
            "each variation shape-normalized to data) (+) model "
            "PS=max(MESS+Vincia,FSR), HAD=max(CR1,CR2,frag), quadrature"
        ),
        "systematics": list(prepared.systematics),
        "rho_definition": PHYSICAL_RHO_DEFINITION,
        "coordinate": TRANSFORMED_COORDINATE_DEFINITION,
        "inputs": [
            {"era": s.era,
             "mc": {"path": str(s.mc), "sha256": file_sha256(s.mc)},
             "data": {"path": str(s.data), "sha256": file_sha256(s.data)}}
            for s in inputs.source_files
        ],
        "model_envelope": model_envelope.identity_payload(),
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    modes = (("groomed", "ungroomed") if args.grooming_mode == "both"
             else (args.grooming_mode,))
    provenance = {
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()),
                               *sys.argv[1:]]),
        "repo_version": repo_version(),
        "runs": [],
    }
    for channel in args.channel:
        inputs = load_pairsplit_run2_inputs(channel)
        vincia_source = load_pairsplit_vincia_source(channel)
        for mode in modes:
            print(f"data/MC: {channel} {mode}")
            provenance["runs"].append(
                run_channel_mode(args, channel, mode, inputs, vincia_source)
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n",
                               encoding="utf-8")
    print(provenance_path)


if __name__ == "__main__":
    main()
