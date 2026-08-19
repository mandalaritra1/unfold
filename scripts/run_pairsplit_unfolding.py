#!/usr/bin/env python3
"""Run Run-2 pair-split unfolding through the shared TUnfold core.

The physical observable is rho = m / (pT * R), R = 0.8.  Its stored
coordinate is two_log10_rho = 2 * log10(rho).  This entry point uses TUnfold
with built-in input and response-matrix statistics; it does not use
jackknife replicas or iterative Bayes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unfold.tools.pairsplit_run2_inputs import (
    PAIR_SPLIT_CHANNELS,
    PAIR_SPLIT_INPUT_ROOT,
    PAIR_SPLIT_BINNING_VARIANTS,
    FULL_SAFE_SYSTEMATIC_REQUEST,
    LEGACY_HISTOGRAM_KEYS,
    PHYSICAL_RHO_DEFINITION,
    TRANSFORMED_COORDINATE_DEFINITION,
    TRANSFORMED_COORDINATE_NAME,
    PairSplitPreparedInputs,
    PairSplitSourceFiles,
    load_pairsplit_run2_inputs,
    prepare_pairsplit_inputs,
    resolve_pairsplit_systematics,
)
from unfold.tools.pairsplit_model_envelope import (
    PairSplitModelEnvelopeInputs,
    derive_pairsplit_model_envelope_inputs,
)
from unfold.tools.pairsplit_vincia import (
    attach_pairsplit_vincia_prediction,
    derive_pairsplit_vincia_prediction,
    load_pairsplit_vincia_source,
    validate_compiled_reference,
)


DEFAULT_SYSTEMATIC_REQUEST = FULL_SAFE_SYSTEMATIC_REQUEST
# These narrower intervals remain useful for a stable, like-for-like study
# diagnostic.  They are deliberately distinct from the named peak-window
# normalization variant below.
CORE_STABILITY_WINDOWS = {
    "dijet": (-2.85, -0.55),
    "trijet": (-3.0, -0.7),
}
PLOTTED_DISPLAY_WINDOW = (-4.0, 0.0)
UNGROOMED_WINDOW = (-2.5, 0.0)
NORMALIZATION_WINDOWS = {
    "full": PLOTTED_DISPLAY_WINDOW,
    # The earlier hadronic-rho bin study's ``_win18`` dijet variant.  The
    # upper edge deliberately excludes the [-0.55, 0] catch-all bin.
    "peak": (-1.8, -0.55),
}
STUDY_RECOMMENDED_BINNING = "study_recommended"
STUDY_RECOMMENDED_VARIANT_BY_CHANNEL = {
    "dijet": "coarse_tail",
    "trijet": "two_to_one",
}


class RunDiagnostics:
    """Numerical audit payload kept separate from the unfolding core state."""

    def __init__(self, *, metrics: Mapping[str, object], arrays: Mapping[str, np.ndarray]):
        self.metrics = metrics
        self.arrays = arrays


def _json_native(value):
    """Convert nested diagnostic metrics to strict builtin JSON scalar types."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_native(item) for item in value]
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grooming-mode",
        choices=("groomed", "ungroomed", "both"),
        default="groomed",
        help=(
            "Observable mode. 'both' runs independent groomed and ungroomed "
            "unfolds; ungroomed is shown and normalized over [-2.5, 0]."
        ),
    )
    parser.add_argument(
        "--channel",
        action="append",
        choices=PAIR_SPLIT_CHANNELS,
        default=None,
        help="Channel to run; repeat this option. Defaults to both channels.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PAIR_SPLIT_INPUT_ROOT,
        help="Root containing <era>/<channel>_{mc,data} pair-split pickles.",
    )
    parser.add_argument(
        "--binning",
        choices=(STUDY_RECOMMENDED_BINNING, *PAIR_SPLIT_BINNING_VARIANTS),
        default=STUDY_RECOMMENDED_BINNING,
        help=(
            "Named candidate base-to-GEN aggregation. The default resolves "
            "to coarse_tail for dijet and two_to_one for trijet; an explicit "
            "candidate applies unchanged to every requested channel."
        ),
    )
    parser.add_argument(
        "--regularization",
        choices=("curvature", "none"),
        default="none",
        help="TUnfold regularization mode.",
    )
    parser.add_argument(
        "--normalization-window",
        choices=tuple(NORMALIZATION_WINDOWS),
        default="full",
        help=(
            "Per-pT unit-area denominator in two_log10_rho. 'full' uses "
            "[-4, 0]; 'peak' uses the prior dijet study window [-1.8, -0.55] "
            "while retaining [-4, 0] on the plots."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Optional fixed TUnfold tau; otherwise curvature scans the L-curve.",
    )
    parser.add_argument(
        "--systematics",
        default=DEFAULT_SYSTEMATIC_REQUEST,
        help=(
            "Comma-separated exact names or safe bases; e.g. "
            f"{DEFAULT_SYSTEMATIC_REQUEST}. JES and JER select the derived "
            "Run-2 correlated/era-uncorrelated virtual legs; raw combined "
            "producer categories are not selectable."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "pairsplit_run2",
        help="Root for per-channel candidate outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Canary option: save arrays and manifest without generating plots.",
    )
    parser.add_argument(
        "--model-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the Z+jet two-leg model prescription with prepared pair-split "
            "MESS+Vincia/CR/fragmentation response variations. Use "
            "--no-model-envelope only as a legacy comparison."
        ),
    )
    parser.add_argument("--cms-label", default="Internal")
    parser.add_argument("--lumi", type=float, default=138.0)
    parser.add_argument("--com", type=float, default=13.0)
    args = parser.parse_args(argv)
    if args.tau is not None and args.regularization == "none":
        parser.error("--tau requires --regularization curvature")
    args.channel = tuple(args.channel or PAIR_SPLIT_CHANNELS)
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_metadata(source_files: Sequence[PairSplitSourceFiles]) -> list[dict]:
    records = []
    for source in source_files:
        records.append(
            {
                "era": source.era,
                "mc": {"path": str(source.mc.resolve()), "sha256": file_sha256(source.mc)},
                "data": {"path": str(source.data.resolve()), "sha256": file_sha256(source.data)},
            }
        )
    return records


def _core_output_directory(path: Path) -> str:
    """Use a core-safe relative path even when --output-root is external."""

    return os.path.relpath(path.resolve(), REPO_ROOT) + "/"


def _tau_directory_token(regularization: str, tau: float | None) -> str:
    """Return a path-safe, deterministic description of the requested tau."""

    if tau is None:
        return "lcurve" if regularization == "curvature" else "disabled"
    return f"{tau:.17g}".replace("-", "m").replace("+", "p").replace(".", "p")


def run_configuration_identity(
    args: argparse.Namespace,
    resolved_systematics: Sequence[str],
    mess_vincia_source: Mapping[str, object],
    model_envelope_source: Mapping[str, object] | None = None,
    *,
    grooming_mode: str = "groomed",
) -> dict[str, object]:
    """Identify the physics configuration used by one immutable output directory.

    The short fingerprint avoids filesystem-length limits for the full
    systematic list while the manifest retains the complete canonical payload.
    """

    normalization_name, normalization_window = resolved_normalization(
        args, grooming_mode
    )
    configuration = {
        "grooming_mode": grooming_mode,
        "binning": args.binning,
        "normalization_window": {
            "name": normalization_name,
            "two_log10_rho_range": list(normalization_window),
        },
        "regularization": args.regularization,
        "requested_tau": args.tau,
        "systematics": list(resolved_systematics),
        # This is deliberately part of the directory identity: a new audited
        # MESS campaign or source hash must never reuse a stale Vincia overlay.
        "mess_vincia_source": mess_vincia_source,
        "model_envelope": {
            "enabled": bool(args.model_envelope),
            "source": model_envelope_source,
        },
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    directory_name = (
        f"regularization-{args.regularization}"
        f"__tau-{_tau_directory_token(args.regularization, args.tau)}"
        f"__normalization-{normalization_name}"
        f"__systematics-{fingerprint}"
    )
    return {
        "directory_name": directory_name,
        "fingerprint": fingerprint,
        "configuration": configuration,
    }


def resolved_normalization(
    args: argparse.Namespace, grooming_mode: str
) -> tuple[str, tuple[float, float]]:
    """Resolve the mode-specific published unit-area interval."""

    if grooming_mode == "ungroomed":
        return "minus2p5_to_zero", UNGROOMED_WINDOW
    return args.normalization_window, NORMALIZATION_WINDOWS[args.normalization_window]


def build_pairsplit_spec(
    channel: str,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    grooming_mode: str = "groomed",
):
    """Build a pair-split spec without registering a global analysis tag."""

    from unfold.tools.unfolder_core import RHO_FIXED_JEC_SPEC

    label = rf"$\log_{{10}}(\rho^2)$, {grooming_mode}"
    _, normalization_window = resolved_normalization(args, grooming_mode)
    normalization_updates = (
        {
            "normalization_window_groomed": normalization_window,
            "display_window_groomed": PLOTTED_DISPLAY_WINDOW,
        }
        if grooming_mode == "groomed"
        else {
            "normalization_window_ungroomed": normalization_window,
            "display_window_ungroomed": UNGROOMED_WINDOW,
        }
    )
    return replace(
        RHO_FIXED_JEC_SPEC,
        output_dir=_core_output_directory(output_dir),
        edges_reco_attr="two_log10_rho_reco_edges",
        edges_gen_attr="two_log10_rho_gen_edges",
        reco_edges_by_pt_attr="reco_two_log10_rho_edges_by_pt",
        gen_edges_by_pt_attr="gen_two_log10_rho_edges_by_pt",
        x_label_groomed=label,
        x_label_ungroomed=label,
        short_label_groomed=label,
        short_label_ungroomed=label,
        xlim_lower_groomed=-4.0,
        xlim_lower_ungroomed=-2.5,
        stat_propagation="jacobian",
        regularization=args.regularization,
        tau=args.tau,
        area_constraint=True,
        model_envelope=bool(args.model_envelope),
        model_envelope_source=(
            "prepared_systematics" if args.model_envelope else "zjet_offline"
        ),
        # The normalization denominator is variant-controlled.  The display
        # remains fixed so the peak-normalized study retains the low shoulder
        # and high catch-all as diagnostics, matching the earlier `_win18`
        # logic without changing the absolute unfolded result.
        normalize_over_shown=False,
        # Prescaled data vs MC have unrelated absolute normalizations: the
        # bottom-line residuals must be per-pT-slice shape comparisons.
        bottom_line_scale_mc_per_pt=True,
        **normalization_updates,
    )


def resolve_channel_binning(requested_binning: str, channel: str) -> str:
    """Resolve the explicit cross-channel study recommendation for one channel."""

    if channel not in PAIR_SPLIT_CHANNELS:
        raise ValueError(f"Unsupported pair-split channel {channel!r}")
    if requested_binning == STUDY_RECOMMENDED_BINNING:
        return STUDY_RECOMMENDED_VARIANT_BY_CHANNEL[channel]
    if requested_binning not in PAIR_SPLIT_BINNING_VARIANTS:
        raise ValueError(f"Unsupported pair-split binning request {requested_binning!r}")
    return requested_binning


def channel_resolved_args(args: argparse.Namespace, channel: str) -> argparse.Namespace:
    """Return arguments whose binning value is the concrete channel candidate."""

    resolved_binning = resolve_channel_binning(args.binning, channel)
    resolved = vars(args).copy()
    resolved["requested_binning"] = args.binning
    resolved["binning"] = resolved_binning
    return argparse.Namespace(**resolved)


def _flat_offsets(edges_by_pt: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return flat-spectrum start offsets and bin counts for each pT slice."""

    counts = np.asarray([len(edges) - 1 for edges in edges_by_pt], dtype=int)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(int)
    return starts, counts


def _normalization_mask(edges: Sequence[float], window: tuple[float, float] | None) -> np.ndarray:
    """Select complete coordinate bins inside the explicit pair-split window."""

    edges = np.asarray(edges, dtype=float)
    if window is None:
        return np.ones(edges.size - 1, dtype=bool)
    low, high = window
    return (edges[:-1] >= low - 1e-9) & (edges[1:] <= high + 1e-9)


def reported_normalization_mask(
    gen_edges_by_pt: Sequence[Sequence[float]],
    first_reported_pt_bin: int,
    window: tuple[float, float] | None,
) -> np.ndarray:
    """Flatten the bins reported after applying the full plotted range."""

    starts, counts = _flat_offsets(gen_edges_by_pt)
    mask = np.zeros(int(counts.sum()), dtype=bool)
    for i_pt, (start, count, edges) in enumerate(zip(starts, counts, gen_edges_by_pt)):
        if i_pt < first_reported_pt_bin:
            continue
        mask[start:start + count] = _normalization_mask(edges, window)
    return mask


def reco_to_gen_layout_map(
    reco_edges_by_pt: Sequence[Sequence[float]],
    gen_edges_by_pt: Sequence[Sequence[float]],
) -> np.ndarray:
    """Map a rectangular reco layout into the nested truth-bin layout.

    Each reco bin must be completely contained in exactly one generator bin in
    the same pT slice.  The matrix maps a flat reco spectrum to flat gen bins
    by summation and is therefore valid for both yields and response rows.
    """

    if len(reco_edges_by_pt) != len(gen_edges_by_pt):
        raise ValueError("Reco and gen layouts must have the same number of pT slices")
    reco_starts, reco_counts = _flat_offsets(reco_edges_by_pt)
    gen_starts, gen_counts = _flat_offsets(gen_edges_by_pt)
    mapping = np.zeros((int(gen_counts.sum()), int(reco_counts.sum())), dtype=float)
    for i_pt, (reco_edges, gen_edges) in enumerate(zip(reco_edges_by_pt, gen_edges_by_pt)):
        reco_edges = np.asarray(reco_edges, dtype=float)
        gen_edges = np.asarray(gen_edges, dtype=float)
        for reco_bin, (low, high) in enumerate(zip(reco_edges[:-1], reco_edges[1:])):
            matches = np.flatnonzero(
                (gen_edges[:-1] <= low + 1e-9) & (gen_edges[1:] >= high - 1e-9)
            )
            if matches.size != 1:
                raise ValueError(
                    "Reco bin cannot be mapped to one gen-layout bin: "
                    f"pT slice {i_pt}, [{low}, {high}]"
                )
            mapping[
                gen_starts[i_pt] + int(matches[0]),
                reco_starts[i_pt] + reco_bin,
            ] = 1.0
    return mapping


def _svd_metrics(matrix: np.ndarray, *, rcond: float = 1e-12) -> tuple[dict[str, object], np.ndarray]:
    """Return rectangular-matrix singular diagnostics without a pass/fail cut."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Response diagnostics require a finite two-dimensional matrix")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    full_rank = min(matrix.shape)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = rcond * largest
    rank = int(np.count_nonzero(singular_values > threshold))
    condition_is_infinite = rank < full_rank
    condition = (
        None
        if condition_is_infinite or rank == 0
        else float(singular_values[0] / singular_values[-1])
    )
    return {
        "shape": list(matrix.shape),
        "rcond": rcond,
        "rank": rank,
        "full_rank": full_rank,
        "condition_number": condition,
        "condition_is_infinite": condition_is_infinite,
    }, singular_values


def _covariance_metrics(covariance: np.ndarray, *, rcond: float = 1e-12) -> tuple[dict[str, object], np.ndarray]:
    """Summarize covariance symmetry and resolved eigenmodes."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance diagnostics require a square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance diagnostics require finite entries")
    asymmetry = float(np.max(np.abs(covariance - covariance.T))) if covariance.size else 0.0
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0, 1.0)
    threshold = rcond * scale
    return {
        "shape": list(covariance.shape),
        "rcond": rcond,
        "max_abs_asymmetry": asymmetry,
        "rank": int(np.count_nonzero(np.abs(eigenvalues) > threshold)),
        "positive_rank": int(np.count_nonzero(eigenvalues > threshold)),
        "negative_eigenvalue_count": int(np.count_nonzero(eigenvalues < -threshold)),
        "minimum_eigenvalue": float(eigenvalues[0]) if eigenvalues.size else None,
        "maximum_eigenvalue": float(eigenvalues[-1]) if eigenvalues.size else None,
    }, eigenvalues


def _full_covariance_chi2(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    rcond: float = 1e-12,
) -> dict[str, object]:
    """Use the positive resolved modes of a full covariance for a residual chi2."""

    residual = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (residual.size, residual.size):
        raise ValueError("Residual and covariance dimensions do not agree")
    covariance_metrics, eigenvalues = _covariance_metrics(covariance, rcond=rcond)
    symmetric = 0.5 * (covariance + covariance.T)
    _, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0, 1.0)
    keep = eigenvalues > rcond * scale
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        chi2 = None
        condition = None
    else:
        projected = eigenvectors[:, keep].T @ residual
        chi2 = float(np.sum(projected**2 / eigenvalues[keep]))
        condition = float(np.max(eigenvalues[keep]) / np.min(eigenvalues[keep]))
    return {
        "chi2": chi2,
        "rank": rank,
        "condition_number": condition,
        "covariance": covariance_metrics,
    }


def _fraction_metrics(values: np.ndarray) -> dict[str, object]:
    """Compact finite-fraction summary used for fakes, misses, purity, and stability."""

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"n_bins": int(values.size), "finite_bins": 0, "minimum": None, "maximum": None}
    return {
        "n_bins": int(values.size),
        "finite_bins": int(finite.size),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _unfolded_bin_metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    """Count finite and negative unfolded bins in one explicit reporting space."""

    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    finite = np.isfinite(selected)
    return {
        "n_bins": int(selected.size),
        "finite_bins": int(np.count_nonzero(finite)),
        "nonfinite_bins": int(np.count_nonzero(~finite)),
        "negative_bins": int(np.count_nonzero(selected[finite] < 0.0)),
    }


def _covariance_spaces(
    covariance: np.ndarray,
    reported_window_mask: np.ndarray,
    *,
    expected_null_modes: int,
) -> dict[str, object]:
    """Report full and all-shown normalized covariance modes separately."""

    reported_covariance = np.asarray(covariance, dtype=float)[
        np.ix_(reported_window_mask, reported_window_mask)
    ]
    full_metrics, _ = _covariance_metrics(covariance)
    reported_metrics, _ = _covariance_metrics(reported_covariance)
    reported_metrics["expected_normalization_null_modes"] = expected_null_modes
    reported_metrics["observed_null_modes"] = (
        int(reported_covariance.shape[0]) - int(reported_metrics["rank"])
    )
    return {"full": full_metrics, "reported_normalization_window": reported_metrics}


def _windowed_migration_summary(
    purity: np.ndarray,
    stability: np.ndarray,
    gen_edges_by_pt: Sequence[Sequence[float]],
    first_reported_pt_bin: int,
    window: tuple[float, float] | None,
) -> dict[str, object]:
    """Summarize migration quality in one explicit per-pT coordinate window."""

    starts, counts = _flat_offsets(gen_edges_by_pt)
    window_mask = reported_normalization_mask(
        gen_edges_by_pt, first_reported_pt_bin, window
    )
    per_pt = []
    for i_pt, (start, count, edges) in enumerate(zip(starts, counts, gen_edges_by_pt)):
        if i_pt < first_reported_pt_bin:
            continue
        local_mask = _normalization_mask(edges, window)
        local_slice = slice(start, start + count)
        per_pt.append(
            {
                "pt_bin": i_pt,
                "window_bin_count": int(np.count_nonzero(local_mask)),
                "purity": _fraction_metrics(purity[local_slice][local_mask]),
                "stability": _fraction_metrics(stability[local_slice][local_mask]),
            }
        )
    return {
        "purity": _fraction_metrics(purity[window_mask]),
        "stability": _fraction_metrics(stability[window_mask]),
        "per_pt": per_pt,
    }


def _bias_metrics(bias: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Report scale-free and absolute closure bias metrics without a threshold."""

    bias = np.asarray(bias, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if bias.size == 0:
        return {"n_bins": 0, "l1": 0.0, "l2": 0.0, "max_abs": 0.0, "relative_l1": None}
    reference_scale = float(np.sum(np.abs(reference)))
    return {
        "n_bins": int(bias.size),
        "l1": float(np.sum(np.abs(bias))),
        "l2": float(np.linalg.norm(bias)),
        "max_abs": float(np.max(np.abs(bias))),
        "relative_l1": (
            None if reference_scale == 0.0 else float(np.sum(np.abs(bias)) / reference_scale)
        ),
    }


def self_closure_diagnostics(
    closure_unfolder,
    *,
    normalization_window: tuple[float, float] | None,
    first_reported_pt_bin: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Quantify nominal-MC closure in raw and normalized reported coordinates."""

    truth = np.asarray(closure_unfolder.y_true, dtype=float)
    unfolded = np.asarray(closure_unfolder.y_unf, dtype=float)
    if truth.shape != unfolded.shape:
        raise ValueError("Closure truth and unfolded spectra must have matching shapes")
    edges_by_pt = closure_unfolder.gen_edges_by_pt
    starts, counts = _flat_offsets(edges_by_pt)
    window_mask = reported_normalization_mask(
        edges_by_pt, first_reported_pt_bin, normalization_window
    )
    normalized_truth = np.full_like(truth, np.nan)
    normalized_unfolded = np.full_like(unfolded, np.nan)
    truth_totals = np.full(len(edges_by_pt), np.nan)
    unfolded_totals = np.full(len(edges_by_pt), np.nan)
    per_pt = []
    for i_pt, (start, count, edges) in enumerate(zip(starts, counts, edges_by_pt)):
        local_slice = slice(start, start + count)
        local_mask = _normalization_mask(edges, normalization_window)
        if i_pt < first_reported_pt_bin:
            continue
        local_truth = truth[local_slice]
        local_unfolded = unfolded[local_slice]
        widths = np.diff(np.asarray(edges, dtype=float))
        truth_total = float(local_truth[local_mask].sum())
        unfolded_total = float(local_unfolded[local_mask].sum())
        truth_totals[i_pt] = truth_total
        unfolded_totals[i_pt] = unfolded_total
        normalized_truth_slice = normalized_truth[local_slice]
        normalized_unfolded_slice = normalized_unfolded[local_slice]
        if truth_total != 0.0:
            normalized_truth_slice[local_mask] = (
                local_truth[local_mask] / widths[local_mask] / truth_total
            )
        if unfolded_total != 0.0:
            normalized_unfolded_slice[local_mask] = (
                local_unfolded[local_mask] / widths[local_mask] / unfolded_total
            )
        normalized_bias = (
            normalized_unfolded_slice[local_mask] - normalized_truth_slice[local_mask]
        )
        per_pt.append(
            {
                "pt_bin": i_pt,
                "pt_range_GeV": [float(closure_unfolder.pt_edges[i_pt]), float(closure_unfolder.pt_edges[i_pt + 1])],
                "window_bin_count": int(np.count_nonzero(local_mask)),
                "truth_window_total": truth_total,
                "unfolded_window_total": unfolded_total,
                "bias": _bias_metrics(
                    normalized_bias,
                    normalized_truth_slice[local_mask],
                ),
            }
        )
    raw_bias = unfolded - truth
    return {
        "tau": float(closure_unfolder.tau or 0.0),
        "raw_bias": _bias_metrics(raw_bias, truth),
        "per_pt_window_normalized_bias": per_pt,
    }, {
        "closure_truth": truth,
        "closure_unfolded": unfolded,
        "closure_raw_bias": raw_bias,
        "closure_window_mask": window_mask,
        "closure_window_normalized_truth": normalized_truth,
        "closure_window_normalized_unfolded": normalized_unfolded,
        "closure_window_normalized_bias": normalized_unfolded - normalized_truth,
        "closure_truth_window_totals": truth_totals,
        "closure_unfolded_window_totals": unfolded_totals,
    }


def collect_run_diagnostics(
    unfolder,
    closure_unfolder,
    *,
    normalization_window: tuple[float, float] | None,
    study_core_window: tuple[float, float] | None,
    first_reported_pt_bin: int,
    data_covariance_source: str,
) -> RunDiagnostics:
    """Collect audit arrays and metrics from the nominal prepared TUnfold result."""

    response_raw = np.asarray(unfolder.mosaic, dtype=float)
    matched_reco = response_raw.sum(axis=1)
    matched_gen = response_raw.sum(axis=0)
    column_normalized = np.divide(
        response_raw,
        matched_gen[None, :],
        out=np.zeros_like(response_raw),
        where=matched_gen[None, :] != 0.0,
    )
    raw_svd, raw_singular_values = _svd_metrics(response_raw)
    normalized_svd, normalized_singular_values = _svd_metrics(column_normalized)
    layout_map = reco_to_gen_layout_map(unfolder.reco_edges_by_pt, unfolder.gen_edges_by_pt)
    migration = layout_map @ response_raw
    migration_diagonal = np.diag(migration)
    migration_reco = migration.sum(axis=1)
    migration_gen = migration.sum(axis=0)
    purity = np.divide(
        migration_diagonal,
        migration_reco,
        out=np.zeros_like(migration_diagonal),
        where=migration_reco != 0.0,
    )
    stability = np.divide(
        migration_diagonal,
        migration_gen,
        out=np.zeros_like(migration_diagonal),
        where=migration_gen != 0.0,
    )
    fakes = np.asarray(unfolder.fakes_2d, dtype=float)
    misses = np.asarray(unfolder.misses_2d, dtype=float)
    fake_fraction = np.divide(
        fakes,
        matched_reco + fakes,
        out=np.zeros_like(fakes),
        where=(matched_reco + fakes) != 0.0,
    )
    miss_fraction = np.divide(
        misses,
        matched_gen + misses,
        out=np.zeros_like(misses),
        where=(matched_gen + misses) != 0.0,
    )
    measured = np.asarray(unfolder.y_meas, dtype=float)
    measured_covariance = getattr(unfolder, "corrected_measured_covariance", None)
    if measured_covariance is None:
        measured_variances = np.asarray(unfolder.corrected_measured_variances, dtype=float)
        measured_covariance = np.diag(measured_variances)
    else:
        measured_covariance = np.asarray(measured_covariance, dtype=float)
    refolded = np.asarray(unfolder.x_folded, dtype=float)
    refolded_residual = measured - refolded
    refolded_chi2 = _full_covariance_chi2(refolded_residual, measured_covariance)
    output_covariance = np.asarray(unfolder.cov_np, dtype=float)
    output_covariance_metrics, output_covariance_eigenvalues = _covariance_metrics(output_covariance)
    unfolded = np.asarray(unfolder.y_unf, dtype=float)
    reported_window_mask = reported_normalization_mask(
        unfolder.gen_edges_by_pt, first_reported_pt_bin, normalization_window
    )
    normalized_result = np.concatenate(
        [np.asarray(result["unfolded"], dtype=float) for result in unfolder.normalized_results]
    )
    normalized_covariances = {
        "input": np.asarray(unfolder.norm_cov_input, dtype=float),
        "matrix": np.asarray(unfolder.norm_cov_matrix, dtype=float),
        "stat": np.asarray(unfolder.norm_cov_stat, dtype=float),
        "total": np.asarray(unfolder.get_total_covariance(), dtype=float),
    }
    expected_null_modes = int(sum(
        bool(
            i_pt >= first_reported_pt_bin
            and np.any(_normalization_mask(edges, normalization_window))
        )
        for i_pt, edges in enumerate(unfolder.gen_edges_by_pt)
    ))
    closure_metrics, closure_arrays = self_closure_diagnostics(
        closure_unfolder,
        normalization_window=normalization_window,
        first_reported_pt_bin=first_reported_pt_bin,
    )
    arrays = {
        "measured_corrected_spectrum": measured,
        "measured_corrected_covariance": measured_covariance,
        "refolded_spectrum": refolded,
        "refolded_residual": refolded_residual,
        "response_raw": response_raw,
        "response_column_normalized": column_normalized,
        "response_raw_singular_values": raw_singular_values,
        "response_column_normalized_singular_values": normalized_singular_values,
        "reco_to_gen_layout_map": layout_map,
        "gen_layout_migration": migration,
        "gen_layout_purity": purity,
        "gen_layout_stability": stability,
        "fake_fraction": fake_fraction,
        "miss_fraction": miss_fraction,
        "output_covariance_eigenvalues": output_covariance_eigenvalues,
        "normalized_result": normalized_result,
        "norm_cov_input": normalized_covariances["input"],
        "norm_cov_matrix": normalized_covariances["matrix"],
        "norm_cov_stat": normalized_covariances["stat"],
        "norm_cov_total": normalized_covariances["total"],
        "unfolded_full_mask": np.ones(unfolded.size, dtype=bool),
        "unfolded_reported_window_mask": reported_window_mask,
        **closure_arrays,
    }
    metrics = {
        "coordinate": {
            "rho_definition": PHYSICAL_RHO_DEFINITION,
            "transformed_coordinate_definition": TRANSFORMED_COORDINATE_DEFINITION,
        },
        "measured_corrected_covariance": {
            "source": data_covariance_source,
            **_covariance_metrics(measured_covariance)[0],
        },
        "refolded_residual": refolded_chi2,
        "response_raw": raw_svd,
        "response_column_normalized": normalized_svd,
        "gen_layout_migration": {
            "shape": list(migration.shape),
            "all_bins": {
                "purity": _fraction_metrics(purity),
                "stability": _fraction_metrics(stability),
            },
            "reported_normalization_window": _windowed_migration_summary(
                purity,
                stability,
                unfolder.gen_edges_by_pt,
                first_reported_pt_bin,
                normalization_window,
            ),
            "study_core_window": _windowed_migration_summary(
                purity,
                stability,
                unfolder.gen_edges_by_pt,
                first_reported_pt_bin,
                study_core_window,
            ),
        },
        "fake_fraction": _fraction_metrics(fake_fraction),
        "miss_fraction": _fraction_metrics(miss_fraction),
        "output_covariance": output_covariance_metrics,
        "normalized_covariances": {
            name: _covariance_spaces(
                covariance,
                reported_window_mask,
                expected_null_modes=expected_null_modes,
            )
            for name, covariance in normalized_covariances.items()
        },
        "unfolded_bins": {
            "full": _unfolded_bin_metrics(unfolded, np.ones(unfolded.size, dtype=bool)),
            "reported_normalization_window": _unfolded_bin_metrics(
                unfolded, reported_window_mask
            ),
        },
        "nominal_mc_self_closure": closure_metrics,
    }
    return RunDiagnostics(metrics=_json_native(metrics), arrays=arrays)


def run_nominal_mc_self_closure(
    unfolder_class,
    data_unfolder,
    *,
    spec,
    prepared: PairSplitPreparedInputs,
    cms_label: str,
    lumi: float,
    com: float,
    grooming_mode: str = "groomed",
):
    """Unfold nominal MC reco through the identical prepared-input TUnfold path."""

    if getattr(spec, "regularization", "none") == "none":
        closure_tau = None
    else:
        if data_unfolder.tau is None:
            raise RuntimeError("The data-selected tau is unavailable for MC self-closure")
        closure_tau = float(data_unfolder.tau)
    # Self-closure is a nominal-response validation, not a second evaluation of
    # the data model envelope.  Its input list deliberately contains nominal
    # only, so disable the envelope for this isolated unfold.
    closure_updates = {"tau": closure_tau}
    if hasattr(spec, "model_envelope"):
        closure_updates["model_envelope"] = False
    if hasattr(spec, "model_envelope_source"):
        closure_updates["model_envelope_source"] = "zjet_offline"
    # Closure figures land in a dedicated subtree: plot_unfolded_fancy also
    # draws mode summaries, which would otherwise overwrite the data results.
    if hasattr(spec, "output_dir"):
        closure_updates["output_dir"] = spec.output_dir + "closure/"
    closure_spec = replace(spec, **closure_updates)
    closure_data_inputs = dict(prepared.data_inputs)
    reco_key = LEGACY_HISTOGRAM_KEYS[grooming_mode]["reco"]
    closure_data_inputs[reco_key] = prepared.mc_inputs[reco_key]
    closure_reco_variances = np.asarray(
        data_unfolder.reco_mc_var_dict["nominal"], dtype=float
    )
    return unfolder_class.from_prepared_inputs(
        closure_spec,
        grooming_mode == "groomed",
        mc_inputs=prepared.mc_inputs,
        data_inputs=closure_data_inputs,
        analysis_binning=prepared.analysis_binning,
        systematics=("nominal",),
        measured_covariance=np.diag(closure_reco_variances),
        first_reported_pt_bin=prepared.first_reported_pt_bin,
        cms_label=cms_label,
        lumi=lumi,
        com=com,
    )


def write_artifact(
    unfolder,
    prepared: PairSplitPreparedInputs,
    output_dir: Path,
    diagnostics: RunDiagnostics,
    vincia_prediction,
    model_envelope: PairSplitModelEnvelopeInputs | None = None,
) -> Path:
    """Save reproducible inputs, result arrays, and numerical audit diagnostics."""

    grooming_mode = str(prepared.metadata["grooming_mode"])
    path = output_dir / "artifacts" / f"{grooming_mode}_results.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    non_nominal = [name for name in prepared.systematics if name != "nominal"]
    artifact_arrays = {
        "pt_edges": np.asarray(prepared.analysis_binning.pt_edges, dtype=float),
        "two_log10_rho_reco_edges": np.asarray(
            prepared.analysis_binning.two_log10_rho_reco_edges, dtype=float
        ),
        "two_log10_rho_gen_edges": np.asarray(
            prepared.analysis_binning.two_log10_rho_gen_edges, dtype=float
        ),
        "measured_covariance": prepared.measured_covariance,
        "response_mosaic": unfolder.mosaic,
        "response_mosaic_variance": unfolder.mosaic_var_dict["nominal"],
        "unfolded": unfolder.y_unf,
        "covariance": unfolder.cov_np,
        "input_covariance": unfolder.cov_data_np,
        "systematic_names": np.asarray(non_nominal),
        "systematic_unfolded": np.asarray(
            [unfolder.y_unf_dict[name] for name in non_nominal], dtype=float
        ),
        "systematic_normalized": np.asarray(
            [
                np.concatenate(
                    [
                        np.asarray(per_pt["unfolded"][name], dtype=float)
                        for per_pt in unfolder.normalized_systematics
                    ]
                )
                for name in non_nominal
            ],
            dtype=float,
        ),
        "tau": np.asarray(float(unfolder.tau or 0.0)),
        # The plotting path consumes these exact arrays.  Keeping them in the
        # artifact makes later data/MESS comparisons independent of legend text
        # or a PDF reparse.
        **vincia_prediction.artifact_arrays(),
        **(model_envelope.artifact_arrays() if model_envelope is not None else {}),
        **diagnostics.arrays,
    }
    if getattr(getattr(unfolder, "spec", None), "model_envelope", False):
        artifact_arrays.update(
            {
                "model_ps_fraction": np.concatenate(
                    [unfolder.model_ps_frac[i] for i in range(len(unfolder.gen_edges_by_pt))]
                ),
                "model_had_fraction": np.concatenate(
                    [unfolder.model_had_frac[i] for i in range(len(unfolder.gen_edges_by_pt))]
                ),
                "model_total_fraction": np.concatenate(
                    [
                        unfolder.normalized_results[i]["model_unc_frac"]
                        for i in range(len(unfolder.gen_edges_by_pt))
                    ]
                ),
                "model_ps_selected_signed_fraction": np.asarray(
                    unfolder.model_ps_shift_flat, dtype=float
                ),
                "model_had_selected_signed_fraction": np.asarray(
                    unfolder.model_had_shift_flat, dtype=float
                ),
            }
        )
    np.savez_compressed(path, **artifact_arrays)
    return path


def build_manifest(
    *,
    args: argparse.Namespace,
    channel: str,
    prepared: PairSplitPreparedInputs,
    source_records: Sequence[dict],
    artifact: Path,
    resolved_tau: float,
    root_version: str,
    run_identity: dict[str, object],
    diagnostics: RunDiagnostics,
    vincia_prediction,
    vincia_reference_validation: Mapping[str, object],
    model_envelope: PairSplitModelEnvelopeInputs | None = None,
    model_selection: Mapping[str, object] | None = None,
) -> dict:
    """Record terminology, exact group maps, and the current uncertainty scope."""

    return {
        "workflow": f"Run-2 {prepared.metadata['grooming_mode']} pair-split TUnfold",
        "channel": channel,
        "grooming_mode": prepared.metadata["grooming_mode"],
        "run_identity": run_identity,
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        # The raw command can mislead for ungroomed runs: a --normalization-window
        # value in argv applies to the groomed leg only (a 'both' invocation
        # records the same command in both manifests).
        "command_note": (
            "--normalization-window applies to the groomed leg only; ungroomed "
            "always resolves to minus2p5_to_zero ([-2.5, 0]). "
            "unfolding.plot_normalization records the authoritative window."
        ),
        "observable": {
            "rho_definition": PHYSICAL_RHO_DEFINITION,
            "transformed_coordinate_name": TRANSFORMED_COORDINATE_NAME,
            "transformed_coordinate_definition": TRANSFORMED_COORDINATE_DEFINITION,
            "jet_radius": 0.8,
        },
        "unfolding": {
            "backend": "TUnfoldDensity",
            "stat_propagation": "jacobian",
            "response_statistics": "TUnfold built-in GetEmatrixSysUncorr from weighted sumw2",
            "area_constraint": True,
            "luminosity_fb": float(args.lumi),
            "center_of_mass_energy_TeV": float(args.com),
            "regularization": args.regularization,
            "tau": {"requested": args.tau, "resolved": resolved_tau},
            "jackknife": "not used",
            "iterative_bayes": "not used",
            "plot_normalization": {
                "mode": "unit_area_over_coordinate_window",
                "name": resolved_normalization(
                    args, str(prepared.metadata["grooming_mode"])
                )[0],
                "two_log10_rho_range": list(
                    resolved_normalization(
                        args, str(prepared.metadata["grooming_mode"])
                    )[1]
                ),
            },
            "plot_display": {
                "two_log10_rho_range": list(
                    PLOTTED_DISPLAY_WINDOW
                    if prepared.metadata["grooming_mode"] == "groomed"
                    else UNGROOMED_WINDOW
                ),
            },
            "study_core_window_two_log10_rho": CORE_STABILITY_WINDOWS[channel],
            "study_core_window_purpose": (
                "separate stability diagnostic only; not the plotted or "
                "reported normalization range"
            ),
            "data_covariance_source": prepared.metadata["data_covariance_source"],
        },
        "binning": {
            "requested_candidate": getattr(args, "requested_binning", args.binning),
            "resolved_candidate": prepared.candidate.variant,
            "candidate": prepared.candidate.variant,
            "pt_edges_GeV": prepared.analysis_binning.pt_edges,
            "base_reco_two_log10_rho_edges": prepared.candidate.base_reco_two_log10_rho_edges,
            "gen_two_log10_rho_edges": prepared.candidate.gen_two_log10_rho_edges,
            "pt_groups": prepared.candidate.pt_groups,
            "source_sink_pt_bin_indices": prepared.candidate.sink_pt_source_bin_indices,
            "first_reported_pt_bin": prepared.first_reported_pt_bin,
            "reported_two_log10_rho_minimum": prepared.candidate.reported_two_log10_rho_minimum,
            "base_to_gen_two_log10_rho_groups": prepared.candidate.base_to_gen_two_log10_rho_groups,
            "fine_to_base_reco_two_log10_rho_groups": prepared.metadata[
                "reco_two_log10_rho_groups"
            ],
            "fine_to_base_gen_two_log10_rho_groups": prepared.metadata[
                "gen_base_two_log10_rho_groups"
            ],
        },
        "systematics": {
            "requested": args.systematics,
            "resolved": prepared.systematics,
            "included": list(prepared.systematics),
            "excluded": [
                *prepared.metadata["run2_era_correlation"].get(
                    "excluded_defective_jes_sources", []
                ),
                *prepared.metadata["systematic_exclusions"],
                "jackknife replicas",
                "iterative Bayes",
            ],
            "excluded_details": prepared.metadata["systematic_exclusions"],
            "systematic_variance_policy": prepared.metadata[
                "systematic_variance_policy"
            ],
            "run2_era_correlation": prepared.metadata["run2_era_correlation"],
        },
        "model_uncertainty": {
            "enabled": bool(args.model_envelope),
            "prescription": (
                "PS=max(MESS+Vincia,FSR); "
                "HAD=max(CR1,CR2,frag-hard,frag-soft); "
                "model=sqrt(PS^2+HAD^2)"
                if args.model_envelope
                else None
            ),
            "raw_isr_in_band": False,
            "selected_covariance_sources": (
                dict(model_selection) if model_selection is not None else None
            ),
            "prepared_inputs": (
                model_envelope.provenance_payload()
                if model_envelope is not None
                else None
            ),
        },
        "inputs": list(source_records),
        "mess_vincia_prediction": {
            "derived_curve": vincia_prediction.provenance_payload(),
            "compiled_high_pt_regression": dict(vincia_reference_validation),
        },
        "artifact": str(artifact.resolve()),
        "diagnostics": diagnostics.metrics,
        "root_version": root_version,
    }


def run_focused_plots(unfolder, output_dir: Path, closure_unfolder=None) -> Path | None:
    """Generate the final prepared-input figures through maintained core APIs."""

    # The core's fancy panel owns the normalized band and uses bin-edge stairs
    # rather than a center-based step fill.  The prepared counts fold view
    # keeps the fake correction and full measured covariance visible.
    unfolder.plot_unfolded_fancy(show=False)
    unfolder.plot_folded(show=False, counts=True)
    unfolder.plot_purity_stability(show=False)
    unfolder.plot_systematic_fraction_grouped(show=False, log=False)
    if getattr(getattr(unfolder, "spec", None), "model_envelope", False):
        unfolder.plot_model_envelope(show=False)
    unfolder.plot_correlation(show=False, shown_only=True, covariance="total")
    # Z+jet-parity validation checks (2026-08-18): the applicable subset of the
    # core run_all_plots suite.  Not applicable here: HERWIG overlays/bias,
    # jackknife convergence, lepton-SF groups, and the L-curve (tau = 0).
    unfolder.plot_statistical_fraction(show=False)
    unfolder.plot_fakes_misses(show=False)
    unfolder.plot_response_matrix(probability=True, show=False)
    unfolder.plot_uncertainty_heatmap(show=False)
    unfolder.plot_correlation(show=False, shown_only=True)  # stat-only companion
    unfolder.plot_bottom_line(show=False)
    unfolder.plot_bottom_line(show=False, rebin_reco_to_gen=True,
                              annotate_chi2=False)
    unfolder.plot_bottom_line_chi2_summary(show=False, normalized=True,
                                           min_edge="shown")
    unfolder.plot_bottom_line_chi2_summary(show=False, min_edge="shown")
    if closure_unfolder is not None:
        # Self-closure panels (stat-only band; saved under <run>/closure/).
        closure_unfolder.closure = True
        closure_unfolder.plot_unfolded_fancy(show=False)
    gallery_builder = REPO_ROOT / "outputs" / "build_rho_gallery.py"
    if not gallery_builder.is_file() or shutil.which("pdftoppm") is None:
        return None
    gallery_path = output_dir / "index.html"
    subprocess.run(
        [sys.executable, str(gallery_builder), "--root", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    return gallery_path


def run_channel(
    args: argparse.Namespace, channel: str, *, grooming_mode: str = "groomed"
) -> Path:
    """Load and unfold one channel, keeping only one channel's arrays resident."""

    import ROOT

    from unfold.tools.unfolder_core import Unfolder

    args = channel_resolved_args(args, channel)
    inputs = load_pairsplit_run2_inputs(channel, input_root=args.input_root)
    resolved_systematics = resolve_pairsplit_systematics(
        inputs.modes[grooming_mode].systematics,
        args.systematics,
    )
    _, normalization_window = resolved_normalization(args, grooming_mode)
    # Pair-split must consume the audited hadronic final-all rows; never let
    # the generic core fall through to its unrelated Z+jet Vincia cache.
    vincia_source = load_pairsplit_vincia_source(channel)
    model_envelope = (
        derive_pairsplit_model_envelope_inputs(
            inputs,
            vincia_source,
            variant=args.binning,
            normalization_window=normalization_window,
            grooming_mode=grooming_mode,
        )
        if args.model_envelope
        else None
    )
    prepared = prepare_pairsplit_inputs(
        inputs,
        args.binning,
        resolved_systematics,
        grooming_mode=grooming_mode,
        model_variations=(
            model_envelope.prepared_variations()
            if model_envelope is not None
            else None
        ),
        model_metadata=(
            model_envelope.provenance_payload()
            if model_envelope is not None
            else None
        ),
    )
    vincia_prediction = derive_pairsplit_vincia_prediction(
        vincia_source,
        pt_edges=prepared.analysis_binning.pt_edges,
        gen_edges_by_pt=prepared.analysis_binning.gen_two_log10_rho_edges_by_pt,
        normalization_window=normalization_window,
        grooming_mode=grooming_mode,
    )
    vincia_reference_validation = validate_compiled_reference(
        vincia_source,
        target_prediction=vincia_prediction,
    )
    run_identity = run_configuration_identity(
        args,
        resolved_systematics,
        vincia_source.identity_payload(),
        model_envelope.identity_payload() if model_envelope is not None else None,
        grooming_mode=grooming_mode,
    )
    output_parent = args.output_root / channel / args.binning
    if grooming_mode == "ungroomed":
        output_parent = output_parent / "ungroomed"
    output_dir = (output_parent / str(run_identity["directory_name"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = build_pairsplit_spec(
        channel, output_dir, args, grooming_mode=grooming_mode
    )
    ROOT.gErrorIgnoreLevel = ROOT.kError
    unfolder = Unfolder.from_prepared_inputs(
        spec,
        grooming_mode == "groomed",
        mc_inputs=prepared.mc_inputs,
        data_inputs=prepared.data_inputs,
        analysis_binning=prepared.analysis_binning,
        systematics=prepared.systematics,
        measured_covariance=prepared.measured_covariance,
        first_reported_pt_bin=prepared.first_reported_pt_bin,
        cms_label=args.cms_label,
        lumi=args.lumi,
        com=args.com,
    )
    attach_pairsplit_vincia_prediction(unfolder, vincia_prediction)
    closure_unfolder = run_nominal_mc_self_closure(
        Unfolder,
        unfolder,
        spec=spec,
        prepared=prepared,
        cms_label=args.cms_label,
        lumi=args.lumi,
        com=args.com,
        grooming_mode=grooming_mode,
    )
    diagnostics = collect_run_diagnostics(
        unfolder,
        closure_unfolder,
        normalization_window=normalization_window,
        study_core_window=(
            CORE_STABILITY_WINDOWS[channel]
            if grooming_mode == "groomed"
            else UNGROOMED_WINDOW
        ),
        first_reported_pt_bin=prepared.first_reported_pt_bin,
        data_covariance_source=str(prepared.metadata["data_covariance_source"]),
    )
    artifact = write_artifact(
        unfolder,
        prepared,
        output_dir,
        diagnostics,
        vincia_prediction,
        model_envelope,
    )
    gallery = None if args.no_plots else run_focused_plots(
        unfolder, output_dir, closure_unfolder=closure_unfolder
    )
    manifest = build_manifest(
        args=args,
        channel=channel,
        prepared=prepared,
        source_records=source_file_metadata(inputs.source_files),
        artifact=artifact,
        resolved_tau=float(unfolder.tau or 0.0),
        root_version=ROOT.gROOT.GetVersion(),
        run_identity=run_identity,
        diagnostics=diagnostics,
        vincia_prediction=vincia_prediction,
        vincia_reference_validation=vincia_reference_validation,
        model_envelope=model_envelope,
        model_selection=(
            {
                "scope": getattr(spec, "model_covariance_scope", "global_shown"),
                "global": {
                    "parton_shower": unfolder.model_ps_source,
                    "hadronization": unfolder.model_had_source,
                },
                "by_pt_slice": {
                    str(i): {
                        "parton_shower": unfolder.model_ps_sources_by_pt[i],
                        "hadronization": unfolder.model_had_sources_by_pt[i],
                    }
                    for i in unfolder.model_ps_sources_by_pt
                },
            }
            if args.model_envelope
            else None
        ),
    )
    manifest["plots"] = {"enabled": not args.no_plots, "gallery": str(gallery) if gallery else None}
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    grooming_modes = (
        ("groomed", "ungroomed")
        if args.grooming_mode == "both"
        else (args.grooming_mode,)
    )
    for channel in args.channel:
        resolved_binning = resolve_channel_binning(args.binning, channel)
        for grooming_mode in grooming_modes:
            print(
                f"Running {grooming_mode} pair-split {channel} with {resolved_binning} "
                f"(requested {args.binning})"
            )
            print(
                f"manifest: {run_channel(args, channel, grooming_mode=grooming_mode)}"
            )


if __name__ == "__main__":
    main()
