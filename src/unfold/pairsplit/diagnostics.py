"""Numerical audit of one pair-split run (metrics for the manifest, arrays for the artifact).

Everything here reads a finished ``Unfolder``; nothing changes its state.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from unfold.pairsplit.inputs import PHYSICAL_RHO_DEFINITION, TRANSFORMED_COORDINATE_DEFINITION


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


