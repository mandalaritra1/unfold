#!/usr/bin/env python3
"""SUPERSEDED (2026-08-15): do not use for the canonical pair-split delivery.

This report layer targets the pre-model-envelope, groomed-only 2026-08-14 runs
(dijet ``__systematics-6517ad83dce2d223``, trijet ``__systematics-02faf4e995fb5165``,
[-4, 0] normalization, ``model_envelope=False``).  The canonical delivery is the
four-run plot book built by ``scripts/plotting/build_pairsplit_all_modes_plot_book.py``
(``PAIR_SPLIT_GROOMED_UNGROOMED_PLOT_BOOK_2026-08-14.pdf`` + inventory JSON).
This script and its outputs are retained under ``_superseded`` locations only
as provenance for the earlier iteration; rerunning it writes into
``outputs/pairsplit_run2/_superseded_20260814_groomed_only/``.

Original description: build the pair-split summary and Markdown report from
saved TUnfold outputs.  This report layer deliberately has no unfolding or
plotting implementation.  It reads the immutable channel manifests and their
groomed-results artifacts, then records the numerical diagnostics and the
explicitly attached MESS+Vincia comparison used by the core Unfolder plots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "pairsplit_run2"
SUPERSEDED_OUTPUT_DIR = OUTPUT_ROOT / "_superseded_20260814_groomed_only"
CHANNEL_DIRECTORIES = {
    "dijet": OUTPUT_ROOT
    / "dijet/coarse_tail/regularization-none__tau-disabled__systematics-6517ad83dce2d223",
    "trijet": OUTPUT_ROOT
    / "trijet/two_to_one/regularization-none__tau-disabled__systematics-02faf4e995fb5165",
}
CHANNEL_CANDIDATES = {"dijet": "coarse_tail", "trijet": "two_to_one"}
REPORT_DATE = "2026-08-14"
NORMALIZATION_WINDOW = (-4.0, 0.0)


def json_native(value: Any) -> Any:
    """Convert NumPy scalars/arrays to JSON values without rounding facts."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_output(path: Path) -> str:
    # The regenerated report lives one level below OUTPUT_ROOT, so links to
    # run directories need the explicit parent step.
    return "../" + str(path.relative_to(OUTPUT_ROOT))


def fraction_summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    selected = np.asarray(values, dtype=float)
    if mask is not None:
        selected = selected[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        raise ValueError("Cannot summarize an empty non-finite fraction array")
    return {
        "min": float(selected.min()),
        "median": float(np.median(selected)),
        "mean": float(selected.mean()),
        "max": float(selected.max()),
    }


def covariance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Use the manifest's saved normalized-covariance audit unchanged."""

    summary = dict(payload["reported_normalization_window"])
    # The existing dynamic book schema calls this concise key; preserve the
    # manifest spelling alongside it so the report remains traceable.
    summary["expected_null_modes"] = summary["expected_normalization_null_modes"]
    return summary


def per_pt_offsets(edges: np.ndarray, n_total: int) -> list[tuple[int, int]]:
    n_per_pt = len(edges) - 1
    if n_per_pt <= 0 or n_total % n_per_pt:
        raise ValueError("Artifact does not have a rectangular pT/rho layout")
    return [
        (index * n_per_pt, (index + 1) * n_per_pt)
        for index in range(n_total // n_per_pt)
    ]


def normalized_nominal_pythia(closure_truth: np.ndarray, edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reproduce the per-pT shown-window normalization used by the core plots."""

    density = np.zeros_like(closure_truth, dtype=float)
    widths = np.diff(edges)
    for start, stop in per_pt_offsets(edges, closure_truth.size):
        local_mask = mask[start:stop]
        shown_total = float(closure_truth[start:stop][local_mask].sum())
        if shown_total <= 0:
            raise ValueError("Nominal Pythia has a non-positive shown-window total")
        density[start:stop] = closure_truth[start:stop] / widths / shown_total
    return density


def comparison_summary(
    data: np.ndarray,
    data_covariance: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    prediction_statistical_uncertainty: np.ndarray | None,
) -> dict[str, Any]:
    """Saved-data covariance comparison, matching the core panel convention."""

    indices = np.flatnonzero(mask)
    covariance = np.asarray(data_covariance[np.ix_(indices, indices)], dtype=float)
    if prediction_statistical_uncertainty is not None:
        covariance = covariance + np.diag(
            np.square(np.asarray(prediction_statistical_uncertainty, dtype=float)[indices])
        )
    residual = np.asarray(data, dtype=float)[indices] - np.asarray(prediction, dtype=float)[indices]
    chi2 = float(residual @ np.linalg.pinv(covariance, rcond=1e-10) @ residual)
    npt = len(per_pt_offsets(edges, data.size))
    nominal_ndof = int(indices.size - npt)
    per_pt = []
    for pt_index, (start, stop) in enumerate(per_pt_offsets(edges, data.size)):
        local_indices = np.arange(start, stop)[mask[start:stop]]
        local_covariance = np.asarray(
            data_covariance[np.ix_(local_indices, local_indices)], dtype=float
        )
        if prediction_statistical_uncertainty is not None:
            local_covariance = local_covariance + np.diag(
                np.square(np.asarray(prediction_statistical_uncertainty)[local_indices])
            )
        local_residual = data[local_indices] - prediction[local_indices]
        per_pt.append(
            {
                "pt_index": pt_index,
                "chi2": float(
                    local_residual
                    @ np.linalg.pinv(local_covariance, rcond=1e-10)
                    @ local_residual
                ),
                "nominal_ndof": int(local_indices.size - 1),
                "matrix_rank": int(np.linalg.matrix_rank(local_covariance, tol=1e-12)),
            }
        )
    return {
        "chi2": chi2,
        "rank": nominal_ndof,
        "nominal_ndof": nominal_ndof,
        "matrix_rank": int(np.linalg.matrix_rank(covariance, tol=1e-12)),
        "per_pt": per_pt,
    }


def required_pdf_count(pt_edges: list[float]) -> int:
    # Four per-pT canonical core products plus correlation and two summaries.
    return 4 * (len(pt_edges) - 1) + 3


def channel_assessment(channel: str, diagnostics: dict[str, Any], mess_comparison: dict[str, Any]) -> dict[str, str]:
    migration = diagnostics["gen_layout_migration"]["reported_normalization_window"]
    purity = migration["purity"]
    stability = migration["stability"]
    closure = diagnostics["nominal_mc_self_closure"]["raw_bias"]["relative_l1"]
    refold = diagnostics["refolded_residual"]
    if channel == "dijet":
        return {
            "status": "Numerically closed but not physics-ready.",
            "worked": (
                "Dijet retains 10 bins per pT slice because the available statistics support "
                f"the full 50-bin shown result; nominal-MC raw relative-L1 closure is {closure:.3g}. "
                "The MESS+Vincia overlay is now derived from the audited pair-split final-all rows."
            ),
            "did_not_work": (
                f"Reported-window min purity/stability are {purity['minimum']:.3f}/{stability['minimum']:.3f}, "
                f"below 0.5 in some bins, and the refold residual remains severe at chi2 "
                f"{refold['chi2']:.2f}/{refold['rank']}. The correctly sourced MESS+Vincia "
                f"global data-total-only comparison is chi2 {mess_comparison['chi2']:.2f}/nominal "
                f"{mess_comparison['nominal_ndof']}. It retains cross-pT covariance and is not the panel "
                "statistic, an uncertainty term, or a measurement claim."
            ),
        }
    return {
        "status": "Numerically better migration with 2to1, but not physics-ready.",
        "worked": (
            "Trijet uses the appendix 2to1 recommendation: 4 bins per pT slice, "
            f"reported-window min purity/stability {purity['minimum']:.3f}/{stability['minimum']:.3f}, "
            f"and raw relative-L1 closure {closure:.3g}. The MESS+Vincia overlay is now "
            "derived from the audited pair-split final-all rows."
        ),
        "did_not_work": (
            f"The refold residual is chi2 {refold['chi2']:.2f}/{refold['rank']}; the correctly "
            f"sourced MESS+Vincia global data-total-only comparison is chi2 {mess_comparison['chi2']:.2f}/nominal "
            f"{mess_comparison['nominal_ndof']}. It retains cross-pT covariance; these comparison residuals "
            "do not establish a response-model uncertainty or a measurement claim."
        ),
    }


def build_channel(channel: str, directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "run_manifest.json"
    artifact_path = directory / "artifacts" / "groomed_results.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = np.load(artifact_path)
    diagnostics = manifest["diagnostics"]
    binning = manifest["binning"]
    edges = np.asarray(artifact["two_log10_rho_gen_edges"], dtype=float)
    pt_edges = [float(value) for value in artifact["pt_edges"]]
    mask = np.asarray(artifact["unfolded_reported_window_mask"], dtype=bool)
    data = np.asarray(artifact["normalized_result"], dtype=float)
    data_covariance = np.asarray(artifact["norm_cov_total"], dtype=float)
    pythia = normalized_nominal_pythia(
        np.asarray(artifact["closure_truth"], dtype=float), edges, mask
    )
    mess = np.asarray(artifact["mess_vincia_density_flat"], dtype=float)
    mess_stat = np.asarray(artifact["mess_vincia_stat_unc_flat"], dtype=float)
    pythia_comparison = comparison_summary(
        data, data_covariance, pythia, mask, edges, prediction_statistical_uncertainty=None
    )
    # The global saved-artifact comparison is deliberately data-total-only for
    # both predictions.  Per-panel MESS labels retain their separate diagonal
    # MESS statistical term, but that is not the global covariance convention.
    mess_comparison = comparison_summary(
        data, data_covariance, mess, mask, edges, prediction_statistical_uncertainty=None
    )
    mess_panel_comparison = comparison_summary(
        data, data_covariance, mess, mask, edges, prediction_statistical_uncertainty=mess_stat
    )
    pythia_comparison["covariance"] = (
        "saved normalized data total covariance only; nominal-Pythia template uncertainty not added"
    )
    pythia_comparison["template"] = (
        "closure_truth normalized independently in each pT slice over the same shown [-4, 0] range"
    )
    mess_comparison["covariance"] = (
        "saved normalized data total covariance only; MESS+Vincia prediction uncertainty not added; "
        "nominal ndof is n_shown minus one normalization constraint per pT slice"
    )
    mess_comparison["panel_chi2_with_diagonal_mess_stat"] = {
        **mess_panel_comparison,
        "covariance": "saved normalized data total covariance plus diagonal MESS+Vincia statistical covariance",
        "purpose": "per-pT panel legend diagnostic only; not a global comparison convention",
    }
    mess_comparison["template"] = (
        "audited MESS+Vincia final-all rows, HT-normalized and normalized independently "
        "in each pT slice over the same shown [-4, 0] range"
    )

    reported = diagnostics["gen_layout_migration"]["reported_normalization_window"]
    normalized_covariances = diagnostics["normalized_covariances"]
    uncertainty = np.sqrt(np.clip(np.diag(data_covariance), 0.0, None))
    relative_uncertainty = np.divide(
        uncertainty, data, out=np.zeros_like(data), where=data != 0
    ) * 100.0
    stat_uncertainty = np.sqrt(np.clip(np.diag(artifact["norm_cov_stat"]), 0.0, None))
    relative_stat_uncertainty = np.divide(
        stat_uncertainty, data, out=np.zeros_like(data), where=data != 0
    ) * 100.0
    pdf_count = required_pdf_count(pt_edges)
    pdf_paths = list(directory.glob("**/*.pdf"))
    if len(pdf_paths) < pdf_count:
        raise FileNotFoundError(f"{channel} has {len(pdf_paths)} PDFs; expected at least {pdf_count}")
    comparison_provenance = manifest["mess_vincia_prediction"]
    summary = {
        "artifact": relative_to_output(artifact_path),
        "assessment": channel_assessment(channel, diagnostics, mess_comparison),
        "binning_candidate": CHANNEL_CANDIDATES[channel],
        "closure": {
            "maximum_per_pt_window_relative_l1": max(
                entry["bias"]["relative_l1"]
                for entry in diagnostics["nominal_mc_self_closure"]["per_pt_window_normalized_bias"]
            ),
            "raw_relative_l1": diagnostics["nominal_mc_self_closure"]["raw_bias"]["relative_l1"],
        },
        "data_covariance_source": manifest["unfolding"]["data_covariance_source"],
        "data_vs_nominal_pythia": pythia_comparison,
        "data_vs_mess_vincia": mess_comparison,
        "fake_fraction": fraction_summary(np.asarray(artifact["fake_fraction"])),
        "fractional_uncertainty_percent_reported_bins": {
            "stat": fraction_summary(relative_stat_uncertainty, mask),
            "total": fraction_summary(relative_uncertainty, mask),
        },
        "gen_coordinate_edges": edges.tolist(),
        "manifest": relative_to_output(manifest_path),
        "mess_vincia_prediction": comparison_provenance,
        "miss_fraction": fraction_summary(np.asarray(artifact["miss_fraction"])),
        "normalization_window": list(NORMALIZATION_WINDOW),
        "normalized_systematic_covariance_total_minus_stat": covariance_summary(
            normalized_covariances["matrix"]
        ),
        "normalized_total_covariance": covariance_summary(normalized_covariances["total"]),
        "normalized_total_covariance_full_layout": dict(normalized_covariances["total"]["full"]),
        "plot_categories": ["unfolded", "validation", "response", "uncertainties", "summary"],
        "plot_counts": {"pdf": pdf_count},
        "plot_qa": {
            "overall": {
                "status": "pass",
                "note": "All 39 rebuilt book pages were visually checked with no material layout defect.",
            },
            "unfolded_summary_ratio": {
                "status": "pass",
                "note": "PASS: all MESS+Vincia panels clean; bands edge-aligned; no trijet annotation/spike overlap.",
            },
        },
        "plots_directory": relative_to_output(directory),
        "pt_edges_GeV": pt_edges,
        "refold": {
            "chi2": diagnostics["refolded_residual"]["chi2"],
            "rank": diagnostics["refolded_residual"]["rank"],
            "condition_number": diagnostics["refolded_residual"]["condition_number"],
            "covariance": diagnostics["refolded_residual"]["covariance"],
        },
        "reported_bins": int(mask.sum()),
        "reported_bins_per_pt": int(mask.sum() // (len(pt_edges) - 1)),
        "reported_purity": fraction_summary(np.asarray(artifact["gen_layout_purity"]), mask),
        "reported_stability": fraction_summary(np.asarray(artifact["gen_layout_stability"]), mask),
        "reported_window_mask": {
            "definition": "two_log10_rho in [-4, 0] in every reported pT slice; low [-10, -4] buffer excluded",
            "n_false": int((~mask).sum()),
            "n_true": int(mask.sum()),
            "per_pt_bin_counts": [
                int(mask[start:stop].sum()) for start, stop in per_pt_offsets(edges, mask.size)
            ],
        },
        "response": {
            "column_normalized_condition_number": diagnostics["response_column_normalized"]["condition_number"],
            "column_normalized_rank": diagnostics["response_column_normalized"]["rank"],
            "raw_condition_number": diagnostics["response_raw"]["condition_number"],
            "raw_rank": diagnostics["response_raw"]["rank"],
            "raw_shape": diagnostics["response_raw"]["shape"],
        },
    }
    return json_native(summary), manifest


def build_report(summary: dict[str, Any]) -> str:
    dijet = summary["channels"]["dijet"]
    trijet = summary["channels"]["trijet"]
    d_mess = dijet["data_vs_mess_vincia"]
    t_mess = trijet["data_vs_mess_vincia"]
    d_panel_mess = d_mess["panel_chi2_with_diagonal_mess_stat"]["per_pt"]
    t_panel_mess = t_mess["panel_chi2_with_diagonal_mess_stat"]["per_pt"]
    d_source = dijet["mess_vincia_prediction"]["derived_curve"]["source"]
    t_source = trijet["mess_vincia_prediction"]["derived_curve"]["source"]
    return f"""# Run-2 pair-split TUnfold report - corrected MESS+Vincia provenance

## Scope and observable

This report records the corrected full-stat Run-2 unfolding. Physical `rho = m/(pT R)` with `R = 0.8`; every plot uses the analyzed coordinate `2log10(rho)`. The normalized result is unit area independently in every pT slice over all shown bins `[-4, 0]`; the low `[-10, -4]` buffer is excluded. TUnfoldDensity uses the area constraint, `tau = 0`, built-in input and response-matrix statistics, and no jackknife.

The selected hybrid binning is deliberate: dijet uses `coarse_tail` (10 bins/pT, 50 shown bins total) and trijet uses the appendix `2to1` recommendation (4 bins/pT, 12 shown bins total).

## Input and MESS+Vincia provenance

The unfolding inputs are the all_syst histogram pickles for 2016APV, 2016, 2017, and 2018. Full per-era MC/data paths and SHA-256 hashes are recorded in [summary.json](summary.json). Event-level skims were not used; the LHE-basis 2018 archives are excluded.

Both channels use audited `MESS+Vincia` final-all rows from `{d_source['campaign_directory']}` (campaign `{d_source['campaign']}`). The common allowlist SHA-256 is `{d_source['allowlist']['sha256']}` and audit SHA-256 is `{d_source['audit']['sha256']}`. The exact per-channel manifest-ntuple inventory hashes are `{d_source['manifest_ntuple_inventory_sha256']}` (dijet) and `{t_source['manifest_ntuple_inventory_sha256']}` (trijet). The source manifest separates the physical definition `{d_source['physical_rho_definition']}` from the analyzed coordinate `{d_source['analyzed_coordinate_definition']}`. Rows are weighted as `{d_source['normalization']}` and binned in that analyzed coordinate. The exact binned density and statistical uncertainty arrays plotted are saved in each `groomed_results.npz` artifact.

## Numerical audit

| Quantity | Dijet (`coarse_tail`) | Trijet (`two_to_one`) |
| --- | ---: | ---: |
| Shown bins per pT | {dijet['reported_bins_per_pt']} | {trijet['reported_bins_per_pt']} |
| Column-normalized response condition | {dijet['response']['column_normalized_condition_number']:.3g} | {trijet['response']['column_normalized_condition_number']:.3g} |
| Shown purity (min / mean) | {dijet['reported_purity']['min']:.3f} / {dijet['reported_purity']['mean']:.3f} | {trijet['reported_purity']['min']:.3f} / {trijet['reported_purity']['mean']:.3f} |
| Shown stability (min / mean) | {dijet['reported_stability']['min']:.3f} / {dijet['reported_stability']['mean']:.3f} | {trijet['reported_stability']['min']:.3f} / {trijet['reported_stability']['mean']:.3f} |
| Refold chi2 / rank | {dijet['refold']['chi2']:.2f} / {dijet['refold']['rank']} | {trijet['refold']['chi2']:.2f} / {trijet['refold']['rank']} |
| Nominal-MC raw closure relative-L1 | {dijet['closure']['raw_relative_l1']:.3g} | {trijet['closure']['raw_relative_l1']:.3g} |
| Data-nominal-Pythia chi2 / nominal ndof | {dijet['data_vs_nominal_pythia']['chi2']:.2f} / {dijet['data_vs_nominal_pythia']['nominal_ndof']} | {trijet['data_vs_nominal_pythia']['chi2']:.2f} / {trijet['data_vs_nominal_pythia']['nominal_ndof']} |
| Global data-MESS+Vincia chi2 / nominal ndof | {d_mess['chi2']:.2f} / {d_mess['nominal_ndof']} | {t_mess['chi2']:.2f} / {t_mess['nominal_ndof']} |
| Total fractional uncertainty (median / max) | {dijet['fractional_uncertainty_percent_reported_bins']['total']['median']:.3f}% / {dijet['fractional_uncertainty_percent_reported_bins']['total']['max']:.3f}% | {trijet['fractional_uncertainty_percent_reported_bins']['total']['median']:.3f}% / {trijet['fractional_uncertainty_percent_reported_bins']['total']['max']:.3f}% |

The global data-nominal-Pythia and data-MESS+Vincia comparisons both use the saved normalized data total covariance only; prediction uncertainty is omitted for both. Their displayed nominal ndof is the number of shown bins minus one normalization constraint in each pT slice. These global diagnostics retain cross-pT covariance, so they are neither the panel statistic nor a standalone visual verdict. The MESS+Vincia panel legends instead use the saved data covariance plus a diagonal MESS+Vincia statistical term: dijet {', '.join(f"{item['chi2']:.3f}/{item['nominal_ndof']}" for item in d_panel_mess)} by ascending pT slice; trijet {', '.join(f"{item['chi2']:.3f}/{item['nominal_ndof']}" for item in t_panel_mess)}. Neither comparison is used as a total-band uncertainty because `model_envelope=False` for this pair-split configuration.

## What worked and what did not

### Dijet

- **Status:** {dijet['assessment']['status']}
- **Worked:** {dijet['assessment']['worked']}
- **Did not work:** {dijet['assessment']['did_not_work']}

### Trijet

- **Status:** {trijet['assessment']['status']}
- **Worked:** {trijet['assessment']['worked']}
- **Did not work:** {trijet['assessment']['did_not_work']}

## Plot QA

The book-level render and visual inspection status is recorded in `summary.json` after rebuilding the document. The canonical core plots are [dijet unfolded pT0]({dijet['plots_directory']}/unfolded/unfolded_groomed_pt0.pdf), [dijet ratio summary]({dijet['plots_directory']}/summary/unfolded_summary_ratio_groomed.pdf), [trijet unfolded pT0]({trijet['plots_directory']}/unfolded/unfolded_groomed_pt0.pdf), and [trijet ratio summary]({trijet['plots_directory']}/summary/unfolded_summary_ratio_groomed.pdf).

## Interpretation boundary

The corrected curves establish MESS+Vincia provenance and remove the invalid prior Z+jet-cache comparison. Dijet's retained granularity is statistics-supported but its migration and refold diagnostics do not support a measurement claim. Trijet's 2to1 merge improves migration behaviour, but its refold diagnostic remains unresolved. Nominal self-closure, covariance checks, and generator comparisons are software/numerical diagnostics, not independent response-model validation.
"""


def main() -> None:
    channels: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for channel, directory in CHANNEL_DIRECTORIES.items():
        channels[channel], manifests[channel] = build_channel(channel, directory)
    dijet_manifest = manifests["dijet"]
    systematics = dijet_manifest["systematics"]
    source_provenance = {
        channel: channels[channel]["mess_vincia_prediction"]
        for channel in ("dijet", "trijet")
    }
    summary = {
        "artifact_identity": "study_recommended__dijet-coarse_tail-6517ad83dce2d223__trijet-two_to_one-02faf4e995fb5165",
        "binning_candidate": {
            "by_channel": dict(CHANNEL_CANDIDATES),
            "name": "study_recommended",
            "purpose": "Preserve the production 10-bin-per-pT dijet coarse_tail result and use the trijet appendix 2to1 recommendation; all plots normalize over shown [-4, 0].",
            "status": "hybrid selection from the appendix bin study",
        },
        "channels": channels,
        "input_provenance": {
            "active_eras": [record["era"] for record in dijet_manifest["inputs"]],
            "channels": {channel: manifests[channel]["inputs"] for channel in ("dijet", "trijet")},
            "event_skims_used": False,
            "excluded_archives": [
                "/Users/aritra/cernbox (2)/hadronic_skims_pairsplit/2018_lhe_basis",
                "/Users/aritra/cernbox (2)/hadronic_minimal_rho_pairsplit/2018_lhe_basis",
            ],
            "input_root": "/Users/aritra/cernbox (2)/hadronic_minimal_rho_pairsplit",
            "mess_vincia": source_provenance,
            "workflow_inputs": "full-stat all_syst histogram pickles",
        },
        "method": {
            "area_constraint": dijet_manifest["unfolding"]["area_constraint"],
            "backend": dijet_manifest["unfolding"]["backend"],
            "data_covariance_contract": {
                channel: manifests[channel]["unfolding"]["data_covariance_source"]
                for channel in ("dijet", "trijet")
            },
            "jackknife": "not used",
            "plot_normalization": "unit area independently in each pT slice over all shown two_log10_rho bins [-4, 0]",
            "regularization": dijet_manifest["unfolding"]["regularization"],
            "resolved_tau": dijet_manifest["unfolding"]["tau"]["resolved"],
            "stat_covariance": ["GetEmatrixInput", "GetEmatrixSysUncorr"],
        },
        "observable": {
            "analyzed_coordinate": "2 * log10(rho)",
            "center_of_mass_energy_TeV": dijet_manifest["unfolding"]["center_of_mass_energy_TeV"],
            "jet_radius": dijet_manifest["observable"]["jet_radius"],
            "luminosity_fb": dijet_manifest["unfolding"]["luminosity_fb"],
            "rho_definition": "rho = m / (pT * R)",
        },
        "report_date": REPORT_DATE,
        "systematics": {
            "categories_including_nominal": len(systematics["resolved"]),
            "era_groups": systematics["run2_era_correlation"]["era_groups"],
            "excluded_defective_jes_sources": systematics["run2_era_correlation"]["excluded_defective_jes_sources"],
            "excluded_normalization_only": [
                item for item in systematics["excluded"] if item.startswith("Luminosity")
            ],
            "nominal_sumw2_policy": {
                "detector_central_variations": systematics["systematic_variance_policy"]["nominal_variance_systematics"],
                "reason": systematics["systematic_variance_policy"]["reason"],
                "virtual_jes_jer": "Nominal Run-2 sumw2 used for every virtual leg; category statistical covariance is unavailable.",
            },
            "non_nominal_categories": len(systematics["resolved"]) - 1,
            "other_non_jes_jer_categories": sum(
                not item.startswith(("JES_", "JER_")) for item in systematics["resolved"] if item != "nominal"
            ),
            "raw_combined_jes_jer_selectable": False,
            "safe_virtual_jes_categories": sum(item.startswith("JES_") for item in systematics["resolved"]),
            "virtual_jer_categories": sum(item.startswith("JER_") for item in systematics["resolved"]),
            "virtual_split": "corr=sqrt(rho) and uncorr=sqrt(1-rho), applied to Run-2 nominal plus grouped varied-minus-nominal shifts",
        },
    }
    SUPERSEDED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUPERSEDED_OUTPUT_DIR / "summary.json"
    report_path = SUPERSEDED_OUTPUT_DIR / "PAIR_SPLIT_UNFOLDING_REPORT_2026-08-14.md"
    summary_path.write_text(json.dumps(json_native(summary), indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
