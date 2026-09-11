"""Run one pair-split channel: inputs, model envelope, unfold, artifact, manifest.

The output directory name is a fingerprint of the physics configuration
(``run_configuration_identity``), so a changed binning, systematic list or
model source never overwrites an earlier run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from unfold import plots
from unfold.binning import Binning
from unfold.config import RHO_BASE
from unfold.engine import Unfolder
from unfold.inputs import prepared_inputs
from unfold.paths import REPO_ROOT, PAIRSPLIT_JACKKNIFE_INPUTS
from unfold.pairsplit.diagnostics import RunDiagnostics, collect_run_diagnostics
from unfold.pairsplit.inputs import (
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
from unfold.pairsplit.model_envelope import (
    PairSplitModelEnvelopeInputs,
    derive_pairsplit_model_envelope_inputs,
)
from unfold.pairsplit.vincia import (
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
    "dijet": (-2.5, -0.75),
    "trijet": (-2.5, -1.0),
}
PLOTTED_DISPLAY_WINDOW = (-3.5, 0.0)
UNGROOMED_WINDOW = (-2.5, 0.0)
NORMALIZATION_WINDOWS = {
    "full": PLOTTED_DISPLAY_WINDOW,
    # Aligned-lattice successor of the old [-1.8, -0.55] peak window: the
    # four quarter/half peak bins, upper edge deliberately excluding the
    # [-0.75, 0] kinematic-edge catch-all bin.
    "peak": (-2.0, -0.75),
}
STUDY_RECOMMENDED_BINNING = "study_recommended"
STUDY_RECOMMENDED_VARIANT_BY_CHANNEL = {
    "dijet": "aligned",
    "trijet": "aligned",
}




@dataclass
class PairSplitOptions:
    """Command-line options of a pair-split run (see ``cli.py``)."""

    channel: str = "dijet"
    grooming_mode: str = "both"
    input_root: Path = PAIR_SPLIT_INPUT_ROOT
    binning: str = STUDY_RECOMMENDED_BINNING
    regularization: str = "none"
    normalization_window: str = "full"
    tau: float | None = None
    systematics: str = DEFAULT_SYSTEMATIC_REQUEST
    output_dir: Path = REPO_ROOT / "outputs" / "dijet" / "rho" / "original"   # one subdir per grooming mode
    no_plots: bool = False
    model_envelope: bool = True
    model_covariance: str = "enclosing_ellipsoid"
    cms_label: str = "Internal"
    lumi: float = 138.0
    com: float = 13.0
    requested_binning: str | None = None
    command: str = ""
    stat_method: str = "jackknife"
    jackknife_input_root: Path = PAIRSPLIT_JACKKNIFE_INPUTS


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
    args: PairSplitOptions,
    resolved_systematics: Sequence[str],
    mess_vincia_source: Mapping[str, object],
    model_envelope_source: Mapping[str, object] | None = None,
    *,
    grooming_mode: str = "groomed",
    analysis_binning=None,
    statistics=None,
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
        # The concrete edges are part of the identity (2026-08-18): a changed
        # reco or gen binning under an unchanged candidate name must never
        # silently overwrite an earlier run's immutable directory.
        "binning": (
            {
                "candidate": args.binning,
                "pt_edges_GeV": list(analysis_binning.pt_edges),
                "base_reco_two_log10_rho_edges": list(
                    analysis_binning.two_log10_rho_reco_edges
                ),
                "gen_two_log10_rho_edges": list(
                    analysis_binning.two_log10_rho_gen_edges
                ),
            }
            if analysis_binning is not None
            else args.binning
        ),
        "normalization_window": {
            "name": normalization_name,
            "two_log10_rho_range": list(normalization_window),
        },
        "regularization": args.regularization,
        "requested_tau": args.tau,
        "prediction_statistics": "normalization_jacobian_from_sumw2",
        "statistics": json.loads(json.dumps(statistics)) if statistics is not None else {"requested": args.stat_method},
        "systematics": list(resolved_systematics),
        # This is deliberately part of the directory identity: a new audited
        # MESS campaign or source hash must never reuse a stale Vincia overlay.
        "mess_vincia_source": mess_vincia_source,
        "model_envelope": {
            "enabled": bool(args.model_envelope),
            "source": model_envelope_source,
            "covariance_method": getattr(args, "model_covariance", "enclosing_ellipsoid"),
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
    args: PairSplitOptions, grooming_mode: str
) -> tuple[str, tuple[float, float]]:
    """Resolve the mode-specific published unit-area interval."""

    if grooming_mode == "ungroomed":
        return "minus2p5_to_zero", UNGROOMED_WINDOW
    return args.normalization_window, NORMALIZATION_WINDOWS[args.normalization_window]


def build_pairsplit_spec(
    channel: str,
    output_dir: Path,
    args: PairSplitOptions,
    *,
    grooming_mode: str = "groomed",
):
    """Build a pair-split spec without registering a global analysis tag."""

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
        RHO_BASE,
        output_dir=_core_output_directory(output_dir),
        x_label_groomed=label,
        x_label_ungroomed=label,
        short_label_groomed=label,
        short_label_ungroomed=label,
        xlim_lower_groomed=-3.5,
        xlim_lower_ungroomed=-2.5,
        stat_propagation="jacobian",
        regularization=args.regularization,
        tau=args.tau,
        area_constraint=True,
        model_envelope=bool(args.model_envelope),
        prediction_stat_method="jacobian",
        model_covariance_method=getattr(args, "model_covariance", "enclosing_ellipsoid"),
        model_covariance_scope=(
            "global_templates" if getattr(args, "model_covariance", "enclosing_ellipsoid") == "enclosing_ellipsoid"
            else "global_shown"
        ),
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
        # Channel band colors: light/dark green stays reserved for Z+jet.
        # Dijet: amber/orange; trijet: cyan/teal (CVD-safe, distinct from the
        # Pythia8 blue and MESS+Vincia purple overlay curves).
        band_color_total="#fdd49e" if channel == "dijet" else "#92dadd",
        band_color_stat="#e76300" if channel == "dijet" else "#00707f",
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


def channel_resolved_args(args: PairSplitOptions, channel: str) -> PairSplitOptions:
    """Return options whose binning value is the concrete channel candidate."""

    return replace(args, requested_binning=args.binning,
                   binning=resolve_channel_binning(args.binning, channel))



def analysis_binning_to_binning(analysis_binning) -> Binning:
    return Binning(
        pt_edges=tuple(analysis_binning.pt_edges),
        reco_edges=tuple(analysis_binning.two_log10_rho_reco_edges),
        gen_edges=tuple(analysis_binning.two_log10_rho_gen_edges),
        reco_edges_by_pt=tuple(tuple(e) for e in analysis_binning.reco_two_log10_rho_edges_by_pt),
        gen_edges_by_pt=tuple(tuple(e) for e in analysis_binning.gen_two_log10_rho_edges_by_pt),
    )


def run_nominal_mc_self_closure(
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
    groomed = grooming_mode == "groomed"
    closure_inputs = prepared_inputs(
        closure_spec,
        groomed,
        analysis_binning_to_binning(prepared.analysis_binning),
        mc_inputs=prepared.mc_inputs,
        data_inputs=closure_data_inputs,
        systematics=("nominal",),
        measured_covariance=np.diag(closure_reco_variances),
        first_reported_pt_bin=prepared.first_reported_pt_bin,
    )
    return Unfolder(closure_inputs, closure_spec, groomed, cms_label=cms_label, lumi=lumi, com=com).run()


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
            }
        )
        if unfolder._uses_enclosing_model_covariance():
            artifact_arrays.update({
                "model_ps_covariance": unfolder.model_group_covariances["parton_shower"],
                "model_had_covariance": unfolder.model_group_covariances["hadronization"],
                "model_covariance_method": np.asarray("enclosing_ellipsoid"),
                "model_ps_envelope_fraction": np.concatenate([
                    result["model_envelope_ps_frac"] for result in unfolder.normalized_results]),
                "model_had_envelope_fraction": np.concatenate([
                    result["model_envelope_had_frac"] for result in unfolder.normalized_results]),
            })
        else:
            artifact_arrays.update({
                "model_ps_selected_signed_fraction": np.asarray(unfolder.model_ps_shift_flat, dtype=float),
                "model_had_selected_signed_fraction": np.asarray(unfolder.model_had_shift_flat, dtype=float),
            })
    if getattr(getattr(unfolder, "spec", None), "prediction_stat_method", None) == "jacobian":
        n_bins = len(unfolder.gen_mc_flat_dict["nominal"])
        pythia_covariance = np.zeros((n_bins, n_bins))
        offset = 0
        for i, edges in enumerate(unfolder.gen_edges_by_pt):
            count = len(edges) - 1
            block = unfolder._prediction_stat_covariance(i, "pythia")
            if block is None:
                raise ValueError(f"PYTHIA prediction covariance missing in pT slice {i}")
            pythia_covariance[offset:offset+count, offset:offset+count] = block
            offset += count
        artifact_arrays.update({
            "prediction_stat_method": np.asarray("normalization_jacobian_from_sumw2"),
            "pythia_gen_sumw": unfolder.gen_mc_flat_dict["nominal"],
            "pythia_gen_sumw2": unfolder.gen_mc_var_dict["nominal"],
            "pythia_prediction_stat_covariance": pythia_covariance,
        })
    np.savez_compressed(path, **artifact_arrays)
    if hasattr(unfolder, "jackknife_artifact_arrays"):
        np.savez_compressed(path.with_name("jackknife_statistics.npz"),
                            **unfolder.jackknife_artifact_arrays)
    return path


def build_manifest(
    *,
    args: PairSplitOptions,
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
        "command": args.command,
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
            "fine_to_gen_two_log10_rho_groups": prepared.metadata[
                "fine_to_gen_two_log10_rho_groups"
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
                ("PS=enclose(MESS+Vincia,FSR-up,FSR-down); "
                 "HAD=enclose(CR1,CR2,frag-hard,frag-soft); C_model=C_PS+C_HAD; "
                 "model band=sqrt(diag(C_model)); template bound, not a calibrated confidence region"
                 if getattr(args, "model_covariance", "enclosing_ellipsoid") == "enclosing_ellipsoid"
                 else "PS=max(MESS+Vincia,FSR); HAD=max(CR1,CR2,frag-hard,frag-soft); model=sqrt(PS^2+HAD^2)")
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


def run_channel_plots(unfolder, output_dir: Path, closure_unfolder=None) -> Path | None:
    """The prepared-input subset of the Z+jet figure suite, plus the gallery.

    Not applicable here: HERWIG overlays/bias, jackknife convergence,
    lepton-SF groups, and the L-curve (tau = 0).
    """
    plots.plot_unfolded_fancy(unfolder, show=False)
    plots.plot_folded(unfolder, show=False, counts=True)
    plots.plot_purity_stability(unfolder, show=False)
    plots.plot_systematic_fraction_grouped(unfolder, show=False, log=False)
    if unfolder.spec.model_envelope:
        plots.plot_model_envelope(unfolder, show=False)
    plots.plot_correlation(unfolder, show=False, shown_only=True, covariance="total")
    plots.plot_statistical_fraction(unfolder, show=False)
    plots.plot_fakes_misses(unfolder, show=False)
    plots.plot_response_matrix(unfolder, probability=True, show=False)
    plots.plot_uncertainty_heatmap(unfolder, show=False)
    plots.plot_correlation(unfolder, show=False, shown_only=True)  # stat-only companion
    plots.plot_bottom_line(unfolder, show=False)
    plots.plot_bottom_line(unfolder, show=False, rebin_reco_to_gen=True, annotate_chi2=False)
    plots.plot_bottom_line_chi2_summary(unfolder, show=False, normalized=True, min_edge="shown")
    plots.plot_bottom_line_chi2_summary(unfolder, show=False, min_edge="shown")
    if closure_unfolder is not None:
        # Self-closure panels (stat-only band; saved under <run>/closure/).
        closure_unfolder.closure = True
        plots.plot_unfolded_fancy(closure_unfolder, show=False)
    import shutil
    if shutil.which("pdftoppm") is None:
        return None
    from unfold.gallery import build_gallery
    return build_gallery(output_dir)


def run_channel(
    args: PairSplitOptions, channel: str, *, grooming_mode: str = "groomed"
) -> Path:
    """Load and unfold one channel, keeping only one channel's arrays resident."""

    import ROOT
    from unfold.pairsplit.jackknife import load_jackknife_inputs, check_data_sample, full_sample, apply_jackknife

    args = channel_resolved_args(args, channel)
    inputs = load_pairsplit_run2_inputs(channel, input_root=args.input_root)
    replicas, statistics = load_jackknife_inputs(
        args.jackknife_input_root, channel, grooming_mode, requested=args.stat_method
    )
    print(f"Statistics: requested {args.stat_method}, using {statistics['resolved']}", flush=True)
    if "fallback_reason" in statistics:
        print(f"Analytic fallback: {statistics['fallback_reason']}: "
              + ", ".join(statistics["missing_files"]), flush=True)
    if replicas is not None:
        check_data_sample(inputs.modes[grooming_mode].nominal_data, full_sample(replicas.data)["reco"])
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
        analysis_binning=prepared.analysis_binning,
        statistics=statistics,
    )
    # <tag dir>/<mode>/; the configuration fingerprint stays in the manifest
    output_dir = (Path(args.output_dir) / grooming_mode).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = build_pairsplit_spec(
        channel, output_dir, args, grooming_mode=grooming_mode
    )
    ROOT.gErrorIgnoreLevel = ROOT.kError
    groomed = grooming_mode == "groomed"
    engine_inputs = prepared_inputs(
        spec,
        groomed,
        analysis_binning_to_binning(prepared.analysis_binning),
        mc_inputs=prepared.mc_inputs,
        data_inputs=prepared.data_inputs,
        systematics=prepared.systematics,
        measured_covariance=prepared.measured_covariance,
        first_reported_pt_bin=prepared.first_reported_pt_bin,
    )
    unfolder = Unfolder(engine_inputs, spec, groomed, cms_label=args.cms_label, lumi=args.lumi, com=args.com).run(
        statistics=(lambda u: apply_jackknife(u, replicas, args.binning)) if replicas is not None else None
    )
    attach_pairsplit_vincia_prediction(unfolder, vincia_prediction)
    closure_unfolder = run_nominal_mc_self_closure(
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
    gallery = None if args.no_plots else run_channel_plots(
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
                "method": getattr(spec, "model_covariance_method", "selected_variation"),
                "template_containment": getattr(unfolder, "model_covariance_diagnostics", None),
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
    manifest["unfolding"]["statistics"] = statistics
    if replicas is not None:
        manifest["unfolding"].update(
            stat_propagation="normalize each jackknife replica",
            response_statistics="fixed-fake response and GEN/misses jackknife",
            jackknife="data and MC covariances estimated separately and added",
        )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_all(args: PairSplitOptions) -> list[Path]:
    """Run every requested channel and grooming mode; return the manifest paths."""
    grooming_modes = ("groomed", "ungroomed") if args.grooming_mode == "both" else (args.grooming_mode,)
    manifests = []
    channel = args.channel
    resolved_binning = resolve_channel_binning(args.binning, channel)
    for grooming_mode in grooming_modes:
        print(f"Running {grooming_mode} pair-split {channel} with {resolved_binning} (requested {args.binning})")
        manifests.append(run_channel(args, channel, grooming_mode=grooming_mode))
        print(f"manifest: {manifests[-1]}")
    return manifests
