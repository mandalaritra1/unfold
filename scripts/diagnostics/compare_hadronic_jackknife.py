#!/usr/bin/env python3
"""Experimental hadronic jackknife comparison using the unchanged TUnfold engine.

Consumes the label-aligned audit NPZs. Keeps production binning, normalization,
tau=0, area constraint and the saved nominal data weight matrix. Does not change
the production tag or add jackknife covariances to its analytic uncertainties.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np

from unfold.engine import Unfolder
from unfold.inputs import prepared_inputs
from unfold.hadronic.inputs import (
    HADRONIC_FINE_AXES, HadronicModeArrays, HadronicRun2Inputs,
    prepare_hadronic_inputs,
)
from unfold.hadronic.run import (
    HadronicOptions, analysis_binning_to_binning, build_hadronic_spec,
)


def jackknife_covariance(replicas):
    replicas = np.asarray(replicas, dtype=float)
    delta = replicas - replicas.mean(axis=0)
    return (len(replicas) - 1) / len(replicas) * (delta.T @ delta)


def load_replicas(root, channel, kind, mode):
    merged = {}
    sample = "data" if kind == "data" else "mg_pythia8"
    for era in ("2016APV", "2016", "2017", "2018"):
        with np.load(root / f"rho_jk_{channel}_{sample}_{era}.npz") as arrays:
            for key in arrays.files:
                if key.startswith(mode + "_"):
                    role = key[len(mode) + 1:]
                    merged[role] = merged.get(role, 0) + arrays[key]
    return merged


def full_sample(replicas):
    # Every event contributes to nine replicas. Sumw2 is also a sum of event
    # contributions, so divide it by nine, not by 81 as histogram scaling would.
    return {key: values.sum(axis=0) / 9 for key, values in replicas.items()}


def input_bundle(channel, mode, mc, data, spec, variant, measured_covariance):
    source = HadronicModeArrays(
        grooming_mode=mode, fine_axes=HADRONIC_FINE_AXES[mode], systematics=("nominal",),
        response_by_systematic={"nominal": mc["response"]},
        response_variance_by_systematic={"nominal": mc["response_variance"]},
        reco_by_systematic={"nominal": mc["reco"]},
        reco_variance_by_systematic={"nominal": mc["reco_variance"]},
        gen_by_systematic={"nominal": mc["gen"]},
        gen_variance_by_systematic={"nominal": mc["gen_variance"]},
        nominal_data=data["reco"], nominal_data_variance=data["reco_variance"],
        # Temporary input to the binning adapter. The fit uses the externally
        # supplied full production covariance below, not this diagonal.
        nominal_data_covariance=np.diag(data["reco_variance"].ravel()),
    )
    study = HadronicRun2Inputs(channel=channel, eras=("2016APV", "2016", "2017", "2018"),
        modes={mode: source}, source_files=(), observable_metadata={
            "data_covariance_source_by_mode": {mode: "fixed saved nominal covariance"}})
    adapted = prepare_hadronic_inputs(study, variant, ("nominal",), grooming_mode=mode)
    result = prepared_inputs(spec, mode == "groomed", analysis_binning_to_binning(adapted.analysis_binning),
        mc_inputs=adapted.mc_inputs, data_inputs=adapted.data_inputs, systematics=["nominal"],
        measured_covariance=measured_covariance, first_reported_pt_bin=0)
    result.study_raw_data_sumw2 = np.diag(adapted.measured_covariance).copy()
    return result


def fit(inputs, spec, mode, *, analytic=False):
    unfolder = Unfolder(inputs, spec, mode == "groomed")
    unfolder._perform_unfold()
    assert unfolder.tau == 0
    unfolder._normalize_result()
    if analytic:
        unfolder._compute_normalized_stat_covariance()
    normalized = np.concatenate([row["unfolded"] for row in unfolder.normalized_results])
    assert np.isfinite(normalized).all()
    return unfolder, normalized


def validate_linear_solution(unfolder):
    response = unfolder.mosaic / unfolder.y_true[None, :]
    covariance = unfolder.corrected_measured_covariance
    weight = np.linalg.inv(covariance)
    inverse = np.linalg.inv(response.T @ weight @ response)
    jacobian0 = inverse @ response.T @ weight
    efficiency = response.sum(axis=0)
    jacobian = jacobian0 + np.outer(
        inverse @ efficiency / (efficiency @ inverse @ efficiency),
        np.ones(len(covariance)) - efficiency @ jacobian0,
    )
    np.testing.assert_allclose(jacobian @ unfolder.y_meas, unfolder.y_unf, rtol=1e-8, atol=1e-4)
    np.testing.assert_allclose(jacobian @ covariance @ jacobian.T, unfolder.cov_data_np,
                               rtol=1e-7, atol=1e-3)
    return jacobian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "audit": str(args.audit / "audit.json"),
        "method": "9/10 sum of centered outer products of ten normalized leave-one-out fits",
        "fit_weight": "Fixed raw measured covariance from the saved nominal reference, retaining dijet event correlations. MC-dependent fake survival is applied consistently.",
        "data_comparison": "The main data comparator propagates this new sample's diagonal sumw2 through the same fixed fit. It omits correlations between jets from an event. The older production full covariance is retained only as a separate reference because sample coverage differs.",
        "nominal": "Replica sums / 9 for values and sumw2; replica-independent processing is an assumption, not proven by these outputs.",
        "response_comparison": "MC response and misses vary together; fake survival stays nominal. Compare to weighted TUnfold GetEmatrixSysUncorr. Event-level MC correlations are present only in jackknife.",
        "full_mc_comparison": "Response, misses and fake survival vary together. This additionally probes fake-correction MC uncertainty; do not interpret its difference from analytic response-only uncertainty as a calibration test.",
        "limitations": ["Only ten replicas: each component covariance has rank at most nine; do not invert it for production fits.",
                        "Chunk-local group labels; partition-count or rechunking stability cannot be tested from these files.",
                        "Documented accepted data shortfalls retained.",
                        "Fit weights use the saved nominal covariance; the main analytic data comparator propagates new diagonal sumw2 because new jackknife files omit event-level covariance."],
        "runs": [],
    }
    for channel in ("dijet", "trijet"):
        for mode in ("groomed", "ungroomed"):
            stem = f"{channel}_{mode}"
            reference_dir = args.reference_root / channel / "rho/original" / mode
            reference_path = reference_dir / "artifacts" / f"{mode}_results.npz"
            reference = np.load(reference_path)
            manifest = json.loads((reference_dir / "run_manifest.json").read_text())
            variant = manifest["binning"]["resolved_candidate"]
            options = HadronicOptions(channel=channel, model_envelope=False,
                normalization_window=manifest["unfolding"]["plot_normalization"]["name"], tau=0)
            spec = build_hadronic_spec(channel, args.output / stem, options, grooming_mode=mode)
            data_replicas = load_replicas(args.audit, channel, "data", mode)
            mc_replicas = load_replicas(args.audit, channel, "mc", mode)
            data = full_sample(data_replicas)
            mc = full_sample(mc_replicas)
            nominal_inputs = input_bundle(channel, mode, mc, data, spec, variant,
                                           reference["measured_covariance"])
            np.testing.assert_array_equal(nominal_inputs.binning.pt_edges, reference["pt_edges"])
            np.testing.assert_array_equal(nominal_inputs.binning.gen_edges, reference["two_log10_rho_gen_edges"])
            np.testing.assert_array_equal(nominal_inputs.binning.reco_edges, reference["two_log10_rho_reco_edges"])
            nominal, center = fit(nominal_inputs, spec, mode, analytic=True)
            linear_map = validate_linear_solution(nominal)
            survival = nominal_inputs.fake_survival_dict['nominal']
            normalization_map = nominal._normalization_jacobian()
            analytic_data_same_sample = normalization_map @ linear_map @ np.diag(
                nominal_inputs.study_raw_data_sumw2 * survival**2) @ linear_map.T @ normalization_map.T
            closure_inputs = copy.deepcopy(nominal_inputs)
            closure_inputs.mosaic_2d = closure_inputs.reco_mc_flat_dict["nominal"]
            closure, _ = fit(closure_inputs, spec, mode)
            np.testing.assert_allclose(closure.y_unf, nominal.y_true, rtol=1e-8, atol=1e-4)
            replicas = {key: [] for key in ("data", "response", "full_mc")}
            raw_data_replicas = []
            for label in range(10):
                varied_data = {key: value[label] for key, value in data_replicas.items()}
                data_inputs = input_bundle(channel, mode, mc, varied_data, spec, variant,
                                           reference["measured_covariance"])
                result, normalized = fit(data_inputs, spec, mode)
                np.testing.assert_allclose(linear_map @ result.y_meas, result.y_unf, rtol=1e-8, atol=1e-4)
                replicas["data"].append(normalized)
                raw_data_replicas.append(result.y_meas)
                varied_mc = {key: value[label] for key, value in mc_replicas.items()}
                full_inputs = input_bundle(channel, mode, varied_mc, data, spec, variant,
                                           reference["measured_covariance"])
                _, normalized = fit(full_inputs, spec, mode)
                replicas["full_mc"].append(normalized)
                response_inputs = copy.deepcopy(full_inputs)
                for key in ("fake_fraction_2d", "fake_fraction_2d_dict", "fake_survival_dict"):
                    setattr(response_inputs, key, copy.deepcopy(getattr(nominal_inputs, key)))
                _, normalized = fit(response_inputs, spec, mode)
                replicas["response"].append(normalized)
            covariances = {key: jackknife_covariance(values) for key, values in replicas.items()}
            norms = nominal._normalization_jacobian()
            # For raw full-yield estimates, rescale the 90% replicas by 10/9.
            propagated_data = norms @ linear_map @ jackknife_covariance(np.asarray(raw_data_replicas)*10/9) @ linear_map.T @ norms.T
            linearized_replicas = (norms @ linear_map @ (np.asarray(raw_data_replicas)*10/9).T).T
            np.testing.assert_allclose(propagated_data, jackknife_covariance(linearized_replicas),
                                       rtol=1e-7, atol=1e-12)
            # Exact per-replica normalization is nonlinear. Its difference
            # from first-order propagation is a result to report, not an
            # elementwise relative-error assertion on near-zero covariances.
            n_gen = len(reference["two_log10_rho_gen_edges"]) - 1
            normalization_vectors = []
            for index in range(len(reference["pt_edges"]) - 1):
                vector = np.zeros(len(center))
                vector[index*n_gen:(index+1)*n_gen] = np.diff(reference["two_log10_rho_gen_edges"]) * nominal._shown_gen_mask(index)
                normalization_vectors.append(vector)
            for covariance in covariances.values():
                assert np.linalg.eigvalsh(covariance).min() >= -1e-10 * max(np.diag(covariance).max(), 1e-20)
                for vector in normalization_vectors:
                    np.testing.assert_allclose(covariance @ vector, 0, atol=1e-12)
            mask = reference["unfolded_reported_window_mask"].astype(bool)
            summary = {"channel": channel, "mode": mode, "reference": str(reference_path),
                       "replicas": 10, "slices": [],
                       "normalization_nonlinearity_relative_frobenius": float(np.linalg.norm(propagated_data-covariances['data'])/np.linalg.norm(covariances['data'])),
                       "raw_data_relative_L1_difference": float(np.abs(nominal_inputs.mosaic_2d - reference['measured_corrected_spectrum']/(1-reference['fake_fraction'])).sum() / np.abs(reference['measured_corrected_spectrum']/(1-reference['fake_fraction'])).sum()),
                       "normalized_relative_L1_difference": float(np.abs(center[mask]-reference['normalized_result'][mask]).sum()/np.abs(reference['normalized_result'][mask]).sum()),
                       "rank": {key: int(np.linalg.matrix_rank(value)) for key,value in covariances.items()}}
            for index, low in enumerate(reference["pt_edges"][:-1]):
                selected = mask[index*n_gen:(index+1)*n_gen]
                sl = slice(index*n_gen, (index+1)*n_gen)
                row = {"pt_low": float(low)}
                for key, analytic in (("data", analytic_data_same_sample), ("response", nominal.norm_cov_matrix), ("full_mc", nominal.norm_cov_matrix)):
                    ratio = np.sqrt(np.diag(covariances[key])[sl][selected] / np.diag(analytic)[sl][selected])
                    row[f"{key}_error_ratio_min_median_max"] = [float(ratio.min()), float(np.median(ratio)), float(ratio.max())]
                summary["slices"].append(row)
            np.savez_compressed(args.output / f"{stem}.npz", center=center,
                reference_center=reference['normalized_result'], pt_edges=reference['pt_edges'],
                gen_edges=reference['two_log10_rho_gen_edges'], reported_mask=mask,
                analytic_data=analytic_data_same_sample, analytic_data_older_reference=nominal.norm_cov_input,
                analytic_response=nominal.norm_cov_matrix,
                **{f"jackknife_{key}": value for key,value in covariances.items()},
                **{f"replicas_{key}": np.asarray(value) for key,value in replicas.items()},
                linearized_jackknife_data=propagated_data)
            report["runs"].append(summary)
            (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"COMPLETED {stem}: {summary}", flush=True)


if __name__ == "__main__":
    main()
