from __future__ import annotations

import importlib.util
import sys
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unfold.tools.pairsplit_run2_inputs import (
    DEFECTIVE_JES_SOURCES,
    FULL_SAFE_SYSTEMATIC_REQUEST,
    LEGACY_HISTOGRAM_KEYS,
    PAIR_SPLIT_FINE_AXES,
    PAIR_SPLIT_NOMINAL_VARIANCE_SYSTEMATICS,
    PairSplitModeArrays,
    PairSplitRun2Inputs,
    _add_run2_era_correlation_virtuals,
    pair_split_binning,
    prepare_pairsplit_inputs,
    prepare_pairsplit_groomed_inputs,
    resolve_pairsplit_systematics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_MESS_VINCIA_SOURCE = {
    "campaign": "synthetic-final-all",
    "allowlist": {"sha256": "allowlist-a"},
    "audit": {"sha256": "audit-a"},
    "manifest_ntuple_inventory_sha256": "inventory-a",
}


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_pairsplit_unfolding",
        REPO_ROOT / "scripts" / "run_pairsplit_unfolding.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_pairsplit_inputs() -> PairSplitRun2Inputs:
    fine = PAIR_SPLIT_FINE_AXES["groomed"]
    response_shape = (
        len(fine.pt_edges) - 1,
        len(fine.two_log10_rho_reco_edges) - 1,
        len(fine.pt_edges) - 1,
        len(fine.two_log10_rho_gen_edges) - 1,
    )
    nominal_response = np.zeros(response_shape)
    nominal_variance = np.zeros(response_shape)
    nominal_response[1, 4, 2, 2] = 5.0
    nominal_variance[1, 4, 2, 2] = 7.0
    nominal_response[6, 14, 7, 11] = 3.0
    nominal_variance[6, 14, 7, 11] = 11.0
    shifted_response = nominal_response * 2.0
    shifted_variance = nominal_variance * 3.0

    reco_values = nominal_response.sum(axis=(2, 3))
    reco_variances = nominal_variance.sum(axis=(2, 3))
    gen_values = nominal_response.sum(axis=(0, 1))
    gen_variances = nominal_variance.sum(axis=(0, 1))
    data_values = reco_values + 10.0
    data_variances = reco_variances + 2.0
    source_covariance = np.zeros(reco_values.shape * 2)
    flat_covariance = source_covariance.reshape(data_values.size, data_values.size)
    np.fill_diagonal(flat_covariance, 4.0)
    source_left = 1 * reco_values.shape[1] + 4
    source_right = 2 * reco_values.shape[1] + 6
    flat_covariance[source_left, source_right] = 2.5
    flat_covariance[source_right, source_left] = 2.5

    mode = PairSplitModeArrays(
        grooming_mode="groomed",
        fine_axes=fine,
        systematics=("nominal", "JERUp"),
        response_by_systematic={
            "nominal": nominal_response,
            "JERUp": shifted_response,
        },
        response_variance_by_systematic={
            "nominal": nominal_variance,
            "JERUp": shifted_variance,
        },
        reco_by_systematic={
            "nominal": reco_values,
            "JERUp": reco_values * 2.0,
        },
        reco_variance_by_systematic={
            "nominal": reco_variances,
            "JERUp": reco_variances * 3.0,
        },
        gen_by_systematic={
            "nominal": gen_values,
            "JERUp": gen_values * 2.0,
        },
        gen_variance_by_systematic={
            "nominal": gen_variances,
            "JERUp": gen_variances * 3.0,
        },
        nominal_data=data_values,
        nominal_data_variance=data_variances,
        nominal_data_covariance=source_covariance,
    )
    return PairSplitRun2Inputs(
        channel="dijet",
        eras=("synthetic",),
        modes={"groomed": mode},
        source_files=(),
        observable_metadata={
            "data_covariance_source_by_mode": {"groomed": "full_reco_covariance"}
        },
    )


def synthetic_ungroomed_inputs() -> PairSplitRun2Inputs:
    fine = PAIR_SPLIT_FINE_AXES["ungroomed"]
    response_shape = (
        len(fine.pt_edges) - 1,
        len(fine.two_log10_rho_reco_edges) - 1,
        len(fine.pt_edges) - 1,
        len(fine.two_log10_rho_gen_edges) - 1,
    )
    response = np.zeros(response_shape)
    response[1, 15, 1, 9] = 10.0
    response[8, 24, 8, 13] = 5.0
    variance = response.copy()
    reco = response.sum(axis=(2, 3))
    gen = response.sum(axis=(0, 1))
    data_covariance = np.diag(np.ones(reco.size)).reshape(reco.shape * 2)
    mode = PairSplitModeArrays(
        grooming_mode="ungroomed",
        fine_axes=fine,
        systematics=("nominal",),
        response_by_systematic={"nominal": response},
        response_variance_by_systematic={"nominal": variance},
        reco_by_systematic={"nominal": reco},
        reco_variance_by_systematic={"nominal": variance.sum(axis=(2, 3))},
        gen_by_systematic={"nominal": gen},
        gen_variance_by_systematic={"nominal": variance.sum(axis=(0, 1))},
        nominal_data=reco,
        nominal_data_variance=np.ones_like(reco),
        nominal_data_covariance=data_covariance,
    )
    return PairSplitRun2Inputs(
        channel="dijet",
        eras=("synthetic",),
        modes={"ungroomed": mode},
        source_files=(),
        observable_metadata={
            "data_covariance_source_by_mode": {
                "ungroomed": "full_reco_covariance"
            }
        },
    )


def test_ungroomed_bridge_keeps_hidden_tail_and_five_reported_bins():
    prepared = prepare_pairsplit_inputs(
        synthetic_ungroomed_inputs(),
        "coarse_tail",
        ("nominal",),
        grooming_mode="ungroomed",
    )
    expected_edges = (-10.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0)
    assert prepared.candidate.grooming_mode == "ungroomed"
    assert prepared.candidate.reported_two_log10_rho_minimum == -2.5
    assert prepared.analysis_binning.two_log10_rho_reco_edges == expected_edges
    assert prepared.analysis_binning.two_log10_rho_gen_edges == expected_edges
    assert prepared.metadata["grooming_mode"] == "ungroomed"
    response = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["ungroomed"]["response"]]
    assert response.values(flow=False).shape == (1, 5, 6, 5, 6)
    assert response.values(flow=False).sum() == pytest.approx(15.0)


def detector_variance_pairsplit_inputs() -> PairSplitRun2Inputs:
    """A varied detector leg with nominal GEN and incompatible raw sumw2."""

    base = synthetic_pairsplit_inputs()
    mode = base.modes["groomed"]
    nominal_response = mode.response_by_systematic["nominal"]
    nominal_response_variance = mode.response_variance_by_systematic["nominal"]
    nominal_reco = nominal_response.sum(axis=(2, 3)) + 1.0
    nominal_reco_variance = nominal_response_variance.sum(axis=(2, 3)) + 1.0
    nominal_gen = nominal_response.sum(axis=(0, 1)) + 2.0
    nominal_gen_variance = nominal_response_variance.sum(axis=(0, 1)) + 2.0
    detector_response = nominal_response * 1.1
    detector_reco = nominal_reco * 1.1
    detector_response_variance = nominal_response_variance * 1.5
    detector_reco_variance = nominal_reco_variance * 1.5

    detector_mode = replace(
        mode,
        systematics=("nominal", "puUp", "LuminosityUp"),
        response_by_systematic={
            "nominal": nominal_response,
            "puUp": detector_response,
            "LuminosityUp": nominal_response * 1.016,
        },
        response_variance_by_systematic={
            "nominal": nominal_response_variance,
            "puUp": detector_response_variance,
            "LuminosityUp": nominal_response_variance * 1.016**2,
        },
        reco_by_systematic={
            "nominal": nominal_reco,
            "puUp": detector_reco,
            "LuminosityUp": nominal_reco * 1.016,
        },
        reco_variance_by_systematic={
            "nominal": nominal_reco_variance,
            "puUp": detector_reco_variance,
            "LuminosityUp": nominal_reco_variance * 1.016**2,
        },
        # The producer intentionally keeps detector-side GEN marginals nominal.
        gen_by_systematic={
            "nominal": nominal_gen,
            "puUp": nominal_gen,
            "LuminosityUp": nominal_gen,
        },
        gen_variance_by_systematic={
            "nominal": nominal_gen_variance,
            "puUp": nominal_gen_variance,
            "LuminosityUp": nominal_gen_variance,
        },
    )
    return replace(base, modes={"groomed": detector_mode})


def test_bridge_rebins_values_sumw2_and_full_covariance_in_source_axis_order():
    prepared = prepare_pairsplit_groomed_inputs(
        synthetic_pairsplit_inputs(),
        "two_to_one",
        ("nominal", "JERUp"),
    )
    response = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["groomed"]["response"]]
    response_values = response.values(flow=False)
    response_variances = response.variances(flow=False)

    assert response_values.shape == (2, 5, 13, 5, 7)
    assert response_values[0].sum() == pytest.approx(8.0)
    assert response_variances[0].sum() == pytest.approx(18.0)
    assert response_values[1].sum() == pytest.approx(16.0)
    assert response_variances[1].sum() == pytest.approx(54.0)
    assert prepared.analysis_binning.pt_edges == (200.0, 290.0, 400.0, 480.0, 570.0, 13000.0)
    assert prepared.analysis_binning.two_log10_rho_reco_edges[0:3] == (-10.0, -4.0, -3.4)
    assert prepared.analysis_binning.two_log10_rho_gen_edges == (
        -10.0, -4.0, -2.85, -1.8, -1.3, -0.9, -0.65, 0.0
    )
    assert prepared.first_reported_pt_bin == 0

    # source (pt=1, coord=4) -> output (0, 1); source (2, 6) -> (1, 2).
    output_left = 0 * 13 + 1
    output_right = 1 * 13 + 2
    assert prepared.measured_covariance[output_left, output_right] == pytest.approx(2.5)
    assert prepared.measured_covariance[output_right, output_left] == pytest.approx(2.5)


def test_bridge_builds_model_response_before_gen_merge_and_keeps_absolute_fakes():
    inputs = synthetic_pairsplit_inputs()
    fine = inputs.modes["groomed"].fine_axes
    weight = np.ones(
        (len(fine.pt_edges) - 1, len(fine.two_log10_rho_gen_edges) - 1)
    )
    weight[2, 2] = 2.0
    prepared = prepare_pairsplit_groomed_inputs(
        inputs,
        "two_to_one",
        ("nominal",),
        model_variations={"model_vincia": weight},
        model_metadata={"synthetic": True},
    )
    response = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["groomed"]["response"]]
    reco = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["groomed"]["reco"]]
    gen = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["groomed"]["gen"]]

    assert prepared.systematics == ("nominal", "model_vincia")
    assert response.values(flow=False)[0].sum() == pytest.approx(8.0)
    assert response.values(flow=False)[1].sum() == pytest.approx(13.0)
    assert response.variances(flow=False)[1].sum() == pytest.approx(39.0)
    assert reco.values(flow=False)[1].sum() == pytest.approx(13.0)
    assert gen.values(flow=False)[1].sum() == pytest.approx(13.0)
    assert prepared.metadata["model_envelope"] == {"synthetic": True}

    with pytest.raises(ValueError, match="expected"):
        prepare_pairsplit_groomed_inputs(
            inputs,
            "two_to_one",
            ("nominal",),
            model_variations={"model_vincia": np.ones((2, 2))},
        )


def test_detector_central_variations_keep_nominal_sumw2_and_luminosity_is_excluded():
    inputs = detector_variance_pairsplit_inputs()
    prepared = prepare_pairsplit_groomed_inputs(
        inputs, "window_aligned_coarse", ("nominal", "puUp")
    )
    response = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["groomed"]["response"]]
    reco = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["groomed"]["reco"]]
    gen = prepared.mc_inputs[LEGACY_HISTOGRAM_KEYS["groomed"]["gen"]]

    # Central response/reco values retain the detector variation, while all
    # MC-stat arrays use the valid nominal Run-2 sumw2 contract.
    assert response.values(flow=False)[1].sum() == pytest.approx(
        response.values(flow=False)[0].sum() * 1.1
    )
    assert reco.values(flow=False)[1].sum() == pytest.approx(
        reco.values(flow=False)[0].sum() * 1.1
    )
    assert np.allclose(response.variances(flow=False)[1], response.variances(flow=False)[0])
    assert np.allclose(reco.variances(flow=False)[1], reco.variances(flow=False)[0])
    assert np.allclose(gen.variances(flow=False)[1], gen.variances(flow=False)[0])
    assert "puUp" in prepared.systematics
    assert "puUp" in prepared.metadata["systematic_variance_policy"]["nominal_variance_systematics"]
    assert "JMSUp" in PAIR_SPLIT_NOMINAL_VARIANCE_SYSTEMATICS

    with pytest.raises(ValueError, match="normalization-only luminosity"):
        prepare_pairsplit_groomed_inputs(
            inputs, "window_aligned_coarse", ("nominal", "LuminosityUp")
        )


def test_safe_systematics_resolve_derived_era_legs_and_reject_raw_categories():
    available = (
        "nominal", "JMSUp", "JMSDown", "JMRUp", "JMRDown", "fsrUp", "fsrDown", "puUp", "puDown",
        "LuminosityUp", "LuminosityDown",
        "JER_uncorr_2016Up", "JER_uncorr_2016Down",
        "JER_uncorr_2017Up", "JER_uncorr_2017Down",
        "JER_uncorr_2018Up", "JER_uncorr_2018Down",
        "JES_PileUpDataMC_corrUp", "JES_PileUpDataMC_corrDown",
        "JES_PileUpDataMC_uncorr_2016Up", "JES_PileUpDataMC_uncorr_2016Down",
        "JES_PileUpDataMC_uncorr_2017Up", "JES_PileUpDataMC_uncorr_2017Down",
        "JES_PileUpDataMC_uncorr_2018Up", "JES_PileUpDataMC_uncorr_2018Down",
    )
    assert resolve_pairsplit_systematics(
        available, "nominal,JER,JMS,JMR,FSR"
    ) == (
        "nominal", "JER_uncorr_2016Up", "JER_uncorr_2016Down",
        "JER_uncorr_2017Up", "JER_uncorr_2017Down", "JER_uncorr_2018Up",
        "JER_uncorr_2018Down", "JMSUp", "JMSDown",
        "JMRUp", "JMRDown", "fsrUp", "fsrDown",
    )
    assert resolve_pairsplit_systematics(available, "JES_PileUpDataMC") == (
        "nominal", "JES_PileUpDataMC_corrUp", "JES_PileUpDataMC_corrDown",
        "JES_PileUpDataMC_uncorr_2016Up", "JES_PileUpDataMC_uncorr_2016Down",
        "JES_PileUpDataMC_uncorr_2017Up", "JES_PileUpDataMC_uncorr_2017Down",
        "JES_PileUpDataMC_uncorr_2018Up", "JES_PileUpDataMC_uncorr_2018Down",
    )
    full_safe = resolve_pairsplit_systematics(available, FULL_SAFE_SYSTEMATIC_REQUEST)
    assert full_safe[0] == "nominal"
    assert {"JMSUp", "JMRDown", "puUp", "JER_uncorr_2018Up", "JES_PileUpDataMC_corrDown"} <= set(full_safe)
    assert "LuminosityUp" not in full_safe
    with pytest.raises(ValueError, match="normalization-only luminosity"):
        resolve_pairsplit_systematics(available, "Luminosity")
    with pytest.raises(ValueError, match="Raw combined Run-2 category"):
        resolve_pairsplit_systematics(available, "JERUp")
    with pytest.raises(ValueError, match="Raw combined Run-2 category"):
        resolve_pairsplit_systematics(available, "JES_PileUpDataMCUp")


def test_virtual_run2_legs_use_sqrt_correlation_and_nominal_sumw2():
    raw_systematics = (
        "nominal",
        "JERUp", "JERDown",
        "JES_AbsoluteScaleUp", "JES_AbsoluteScaleDown",
        "JES_PileUpDataMCUp", "JES_PileUpDataMCDown",
        "JES_RelativeJEREC1Up", "JES_RelativeJEREC1Down",
    )
    era_names = ("2016APV", "2016", "2017", "2018")
    shifts = {"2016APV": 1.0, "2016": 2.0, "2017": 3.0, "2018": 4.0}
    roles = ("response", "reco", "gen")
    per_era = {}
    accumulator = {
        "response": {}, "response_variance": {},
        "reco": {}, "reco_variance": {},
        "gen": {}, "gen_variance": {},
    }
    for era in era_names:
        per_era[era] = {name: {} for name in accumulator}
        for role in roles:
            nominal = np.array([10.0])
            variance = np.array([7.0])
            per_era[era][role]["nominal"] = nominal
            per_era[era][f"{role}_variance"]["nominal"] = variance
            for prefix in ("JER", "JES_AbsoluteScale", "JES_PileUpDataMC", "JES_RelativeJEREC1"):
                per_era[era][role][f"{prefix}Up"] = nominal + shifts[era]
                per_era[era][role][f"{prefix}Down"] = nominal - shifts[era]
                # These deliberately differ from nominal to prove virtual
                # legs do not manufacture a variance from source variations.
                per_era[era][f"{role}_variance"][f"{prefix}Up"] = variance * 9.0
                per_era[era][f"{role}_variance"][f"{prefix}Down"] = variance * 11.0
    for name in accumulator:
        accumulator[name] = {
            systematic: sum(per_era[era][name][systematic] for era in era_names)
            for systematic in raw_systematics
        }

    systematics, metadata = _add_run2_era_correlation_virtuals(
        accumulator, per_era, raw_systematics, era_names
    )
    nominal = 40.0
    root_half = np.sqrt(0.5)
    for role in roles:
        assert accumulator[role]["JES_PileUpDataMC_corrUp"][0] == pytest.approx(
            nominal + root_half * 10.0
        )
        # 2016 comprises 2016APV+2016, so its shift is 1+2, not either era alone.
        assert accumulator[role]["JES_PileUpDataMC_uncorr_2016Up"][0] == pytest.approx(
            nominal + root_half * 3.0
        )
        assert accumulator[role]["JER_uncorr_2016Up"][0] == pytest.approx(nominal + 3.0)
        assert accumulator[role]["JES_AbsoluteScale_corrUp"][0] == pytest.approx(nominal + 10.0)
        assert accumulator[f"{role}_variance"]["JES_PileUpDataMC_corrUp"][0] == 28.0
        assert accumulator[f"{role}_variance"]["JER_uncorr_2016Up"][0] == 28.0

    assert "JER_corrUp" not in systematics  # rho(JER)=0
    assert "JES_AbsoluteScale_uncorr_2016Up" not in systematics  # rho=1
    assert "JES_RelativeJEREC1_corrUp" not in systematics
    assert "JES_RelativeJEREC1" in metadata["excluded_defective_jes_sources"]
    assert "JERUp" not in systematics
    assert "JES_PileUpDataMCUp" not in systematics
    assert set(metadata["generated_legs"]) <= set(systematics)
    assert DEFECTIVE_JES_SOURCES == (
        "RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF"
    )


def test_runner_cli_defaults_and_manifest_use_two_log10_rho_terminology(tmp_path):
    runner = load_runner_module()
    args = runner.parse_args(["--channel", "dijet", "--channel", "trijet", "--no-plots"])
    assert args.channel == ("dijet", "trijet")
    assert args.systematics == FULL_SAFE_SYSTEMATIC_REQUEST
    assert args.binning == runner.STUDY_RECOMMENDED_BINNING
    assert args.regularization == "none"
    assert args.model_envelope is True

    dijet_args = runner.channel_resolved_args(args, "dijet")
    trijet_args = runner.channel_resolved_args(args, "trijet")
    assert dijet_args.requested_binning == runner.STUDY_RECOMMENDED_BINNING
    assert trijet_args.requested_binning == runner.STUDY_RECOMMENDED_BINNING
    assert dijet_args.binning == "coarse_tail"
    assert trijet_args.binning == "two_to_one"

    prepared = prepare_pairsplit_groomed_inputs(
        synthetic_pairsplit_inputs(), "coarse_tail", ("nominal",)
    )
    prepared = replace(
        prepared,
        metadata={
            **prepared.metadata,
            "run2_era_correlation": {
                "applied": True,
                "prescription": "sqrt covariance decomposition",
                "generated_legs_by_mode": {
                    "groomed": ["JER_uncorr_2016Up", "JER_uncorr_2016Down"]
                },
                "excluded_defective_jes_sources": ["JES_RelativeJEREC1"],
                "virtual_variance": "nominal Run-2 sumw2 copied",
            },
        },
    )
    run_identity = runner.run_configuration_identity(
        dijet_args, prepared.systematics, SYNTHETIC_MESS_VINCIA_SOURCE
    )
    assert run_identity["configuration"]["binning"] == "coarse_tail"
    assert run_identity != runner.run_configuration_identity(
        trijet_args, prepared.systematics, SYNTHETIC_MESS_VINCIA_SOURCE
    )
    manifest = runner.build_manifest(
        args=dijet_args,
        channel="dijet",
        prepared=prepared,
        source_records=[],
        artifact=tmp_path / "groomed_results.npz",
        resolved_tau=0.0,
        root_version="synthetic",
        run_identity=run_identity,
        diagnostics=runner.RunDiagnostics(metrics={"synthetic": True}, arrays={}),
        vincia_prediction=SimpleNamespace(
            provenance_payload=lambda: {"label": "MESS+Vincia", "synthetic": True}
        ),
        vincia_reference_validation={"applies": True, "synthetic": True},
    )
    observable = manifest["observable"]
    assert observable["transformed_coordinate_name"] == "two_log10_rho"
    assert observable["transformed_coordinate_definition"] == "two_log10_rho = 2 * log10(rho)"
    assert manifest["binning"]["base_reco_two_log10_rho_edges"] == (
        -10.0, -4.0, -3.4, -2.85, -2.25, -1.8, -1.5, -1.3,
        -1.1, -0.9, -0.75, -0.65, -0.55, 0.0,
    )
    assert manifest["binning"]["requested_candidate"] == "study_recommended"
    assert manifest["binning"]["resolved_candidate"] == "coarse_tail"
    assert manifest["unfolding"]["plot_normalization"] == {
        "mode": "unit_area_over_coordinate_window",
        "name": "full",
        "two_log10_rho_range": [-4.0, 0.0],
    }
    assert manifest["unfolding"]["plot_display"] == {
        "two_log10_rho_range": [-4.0, 0.0]
    }
    assert manifest["unfolding"]["study_core_window_two_log10_rho"] == (-2.85, -0.55)
    assert "not the plotted" in manifest["unfolding"]["study_core_window_purpose"]
    assert "log10(rho^2)" not in str(manifest)
    era_correlation = manifest["systematics"]["run2_era_correlation"]
    assert era_correlation["applied"] is True
    assert era_correlation["generated_legs_by_mode"]["groomed"] == [
        "JER_uncorr_2016Up", "JER_uncorr_2016Down"
    ]
    assert era_correlation["virtual_variance"] == "nominal Run-2 sumw2 copied"
    assert "JES_RelativeJEREC1" in manifest["systematics"]["excluded"]
    assert "LuminosityUp" in manifest["systematics"]["excluded"]
    assert "normalization-only" in manifest["systematics"]["excluded_details"]["LuminosityUp"]
    assert "puUp" in manifest["systematics"]["systematic_variance_policy"][
        "nominal_variance_systematics"
    ]
    assert manifest["run_identity"] == run_identity
    assert manifest["diagnostics"] == {"synthetic": True}
    assert manifest["unfolding"]["data_covariance_source"] == "full_reco_covariance"
    assert manifest["unfolding"]["luminosity_fb"] == pytest.approx(138.0)
    assert manifest["unfolding"]["center_of_mass_energy_TeV"] == pytest.approx(13.0)
    assert json.loads(json.dumps(manifest))["workflow"] == "Run-2 groomed pair-split TUnfold"


def test_runner_spec_normalizes_every_shown_bin_and_keeps_core_window_diagnostic(tmp_path):
    runner = load_runner_module()
    requested_args = runner.parse_args(["--channel", "dijet"])
    args = runner.channel_resolved_args(requested_args, "dijet")
    spec = runner.build_pairsplit_spec("dijet", tmp_path, args)

    assert spec.normalization_window_groomed == (-4.0, 0.0)
    assert spec.display_window_groomed == (-4.0, 0.0)
    assert spec.normalize_over_shown is False
    assert spec.model_envelope is True
    assert spec.model_envelope_source == "prepared_systematics"
    assert spec.xlim_lower_groomed == pytest.approx(-4.0)
    assert runner.CORE_STABILITY_WINDOWS["dijet"] == (-2.85, -0.55)
    assert runner.resolve_channel_binning("coarse_tail", "trijet") == "coarse_tail"


def test_runner_peak_normalization_is_separate_from_display_and_identity(tmp_path):
    runner = load_runner_module()
    full = runner.parse_args(["--channel", "dijet"])
    peak = runner.parse_args(
        ["--channel", "dijet", "--normalization-window", "peak"]
    )
    peak_resolved = runner.channel_resolved_args(peak, "dijet")
    spec = runner.build_pairsplit_spec("dijet", tmp_path, peak_resolved)

    assert spec.normalization_window_groomed == (-1.8, -0.55)
    assert spec.display_window_groomed == (-4.0, 0.0)
    assert runner.run_configuration_identity(
        full, ("nominal",), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"] != runner.run_configuration_identity(
        peak, ("nominal",), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"]


def test_runner_ungroomed_window_label_and_identity_are_mode_specific(tmp_path):
    runner = load_runner_module()
    args = runner.channel_resolved_args(
        runner.parse_args(["--channel", "dijet", "--grooming-mode", "ungroomed"]),
        "dijet",
    )
    spec = runner.build_pairsplit_spec(
        "dijet", tmp_path, args, grooming_mode="ungroomed"
    )
    assert spec.normalization_window_ungroomed == (-2.5, 0.0)
    assert spec.display_window_ungroomed == (-2.5, 0.0)
    assert spec.x_label_ungroomed == r"$\log_{10}(\rho^2)$, ungroomed"
    identity = runner.run_configuration_identity(
        args,
        ("nominal",),
        SYNTHETIC_MESS_VINCIA_SOURCE,
        grooming_mode="ungroomed",
    )
    assert "normalization-minus2p5_to_zero" in identity["directory_name"]
    assert identity["configuration"]["grooming_mode"] == "ungroomed"


@pytest.mark.parametrize(
    ("channel", "expected_variant", "expected_edges", "expected_shown_bins"),
    (
        (
            "dijet",
            "coarse_tail",
            (-10.0, -4.0, -2.85, -1.8, -1.5, -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.0),
            10,
        ),
        (
            "trijet",
            "two_to_one",
            (-10.0, -4.0, -2.25, -1.5, -0.75, 0.0),
            4,
        ),
    ),
)
def test_study_recommended_resolves_exact_edges_and_shown_counts(
    channel,
    expected_variant,
    expected_edges,
    expected_shown_bins,
):
    runner = load_runner_module()
    requested_args = runner.parse_args(["--channel", channel])
    resolved_args = runner.channel_resolved_args(requested_args, channel)
    candidate = pair_split_binning(channel, resolved_args.binning)

    assert resolved_args.requested_binning == runner.STUDY_RECOMMENDED_BINNING
    assert resolved_args.binning == expected_variant
    assert candidate.gen_two_log10_rho_edges == expected_edges
    assert sum(edge >= -4.0 for edge in candidate.gen_two_log10_rho_edges[:-1]) == expected_shown_bins


def test_runner_focused_plots_use_canonical_core_prepared_paths(monkeypatch, tmp_path):
    runner = load_runner_module()
    calls = []

    class Recorder:
        spec = SimpleNamespace(model_envelope=True)

        def __getattr__(self, name):
            if not name.startswith("plot_"):
                raise AttributeError(name)

            def record(**kwargs):
                calls.append((name, kwargs))

            return record

    class ClosureRecorder:
        closure = False

        def plot_unfolded_fancy(self, **kwargs):
            calls.append(("closure.plot_unfolded_fancy", dict(kwargs, closure=self.closure)))

    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    closure_recorder = ClosureRecorder()
    assert runner.run_focused_plots(
        Recorder(), tmp_path, closure_unfolder=closure_recorder
    ) is None
    assert calls == [
        ("plot_unfolded_fancy", {"show": False}),
        ("plot_folded", {"show": False, "counts": True}),
        ("plot_purity_stability", {"show": False}),
        ("plot_systematic_fraction_grouped", {"show": False, "log": False}),
        ("plot_model_envelope", {"show": False}),
        (
            "plot_correlation",
            {"show": False, "shown_only": True, "covariance": "total"},
        ),
        # Z+jet-parity validation checks (2026-08-18).
        ("plot_statistical_fraction", {"show": False}),
        ("plot_fakes_misses", {"show": False}),
        ("plot_response_matrix", {"probability": True, "show": False}),
        ("plot_uncertainty_heatmap", {"show": False}),
        ("plot_correlation", {"show": False, "shown_only": True}),
        ("plot_bottom_line", {"show": False}),
        (
            "plot_bottom_line",
            {"show": False, "rebin_reco_to_gen": True, "annotate_chi2": False},
        ),
        (
            "plot_bottom_line_chi2_summary",
            {"show": False, "normalized": True, "min_edge": "shown"},
        ),
        ("plot_bottom_line_chi2_summary", {"show": False, "min_edge": "shown"}),
        ("closure.plot_unfolded_fancy", {"show": False, "closure": True}),
    ]
    assert closure_recorder.closure is True
    runner_source = (REPO_ROOT / "scripts" / "run_pairsplit_unfolding.py").read_text()
    assert "plot_pairsplit_results" not in runner_source
    focused_body = runner_source.split("def run_focused_plots", 1)[1].split("def run_channel", 1)[0]
    # Inapplicable Z+jet products stay out: no herwig, jackknife, or L-curve.
    for banned in ("plot_herwig", "plot_jk(", "plot_jackknife", "plot_lcurve"):
        assert banned not in focused_body


def test_runner_configuration_identity_separates_regularization_tau_and_systematics():
    runner = load_runner_module()
    baseline = runner.parse_args(["--channel", "dijet"])
    baseline_identity = runner.run_configuration_identity(
        baseline, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )
    assert baseline_identity == runner.run_configuration_identity(
        baseline, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )
    assert "regularization-none" in baseline_identity["directory_name"]
    assert "tau-disabled" in baseline_identity["directory_name"]
    assert "normalization-full" in baseline_identity["directory_name"]
    changed_source = {
        **SYNTHETIC_MESS_VINCIA_SOURCE,
        "manifest_ntuple_inventory_sha256": "inventory-b",
    }
    assert baseline_identity["directory_name"] != runner.run_configuration_identity(
        baseline, ("nominal", "JERUp", "JERDown"), changed_source
    )["directory_name"]

    curvature_lcurve = runner.parse_args(
        ["--channel", "dijet", "--regularization", "curvature"]
    )
    fixed_tau = runner.parse_args(
        ["--channel", "dijet", "--regularization", "curvature", "--tau", "0.1"]
    )
    changed_systematics = runner.run_configuration_identity(
        baseline, ("nominal", "JMSUp", "JMSDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )
    window_aligned = runner.parse_args(
        ["--channel", "dijet", "--binning", "window_aligned"]
    )
    window_aligned_coarse = runner.parse_args(
        ["--channel", "dijet", "--binning", "window_aligned_coarse"]
    )
    coarse_tail = runner.parse_args(
        ["--channel", "dijet", "--binning", "coarse_tail"]
    )
    peak_normalization = runner.parse_args(
        ["--channel", "dijet", "--normalization-window", "peak"]
    )
    assert "normalization-peak" in runner.run_configuration_identity(
        peak_normalization,
        ("nominal", "JERUp", "JERDown"),
        SYNTHETIC_MESS_VINCIA_SOURCE,
    )["directory_name"]

    assert baseline_identity["directory_name"] != runner.run_configuration_identity(
        fixed_tau, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"]
    assert baseline_identity["directory_name"] != runner.run_configuration_identity(
        curvature_lcurve, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"]
    assert baseline_identity["directory_name"] != changed_systematics["directory_name"]
    assert baseline_identity["directory_name"] != runner.run_configuration_identity(
        peak_normalization,
        ("nominal", "JERUp", "JERDown"),
        SYNTHETIC_MESS_VINCIA_SOURCE,
    )["directory_name"]
    assert window_aligned.binning == "window_aligned"
    assert window_aligned_coarse.binning == "window_aligned_coarse"
    assert coarse_tail.binning == "coarse_tail"
    assert baseline_identity["directory_name"] != runner.run_configuration_identity(
        window_aligned, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"]
    assert runner.run_configuration_identity(
        window_aligned, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"] != runner.run_configuration_identity(
        window_aligned_coarse, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"]
    assert baseline_identity["directory_name"] != runner.run_configuration_identity(
        coarse_tail, ("nominal", "JERUp", "JERDown"), SYNTHETIC_MESS_VINCIA_SOURCE
    )["directory_name"]


def test_rectangular_reco_to_gen_layout_diagnostics_and_closure_bookkeeping():
    runner = load_runner_module()
    reco_edges = ((-3.0, -2.5, -2.0, -1.0, 0.0),)
    gen_edges = ((-3.0, -2.0, -1.0, 0.0),)
    response = np.array(
        [[5.0, 1.0, 0.0], [3.0, 2.0, 1.0], [0.0, 4.0, 2.0], [1.0, 0.0, 7.0]]
    )
    data_unfolder = SimpleNamespace(
        mosaic=response,
        fakes_2d=np.array([1.0, 2.0, 0.0, 1.0]),
        misses_2d=np.array([2.0, 1.0, 3.0]),
        y_meas=np.array([10.0, 6.0, 7.0, 9.0]),
        corrected_measured_covariance=np.array(
            [[4.0, 0.5, 0.0, 0.0], [0.5, 3.0, 0.0, 0.0],
             [0.0, 0.0, 2.0, 0.2], [0.0, 0.0, 0.2, 2.0]]
        ),
        x_folded=np.array([9.0, 5.0, 5.0, 8.0]),
        cov_np=np.diag([0.5, 0.6, 0.7]),
        y_unf=np.array([1.0, -1.0, 3.0]),
        reco_edges_by_pt=reco_edges,
        gen_edges_by_pt=gen_edges,
        normalized_results=[{"unfolded": np.array([0.0, -1.0 / 2.0, 3.0 / 2.0])}],
        norm_cov_input=np.diag([0.0, 0.2, 0.3]),
        norm_cov_matrix=np.diag([0.0, 0.1, 0.1]),
        norm_cov_stat=np.diag([0.0, 0.3, 0.4]),
        get_total_covariance=lambda: np.diag([0.0, 0.5, 0.6]),
    )
    closure_unfolder = SimpleNamespace(
        y_true=np.array([4.0, 6.0, 8.0]),
        y_unf=np.array([4.4, 5.4, 8.2]),
        tau=0.25,
        pt_edges=(200.0, 300.0),
        gen_edges_by_pt=gen_edges,
    )

    diagnostics = runner.collect_run_diagnostics(
        data_unfolder,
        closure_unfolder,
        normalization_window=(-3.0, 0.0),
        study_core_window=(-2.0, 0.0),
        first_reported_pt_bin=0,
        data_covariance_source="diagonal_reco_sumw2",
    )

    expected_map = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert np.array_equal(diagnostics.arrays["reco_to_gen_layout_map"], expected_map)
    assert np.array_equal(diagnostics.arrays["gen_layout_migration"], expected_map @ response)
    assert diagnostics.metrics["response_raw"]["shape"] == [4, 3]
    assert diagnostics.metrics["response_raw"]["rank"] == 3
    assert diagnostics.metrics["refolded_residual"]["rank"] == 4
    assert diagnostics.metrics["measured_corrected_covariance"]["source"] == "diagonal_reco_sumw2"
    assert diagnostics.metrics["unfolded_bins"]["full"]["negative_bins"] == 1
    assert diagnostics.metrics["unfolded_bins"]["reported_normalization_window"]["negative_bins"] == 1
    assert np.array_equal(diagnostics.arrays["closure_window_mask"], [True, True, True])
    closure = diagnostics.metrics["nominal_mc_self_closure"]
    assert closure["tau"] == pytest.approx(0.25)
    assert closure["raw_bias"]["l1"] == pytest.approx(1.2)
    assert closure["per_pt_window_normalized_bias"][0]["window_bin_count"] == 3
    assert np.isfinite(diagnostics.arrays["closure_window_normalized_bias"][0])
    normalized_total = diagnostics.metrics["normalized_covariances"]["total"]
    assert normalized_total["reported_normalization_window"]["expected_normalization_null_modes"] == 1
    assert {"normalized_result", "norm_cov_input", "norm_cov_matrix", "norm_cov_stat", "norm_cov_total"} <= set(diagnostics.arrays)
    migration = diagnostics.metrics["gen_layout_migration"]["reported_normalization_window"]
    assert migration["per_pt"][0]["window_bin_count"] == 3
    core_migration = diagnostics.metrics["gen_layout_migration"]["study_core_window"]
    assert core_migration["per_pt"][0]["window_bin_count"] == 2
    assert json.loads(json.dumps(diagnostics.metrics))["normalized_covariances"]["total"]


@dataclass(frozen=True)
class ClosureSpec:
    regularization: str
    tau: float | None


class CapturingPreparedUnfolder:
    call = None

    @classmethod
    def from_prepared_inputs(cls, spec, groomed, **kwargs):
        cls.call = {"spec": spec, "groomed": groomed, **kwargs}
        return SimpleNamespace(tau=spec.tau)


def test_nominal_mc_self_closure_uses_prepared_path_and_data_selected_tau():
    runner = load_runner_module()
    prepared = prepare_pairsplit_groomed_inputs(
        synthetic_pairsplit_inputs(), "coarse_tail", ("nominal",)
    )
    n_reco = prepared.measured_covariance.shape[0]
    data_unfolder = SimpleNamespace(
        tau=0.125,
        reco_mc_var_dict={"nominal": np.arange(1.0, n_reco + 1.0)},
    )

    runner.run_nominal_mc_self_closure(
        CapturingPreparedUnfolder,
        data_unfolder,
        spec=ClosureSpec(regularization="curvature", tau=None),
        prepared=prepared,
        cms_label="Internal",
        lumi=138.0,
        com=13.0,
    )

    call = CapturingPreparedUnfolder.call
    assert call["groomed"] is True
    assert call["spec"].tau == pytest.approx(0.125)
    assert call["systematics"] == ("nominal",)
    assert call["data_inputs"][LEGACY_HISTOGRAM_KEYS["groomed"]["reco"]] is prepared.mc_inputs[
        LEGACY_HISTOGRAM_KEYS["groomed"]["reco"]
    ]
    assert np.array_equal(
        call["measured_covariance"],
        np.diag(data_unfolder.reco_mc_var_dict["nominal"]),
    )


def test_artifact_persists_machine_readable_diagnostic_arrays(tmp_path):
    runner = load_runner_module()
    prepared = prepare_pairsplit_groomed_inputs(
        synthetic_pairsplit_inputs(), "coarse_tail", ("nominal",)
    )
    n_reco = prepared.measured_covariance.shape[0]
    n_gen = sum(len(edges) - 1 for edges in prepared.analysis_binning.gen_two_log10_rho_edges_by_pt)
    unfolder = SimpleNamespace(
        mosaic=np.zeros((n_reco, n_gen)),
        mosaic_var_dict={"nominal": np.zeros((n_reco, n_gen))},
        y_unf=np.zeros(n_gen),
        cov_np=np.eye(n_gen),
        cov_data_np=np.eye(n_gen),
        y_unf_dict={},
        tau=0.25,
    )
    diagnostics = runner.RunDiagnostics(
        metrics={},
        arrays={
            "measured_corrected_spectrum": np.arange(n_reco, dtype=float),
            "measured_corrected_covariance": np.eye(n_reco),
            "refolded_spectrum": np.ones(n_reco),
            "closure_raw_bias": np.arange(n_gen, dtype=float),
        },
    )

    vincia_prediction = SimpleNamespace(
        artifact_arrays=lambda: {
            "mess_vincia_density_flat": np.arange(n_gen, dtype=float),
            "mess_vincia_stat_unc_flat": np.ones(n_gen, dtype=float),
            "mess_vincia_bin_offsets": np.array([0, n_gen]),
        }
    )
    artifact = runner.write_artifact(
        unfolder,
        prepared,
        tmp_path,
        diagnostics,
        vincia_prediction,
    )

    with np.load(artifact) as saved:
        assert {"measured_corrected_spectrum", "measured_corrected_covariance", "refolded_spectrum", "closure_raw_bias"} <= set(saved.files)
        assert np.array_equal(saved["closure_raw_bias"], np.arange(n_gen, dtype=float))
        assert np.array_equal(saved["mess_vincia_bin_offsets"], [0, n_gen])
        assert np.array_equal(saved["mess_vincia_density_flat"], np.arange(n_gen))
