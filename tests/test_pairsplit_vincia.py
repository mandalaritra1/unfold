from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from unfold.tools.pairsplit_vincia import (
    PairSplitVinciaPrediction,
    PairSplitVinciaSource,
    PairSplitVinciaValidationError,
    _rebin_histogram_to_target_edges,
    attach_pairsplit_vincia_prediction,
    derive_pairsplit_vincia_prediction,
    load_pairsplit_vincia_source,
    validate_compiled_reference,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_final_all_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Create pooled HT slices, including manifest-dependent cross sections."""

    campaign = root / "campaign"
    manifests: list[dict[str, object]] = []
    audit_rows: dict[str, int] = {}
    fixture = {
        # The first HT slice has a shared denominator of 200 LHE but two
        # different input cross sections. This is the live-campaign contract.
        "QCD_HT300to500": (
            (1, 10.0, 100, np.array([[210.0, 10.0, 8.0, 0.0, 0.0, 2.0], [450.0, 10.0, 18.0, 0.0, 0.0, 1.0]])),
            (2, 30.0, 100, np.array([[220.0, 10.0, 35.2, 0.0, 0.0, 2.0], [500.0, 10.0, 80.0, 0.0, 0.0, 2.0]])),
        ),
        "QCD_HT500to700": (
            (3, 20.0, 200, np.array([[220.0, 10.0, 35.2, 0.0, 0.0, 3.0], [500.0, 10.0, 80.0, 0.0, 0.0, 3.0]])),
        ),
    }
    for ht_bin, ht_entries in fixture.items():
        directory = campaign / ht_bin
        directory.mkdir(parents=True)
        audit_rows[ht_bin] = 0
        for seed, xsec, lhe_events, rows in ht_entries:
            ntuple_name = f"ntuple_{ht_bin}_mess_{seed}_dijet.txt"
            ntuple_path = directory / ntuple_name
            np.savetxt(ntuple_path, rows)
            manifest_path = directory / f"manifest_{ht_bin}_mess_{seed}.json"
            manifest = {
                "complete": True,
                "campaign": "synthetic_mess",
                "phase": "full",
                "selection": "CMS_HADRONIC_PAIR_SPLIT",
                "bin": ht_bin,
                "seed": seed,
                "lhe_events": lhe_events,
                "input_xsec_pb": xsec,
                "dijet_ntuple": ntuple_name,
                "dijet_rows": len(rows),
                "dijet_selected_events": 1,
                "files_sha256": {ntuple_name: _sha256(ntuple_path)},
            }
            manifest_path.write_text(json.dumps(manifest))
            manifests.append(
                {
                    # This deliberately does not exist: the loader must resolve it
                    # through the supplied audited campaign mirror.
                    "path": str(root / "unmounted" / ht_bin / manifest_path.name),
                    "sha256": _sha256(manifest_path),
                    "bin": ht_bin,
                    "phase": "full",
                    "seed": seed,
                    "lhe_events": lhe_events,
                }
            )
            audit_rows[ht_bin] += len(rows)
    audit_path = root / "final_all_audit.json"
    audit = {
        "valid": True,
        "errors": [],
        "campaign": "synthetic_mess",
        "valid_manifests": len(manifests),
        "selected_phases": ["full"],
        "dijet_rows_by_bin": audit_rows,
    }
    audit_path.write_text(json.dumps(audit))
    allowlist_path = root / "final_all_allowlist.json"
    allowlist = {
        "campaign": "synthetic_mess",
        "manifest_count": len(manifests),
        "audit_sha256": _sha256(audit_path),
        "manifests": manifests,
    }
    allowlist_path.write_text(json.dumps(allowlist))
    return allowlist_path, audit_path, campaign


def test_final_all_source_is_hash_checked_and_uses_per_ht_lhe_normalization(tmp_path):
    allowlist, audit, campaign = _write_final_all_fixture(tmp_path)
    source = load_pairsplit_vincia_source(
        "dijet",
        allowlist_path=allowlist,
        audit_path=audit,
        campaign_directory=campaign,
    )

    assert source.per_ht["QCD_HT300to500"]["lhe_events"] == 200
    assert source.per_ht["QCD_HT300to500"]["input_xsec_pb_min"] == 10.0
    assert source.per_ht["QCD_HT300to500"]["input_xsec_pb_max"] == 30.0
    assert np.allclose(source.row_weights_by_ht["QCD_HT300to500"], [0.1, 0.05, 0.3, 0.3])
    assert source.identity_payload()["normalization"] == (
        "per row: rows[:,5] * manifest_input_xsec_pb / sum_ht_lhe_events"
    )

    prediction = derive_pairsplit_vincia_prediction(
        source,
        pt_edges=(200.0, 400.0, 13000.0),
        gen_edges_by_pt=((-4.0, -2.0, -1.0, 0.0),) * 2,
    )
    # HT300--500 has a pooled 200-LHE denominator: 2*10/200=0.1,
    # 1*10/200=0.05, and 2*30/200=0.3. HT500--700 contributes 0.3.
    assert np.allclose(prediction.sumw_by_pt[0], [0.1, 0.6, 0.0])
    assert np.allclose(prediction.sumw_by_pt[1], [0.05, 0.6, 0.0])
    assert np.allclose(prediction.density_by_pt[0], [1.0 / 14.0, 6.0 / 7.0, 0.0])
    # For p=n1/(n1+n2), Var(p)=(n2² V1+n1² V2)/(n1+n2)^4.
    # Density bin 1 has width 2; the two fractions fluctuate oppositely.
    fraction_sigma = np.sqrt(0.6 ** 2 * 0.05 ** 2 + 0.05 ** 2 * 0.18) / 0.65 ** 2
    assert np.allclose(prediction.stat_unc_by_pt[1], [fraction_sigma / 2, fraction_sigma, 0.0])
    arrays = prediction.artifact_arrays()
    assert np.allclose(
        arrays["mess_vincia_density_flat"],
        [1.0 / 14.0, 6.0 / 7.0, 0.0, 1.0 / 26.0, 12.0 / 13.0, 0.0],
    )
    assert np.array_equal(arrays["mess_vincia_bin_offsets"], [0, 3, 6])
    assert np.array_equal(arrays["mess_vincia_gen_edge_offsets"], [0, 4, 8])

    class FakeUnfolder:
        pt_edges = np.array([200.0, 400.0, 13000.0])
        gen_edges_by_pt = ((-4.0, -2.0, -1.0, 0.0),) * 2

    unfolder = FakeUnfolder()
    attach_pairsplit_vincia_prediction(unfolder, prediction)
    assert unfolder.pairsplit_vincia_required is True
    assert unfolder.pairsplit_vincia_prediction.truth_by_pt()[0][0][1] == pytest.approx(6.0 / 7.0)


def test_ungroomed_prediction_uses_m_u_and_skips_groomed_compiled_regression(tmp_path):
    allowlist, audit, campaign = _write_final_all_fixture(tmp_path)
    source = load_pairsplit_vincia_source(
        "dijet",
        allowlist_path=allowlist,
        audit_path=audit,
        campaign_directory=campaign,
    )
    prediction = derive_pairsplit_vincia_prediction(
        source,
        pt_edges=(200.0, 13000.0),
        gen_edges_by_pt=((-4.0, -2.5, -2.0, -1.0, 0.0),),
        normalization_window=(-2.5, 0.0),
        grooming_mode="ungroomed",
    )
    assert prediction.grooming_mode == "ungroomed"
    assert prediction.provenance_payload()["grooming_mode"] == "ungroomed"
    validation = validate_compiled_reference(
        source,
        target_prediction=prediction,
        reference_path=tmp_path / "unused_groomed_reference.npz",
    )
    assert validation["applies"] is False
    assert "groomed coordinates only" in validation["reason"]


def test_final_all_loader_fails_closed_when_an_ntuple_hash_changes(tmp_path):
    allowlist, audit, campaign = _write_final_all_fixture(tmp_path)
    changed_ntuple = next(campaign.glob("*/*_dijet.txt"))
    changed_ntuple.write_text("210 10 8 0 0 2\n")

    with pytest.raises(PairSplitVinciaValidationError, match="ntuple hash mismatch"):
        load_pairsplit_vincia_source(
            "dijet",
            allowlist_path=allowlist,
            audit_path=audit,
            campaign_directory=campaign,
        )


def test_compiled_reference_rebinning_requires_nested_coordinate_edges():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    source_edges = np.array([-10.0, -5.0, -4.0, -2.0, 0.0])
    assert np.allclose(
        _rebin_histogram_to_target_edges(values, source_edges, np.array([-10.0, -4.0, 0.0])),
        [3.0, 7.0],
    )
    with pytest.raises(PairSplitVinciaValidationError, match="not nested"):
        _rebin_histogram_to_target_edges(values, source_edges, np.array([-10.0, -3.0, 0.0]))


def test_trijet_compiled_reference_rebins_each_pt_slice_to_active_truth_edges(tmp_path):
    source_edges = np.array([-10.0, -5.0, -4.0, -2.0, 0.0])
    source_sumw = np.array([1.0, 2.0, 3.0, 4.0])
    source_sumw2 = np.array([10.0, 20.0, 30.0, 40.0])
    arrays = {}
    for pt_low in (200, 290, 400):
        prefix = f"mess_trijet_groomed_rho_pt{pt_low}"
        arrays[f"{prefix}_edges"] = source_edges
        arrays[f"{prefix}_sumw"] = source_sumw
        arrays[f"{prefix}_sumw2"] = source_sumw2
    reference_path = tmp_path / "model_comparison_arrays.npz"
    np.savez(reference_path, **arrays)
    source = PairSplitVinciaSource(
        channel="trijet",
        campaign="synthetic",
        campaign_directory=tmp_path,
        allowlist_path=tmp_path / "allowlist.json",
        audit_path=tmp_path / "audit.json",
        allowlist_sha256="allowlist",
        audit_sha256="audit",
        manifest_ntuple_inventory_sha256="inventory",
        rows_by_ht={},
        row_weights_by_ht={},
        per_ht={},
    )
    target_edges = (-10.0, -4.0, -2.0, 0.0)
    target_sumw = np.array([3.0, 3.0, 4.0])
    target_sumw2 = np.array([30.0, 30.0, 40.0])
    prediction = PairSplitVinciaPrediction(
        source=source,
        pt_edges=(200.0, 290.0, 400.0, 13000.0),
        gen_edges_by_pt=(target_edges,) * 3,
        density_by_pt=(target_sumw,) * 3,
        stat_unc_by_pt=(target_sumw,) * 3,
        stat_covariance_by_pt=(np.diag(target_sumw2),) * 3,
        sumw_by_pt=(target_sumw,) * 3,
        sumw2_by_pt=(target_sumw2,) * 3,
        normalization_totals_by_pt=(10.0,) * 3,
        normalization_masks_by_pt=(np.ones(3, dtype=bool),) * 3,
    )

    result = validate_compiled_reference(
        source,
        target_prediction=prediction,
        reference_path=reference_path,
    )
    assert result["channel"] == "trijet"
    assert len(result["source_coordinate_rebin_by_pt"]) == 3
    assert result["source_coordinate_rebin_by_pt"][2]["max_abs_sumw2_difference"] == 0.0
