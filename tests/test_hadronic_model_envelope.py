from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from unfold.model import (
    compute_prepared_model_shifts,
    group_model_shifts,
)
from unfold.hadronic.model_envelope import (
    _condition_transfer_weight,
    _derive_iterated_weight,
)
from unfold.hadronic.internal_variations import (
    CONFIGS,
    load_hadronic_internal_transfers,
)


def test_raw_internal_ungroomed_harvest_uses_pooled_generated_luminosity(tmp_path):
    campaign = tmp_path / "campaign"
    ht_dir = campaign / "QCD_HT300to500"
    ht_dir.mkdir(parents=True)
    manifest_path = ht_dir / "manifest.json"
    configs = []
    for config in CONFIGS:
        ntuple_name = f"{config}_dijet.txt"
        weights = np.array([2.0, 1.0] if config == "cr1" else [1.0, 1.0])
        rows = np.array(
            [
                [pt, 1.0, 1.0, coordinate, coordinate, weight]
                for pt in (210.0, 300.0, 500.0)
                for coordinate, weight in zip((-2.25, -1.25), weights)
            ]
        )
        np.savetxt(ht_dir / ntuple_name, rows)
        configs.append(
            {
                "config": config,
                "xsec_pb": 10.0,
                "sumw": 20.0,
                "dijet_rows": len(rows),
                "dijet_ntuple": ntuple_name,
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "campaign": "synthetic_internal",
                "mode": "five_million",
                "bin": "QCD_HT300to500",
                "seed": 1,
                "configs": configs,
            }
        )
    )
    final = tmp_path / "final"
    final.mkdir()
    (final / "manifest_snapshot.txt").write_text(f"{manifest_path}\n")
    (final / "campaign_validation.json").write_text(
        json.dumps(
            {
                "complete": True,
                "errors": [],
                "campaign": "synthetic_internal",
                "mode_counts": {"five_million": 1},
            }
        )
    )

    transfers = load_hadronic_internal_transfers(
        "dijet",
        grooming_mode="ungroomed",
        coordinate_edges=(-10.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0),
        campaign_directory=campaign,
        final_directory=final,
        expected_campaign="synthetic_internal",
        required_manifest_count=1,
    )
    for pt_low in (200, 290, 400):
        ratio = transfers.ratio_by_source_and_pt_low["cr1"][pt_low]
        assert ratio[1] == pytest.approx(4.0 / 3.0)
        assert ratio[3] == pytest.approx(2.0 / 3.0)
    assert transfers.provenance["manifest_count"] == 1
    assert len(transfers.provenance["ntuple_inventory_sha256"]) == 64


def test_iterated_weight_closes_populated_bins_and_flat_extrapolates_sparse_tail():
    nominal = np.array([0.1, 10.0, 20.0, 10.0, 0.1])
    target = np.array([0.0, 5.0, 25.0, 10.0, 0.0])
    normalization_mask = np.array([False, True, True, True, False])
    anchors = np.array([False, True, True, True, False])

    weight, metrics = _derive_iterated_weight(
        nominal,
        target,
        normalization_mask=normalization_mask,
        anchors=anchors,
        clip=(0.2, 5.0),
        closure_tolerance=0.01,
        max_iterations=8,
    )

    varied = nominal * weight
    varied_shape = varied / varied[normalization_mask].sum()
    target_shape = target / target[normalization_mask].sum()
    assert np.allclose(varied_shape[anchors], target_shape[anchors])
    assert weight[0] == pytest.approx(weight[1])
    assert weight[-1] == pytest.approx(weight[-2])
    assert metrics["max_relative_closure_residual"] < 1.0e-12


def test_internal_transfer_conditioning_preserves_requested_window_area():
    nominal = np.array([1.0, 10.0, 20.0, 10.0, 1.0])
    transfer = np.array([np.nan, 0.9, 1.2, 1.1, np.nan])
    normalization_mask = np.array([False, True, True, True, False])

    weight, metrics = _condition_transfer_weight(
        nominal,
        transfer,
        normalization_mask=normalization_mask,
        anchor_fraction=0.02,
        clip=(0.2, 5.0),
    )

    assert np.sum(nominal[normalization_mask] * weight[normalization_mask]) == pytest.approx(
        np.sum(nominal[normalization_mask])
    )
    assert np.all(np.isfinite(weight))
    assert np.all(weight > 0.0)
    assert metrics["anchor_count"] == 3


def test_prepared_model_shifts_use_normalized_reunfolded_results_and_group_sources():
    nominal = np.array([0.2, 0.8])
    varied = {
        "model_vincia": np.array([0.18, 0.82]),
        "model_cr1": np.array([0.21, 0.79]),
        "model_cr2": np.array([0.19, 0.81]),
        "model_fraghard": np.array([0.205, 0.795]),
        "model_fragsoft": np.array([0.195, 0.805]),
    }
    unfolder = SimpleNamespace(
        normalized_results=[{"unfolded": nominal}],
        normalized_systematics=[{"unfolded": varied}],
    )

    signed = compute_prepared_model_shifts(unfolder)
    grouped = group_model_shifts(signed, 1)
    assert signed["vincia"][0][0] == pytest.approx(-0.1)
    assert grouped["Vincia"][0][0] == pytest.approx(0.1)
    assert grouped["CR"][0][0] == pytest.approx(0.05)
    assert grouped["frag"][0][0] == pytest.approx(0.025)
