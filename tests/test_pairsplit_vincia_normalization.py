"""Prediction statistics retain normalization-induced correlations."""

from types import SimpleNamespace

import numpy as np

from unfold.tools.pairsplit_vincia import (
    attach_pairsplit_vincia_prediction, derive_pairsplit_vincia_prediction,
)


def _source(coordinates, weights=None):
    coordinates = np.asarray(coordinates)
    pt = np.full(coordinates.size, 250.0)
    mass = pt * 0.8 * 10 ** (coordinates / 2)
    rows = np.column_stack([pt, mass, mass, np.zeros((coordinates.size, 2)), np.ones(coordinates.size)])
    return SimpleNamespace(
        rows_by_ht={"synthetic": rows},
        row_weights_by_ht={"synthetic": np.ones(coordinates.size) if weights is None else np.asarray(weights)},
    )


def test_vincia_unit_weight_bins_have_binomial_normalized_covariance():
    prediction = derive_pairsplit_vincia_prediction(
        _source([-2.5, -2.5, -1.0, -1.0, -1.0]),
        pt_edges=(200, 300), gen_edges_by_pt=((-3, -2, 0),),
        normalization_window=(-3, 0),
    )
    # Five independent unit-weight entries: Var(n1/N)=p1*p2/N.
    # The second density bin is twice as wide as the first.
    expected = (0.4 * 0.6 / 5) * np.array([[1, -0.5], [-0.5, 0.25]])
    np.testing.assert_allclose(prediction.density_by_pt[0], [0.4, 0.3])
    np.testing.assert_array_equal(prediction.sumw_by_pt[0], [2, 3])
    np.testing.assert_array_equal(prediction.sumw2_by_pt[0], [2, 3])
    np.testing.assert_allclose(prediction.stat_covariance_by_pt[0], expected)
    np.testing.assert_allclose(prediction.stat_unc_by_pt[0] ** 2, np.diag(expected))
    np.testing.assert_allclose(expected @ [1, 2], 0, atol=1e-15)
    np.testing.assert_allclose(prediction.artifact_arrays()["mess_vincia_stat_covariance"], expected)
    unfolder = SimpleNamespace(pt_edges=prediction.pt_edges, gen_edges_by_pt=prediction.gen_edges_by_pt)
    attach_pairsplit_vincia_prediction(unfolder, prediction)
    assert unfolder.vincia_stat_covariance_by_pt is prediction.stat_covariance_by_pt


def test_vincia_partial_window_keeps_outside_denominator_fluctuations():
    prediction = derive_pairsplit_vincia_prediction(
        _source([-4.5, -2.5, -1.0], weights=[2, 3, 4]),
        pt_edges=(200, 300), gen_edges_by_pt=((-5, -3, -2, 0),),
        normalization_window=(-3, 0), grooming_mode="ungroomed",
    )
    covariance = prediction.stat_covariance_by_pt[0]
    np.testing.assert_allclose(prediction.density_by_pt[0], [1 / 7, 3 / 7, 2 / 7])
    np.testing.assert_array_equal(prediction.sumw2_by_pt[0], [4, 9, 16])
    # Outside-bin variance includes uncertainty in the denominator 3+4.
    np.testing.assert_allclose(covariance[0, 0], 4 / (2 * 7) ** 2 + 25 / 7 ** 4)
    np.testing.assert_allclose(covariance @ [0, 1, 2], 0, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() > -1e-15
