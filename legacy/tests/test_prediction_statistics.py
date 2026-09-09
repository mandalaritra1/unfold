from types import SimpleNamespace

import numpy as np
import pytest

from unfold.tools.prediction_statistics import normalized_prediction_covariance
from unfold.tools.unfolder_core import Unfolder


def test_covariance_matches_finite_difference_with_correlated_counts():
    counts = np.array([20., 30., 10.])
    widths = np.array([0.25, 0.5, 2.])
    mask = np.array([True, True, False])
    covariance = np.array([[20., 3., 1.], [3., 30., 2.], [1., 2., 10.]])
    def density(values):
        return values / (widths * values[mask].sum())
    step = 1e-3
    jacobian = np.column_stack([
        (density(counts + step*direction) - density(counts - step*direction))/(2*step)
        for direction in np.eye(3)])
    actual = normalized_prediction_covariance(counts, covariance, widths, mask)
    np.testing.assert_allclose(actual, jacobian @ covariance @ jacobian.T, rtol=1e-8)
    np.testing.assert_allclose(actual @ (widths * mask), 0, atol=1e-16)
    np.testing.assert_allclose(actual, normalized_prediction_covariance(
        counts*7, covariance*49, widths, mask), atol=1e-16)


def test_zero_nominal_bin_retains_its_statistical_variance():
    result = normalized_prediction_covariance(
        [100., 0.], np.diag([100., 4.]), [1., 1.], [True, True])
    np.testing.assert_allclose(result, [[.0004, -.0004], [-.0004, .0004]])


def prediction_state(method):
    obj = Unfolder.__new__(Unfolder)
    obj.spec = SimpleNamespace(prediction_stat_method=method)
    obj.gen_edges_by_pt = [np.array([0., 1., 2.])]
    obj._shown_gen_mask = lambda i: np.ones(2, dtype=bool)
    obj.gen_mc_flat_dict = {'nominal': np.array([100., 100.])}
    obj.gen_mc_var_dict = {'nominal': np.array([100., 100.])}
    obj._pythia_gen_theory_band = lambda i: (None, None)
    return obj


def test_prepared_prediction_band_and_chi2_share_normalized_covariance():
    obj = prediction_state('jacobian')
    expected = np.array([[.00125, -.00125], [-.00125, .00125]])
    np.testing.assert_allclose(obj._prediction_chi2_covariance(0), expected)
    up, down = obj._prediction_uncertainty(0)
    np.testing.assert_allclose(up**2, np.diag(expected))
    np.testing.assert_array_equal(up, down)


def test_frozen_legacy_prediction_keeps_fixed_normalization():
    obj = prediction_state('fixed_normalization')
    # Legacy prepared path had no prediction-stat component.
    assert obj._prediction_stat_covariance(0) is None
    obj.pythia_gen_val_flat = np.array([100., 100.])
    obj.pythia_gen_var_flat = np.array([100., 100.])
    np.testing.assert_allclose(obj._prediction_chi2_covariance(0), np.diag([.0025, .0025]))


def test_invalid_normalization_fails():
    with pytest.raises(ValueError, match='positive'):
        normalized_prediction_covariance([0., 0.], np.eye(2), [1., 1.], [True, True])


def test_prepared_self_closure_does_not_add_independent_truth_statistics():
    obj = prediction_state('jacobian')
    obj.closure = True
    assert obj._prediction_stat_covariance(0) is None
