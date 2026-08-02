import numpy as np
from types import SimpleNamespace

from unfold.tools.unfolder_core import Unfolder


def test_model_covariance_keeps_signed_ps_had_correlations():
    unfolder = Unfolder.__new__(Unfolder)
    unfolder.y_unf = np.array([10.0, 20.0, 30.0])
    unfolder.model_ps_shift_flat = np.array([0.10, -0.20, 0.05])
    unfolder.model_had_shift_flat = np.array([0.03, 0.04, -0.02])

    covariance = unfolder._model_cov_unfolded([0, 1, 2])
    ps_absolute = unfolder.y_unf * unfolder.model_ps_shift_flat
    had_absolute = unfolder.y_unf * unfolder.model_had_shift_flat
    expected = np.outer(ps_absolute, ps_absolute) + np.outer(
        had_absolute, had_absolute
    )

    np.testing.assert_allclose(covariance, expected)
    assert covariance[0, 1] < 0.0
    assert np.linalg.matrix_rank(covariance) <= 2


def test_model_covariance_respects_requested_bin_subset():
    unfolder = Unfolder.__new__(Unfolder)
    unfolder.y_unf = np.array([10.0, 20.0, 30.0])
    unfolder.model_ps_shift_flat = np.array([0.10, -0.20, 0.05])
    unfolder.model_had_shift_flat = np.array([0.03, 0.04, -0.02])

    full = unfolder._model_cov_unfolded([0, 1, 2])
    subset = unfolder._model_cov_unfolded([0, 2])

    np.testing.assert_allclose(subset, full[np.ix_([0, 2], [0, 2])])


def test_per_pt_model_covariance_has_no_cross_pt_terms():
    unfolder = Unfolder.__new__(Unfolder)
    unfolder.spec = SimpleNamespace(model_covariance_scope="per_pt")
    unfolder.y_unf = np.array([10.0, 20.0, 30.0, 40.0])
    unfolder.model_ps_shift_flat = np.ones(4)
    unfolder.model_had_shift_flat = np.ones(4)
    unfolder.model_ps_shifts_by_pt_flat = [
        np.array([0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.2, -0.2]),
    ]
    unfolder.model_had_shifts_by_pt_flat = [
        np.array([0.03, 0.04, 0.0, 0.0]),
        np.array([0.0, 0.0, -0.05, 0.06]),
    ]

    covariance = unfolder._model_cov_unfolded([0, 1, 2, 3])

    np.testing.assert_allclose(covariance[:2, 2:], 0.0)
    np.testing.assert_allclose(covariance[2:, :2], 0.0)
