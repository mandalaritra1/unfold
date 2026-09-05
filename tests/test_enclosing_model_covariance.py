import numpy as np
import pytest

from unfold.tools.model_covariance import (
    MODEL_GROUPS, enclosing_template_covariance, two_group_model_covariance,
)


def test_collinear_alternatives_do_not_add_in_quadrature():
    shift = np.array([0.2, -0.1, 0.3])
    matrix, info = enclosing_template_covariance(np.column_stack([shift, -shift, 0.5 * shift]))
    np.testing.assert_allclose(matrix, np.outer(shift, shift), rtol=1e-7, atol=1e-14)
    assert info["rank"] == 1
    assert max(info["mahalanobis_squared"]) <= 1.0 + 1e-12


def test_independent_directions_and_normalization_null():
    shifts = np.array([[0.2, 0.0], [-0.2, 0.3], [0.0, -0.3]])
    matrix, info = enclosing_template_covariance(shifts)
    np.testing.assert_allclose(matrix, shifts @ shifts.T, atol=1e-14)
    np.testing.assert_allclose(matrix @ np.ones(3), 0, atol=1e-14)
    assert info["rank"] == 2
    assert np.linalg.eigvalsh(matrix).min() > -1e-14


def test_duplicates_opposites_permutation_and_zero_do_not_change_ellipsoid():
    shifts = np.array([[1.0, 0.0, 0.6], [0.0, 2.0, 0.4], [-1.0, -2.0, -1.0]])
    original, _ = enclosing_template_covariance(shifts)
    repeated = np.column_stack([shifts[:, 2], -shifts, shifts, np.zeros(3)])
    changed, _ = enclosing_template_covariance(repeated)
    np.testing.assert_allclose(changed, original, rtol=2e-6, atol=1e-8)


def test_affine_coordinate_change_and_full_template_containment():
    shifts = np.array([[1.0, 0.0, 0.7], [0.0, 1.0, 0.7]])
    original, _ = enclosing_template_covariance(shifts)
    transform = np.array([[2.0, 0.4], [0.0, 0.1]])
    changed, info = enclosing_template_covariance(transform @ shifts)
    np.testing.assert_allclose(changed, transform @ original @ transform.T, rtol=2e-6, atol=1e-9)
    assert info["span_relative_residual"] < 1e-12
    for shift in (transform @ shifts).T:
        assert shift @ np.linalg.solve(changed, shift) <= 1 + 2e-6


def test_zero_and_invalid_groups():
    matrix, info = enclosing_template_covariance(np.zeros((5, 3)))
    assert not matrix.any() and info["rank"] == 0
    with pytest.raises(ValueError):
        enclosing_template_covariance(np.array([[np.nan]]))


def test_two_groups_preserve_density_normalization_and_do_not_drop_zero_bins():
    nominal = np.array([0.0, 0.5, 0.5])
    weights = np.array([[2.0, 1.0, 1.0]])
    ps = np.array([0.02, -0.04, 0.0])
    had = np.array([0.0, 0.03, -0.03])
    varied = {name: nominal + (ps if group == 'parton_shower' else had)
              for group, names in MODEL_GROUPS.items() for name in names}
    matrices, _ = two_group_model_covariance(nominal, varied, weights)
    total = sum(matrices.values())
    np.testing.assert_allclose(total, np.outer(ps, ps) + np.outer(had, had), atol=1e-12)
    np.testing.assert_allclose(weights @ total, 0, atol=1e-14)
    assert total[0, 0] > 0  # Absolute shifts still exist when nominal is zero.
    varied['fsrUp'] = nominal + 0.01
    with pytest.raises(ValueError, match='normalization'):
        two_group_model_covariance(nominal, varied, weights)
