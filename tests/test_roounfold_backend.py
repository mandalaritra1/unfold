"""Regression tests for the RooUnfold iterative-Bayes backend.

Guards the sqrt(N) bin-error defect (ai-wiki bugs/roounfold_backend_sqrtn_bin_errors):
the measured histogram fed to RooUnfoldBayes must carry the stored sumw2
errors, otherwise the propagated stat covariance on weighted (prescaled) data
is ~sqrt(sumw2/N) too small (~16x on the 2018 dijet data).

Skipped when ROOT/libRooUnfold is not available (they are external, non-pip
dependencies; see scripts/setup_root.sh and scripts/setup_roounfold.sh).
"""
from __future__ import annotations

import unittest

import numpy as np


def _roounfold_available():
    try:
        from unfold.tools.roounfold_backend import load_roounfold

        load_roounfold()
        return True
    except Exception:
        return False


@unittest.skipUnless(_roounfold_available(), "ROOT/libRooUnfold not available")
class BayesUnfoldBinErrorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.response = rng.uniform(0.1, 1.0, (8, 5)) * 200.0
        self.truth = self.response.sum(axis=0) + 15.0
        self.measured = self.response.sum(axis=1)

    def _unfold(self, **kwargs):
        from unfold.tools.roounfold_backend import bayes_unfold

        return bayes_unfold(
            self.response, self.measured, self.truth, n_iter=4, **kwargs
        )

    def test_unweighted_variances_match_legacy_poisson(self):
        y0, e0 = self._unfold()
        y1, e1 = self._unfold(measured_variances=self.measured)
        np.testing.assert_allclose(y1, y0)
        np.testing.assert_allclose(e1, e0)

    def test_weighted_variances_scale_errors_not_central(self):
        scale = 100.0  # weighted data: sumw2/N = 100 -> errors x10
        y1, e1 = self._unfold(measured_variances=self.measured)
        y2, e2 = self._unfold(measured_variances=scale * self.measured)
        np.testing.assert_allclose(y2, y1)
        np.testing.assert_allclose(e2, np.sqrt(scale) * e1, rtol=1e-6)

    def test_covariance_carries_weighted_variances(self):
        scale = 100.0
        _, e1, cov1 = self._unfold(
            with_covariance=True, measured_variances=self.measured
        )
        _, _, cov2 = self._unfold(
            with_covariance=True, measured_variances=scale * self.measured
        )
        np.testing.assert_allclose(cov2, scale * cov1, rtol=1e-6)
        np.testing.assert_allclose(
            np.sqrt(np.diag(cov2)), np.sqrt(scale) * e1, rtol=1e-6
        )

    def test_poisson_floor(self):
        # Stored variances below the Poisson expectation are floored at
        # sqrt(|content|) per bin (guards against zero-variance bins).
        from unfold.tools.roounfold_backend import _np_to_th1

        values = np.array([4.0, 9.0])
        variances = np.array([0.0, 100.0])
        h = _np_to_th1(values, "hFloorTest", variances=variances)
        self.assertAlmostEqual(h.GetBinError(1), 2.0)  # floored at sqrt(4)
        self.assertAlmostEqual(h.GetBinError(2), 10.0)  # sqrt(sumw2)


if __name__ == "__main__":
    unittest.main()
