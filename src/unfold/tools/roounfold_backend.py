"""D'Agostini (iterative Bayes) unfolding via RooUnfold.

This is the CMS-recommended iterative-Bayes pathway, kept deliberately thin and
backend-isolated: it takes the *same* prepared inputs the TUnfold path uses (a
flattened response matrix, a fake-corrected measured spectrum, and the MC truth
projection including misses) and returns the unfolded vector. Because the
``Unfolder`` jackknife loop only varies those inputs and re-unfolds, the same
jackknife replicas flow through here unchanged -- the statistical uncertainty
stays jackknife-based, exactly as in the TUnfold path.

RooUnfold is an external dependency (not pip). Build it once and point
``UNFOLD_ROOUNFOLD_LIB`` at the shared library (see scripts/setup_roounfold.sh).

Response/efficiency/fakes mapping onto ``RooUnfoldResponse(hMeas, hTruth, hResp)``:
- ``hResp[r, t]``   = response_np[r, t]                 (reco x truth migrations)
- ``hTruth[t]``     = response_np.sum(reco)[t] + misses (so RooUnfold's
                       efficiency_t = matched_t / truth_t reproduces the misses)
- ``hMeas[r]``      = response_np.sum(truth)[r]         (defines fakes = 0, since
                       the measured data is fed already fake-corrected)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_LOADED = False


def load_roounfold():
    """Load libRooUnfold into PyROOT once. Returns the ROOT module."""
    global _LOADED
    import ROOT

    if _LOADED:
        return ROOT
    lib = os.environ.get(
        "UNFOLD_ROOUNFOLD_LIB", os.path.expanduser("~/opt/RooUnfold/libRooUnfold")
    )
    # gSystem.Load resolves the platform suffix (.so/.dylib) itself.
    status = ROOT.gSystem.Load(lib)
    if status < 0:
        raise RuntimeError(
            f"Could not load RooUnfold from {lib!r} (gSystem.Load -> {status}). "
            "Build it and/or set UNFOLD_ROOUNFOLD_LIB; see scripts/setup_roounfold.sh."
        )
    if not hasattr(ROOT, "RooUnfoldBayes"):
        raise RuntimeError(
            f"Loaded {lib!r} but RooUnfoldBayes is not available -- check the build."
        )
    _LOADED = True
    return ROOT


def _np_to_th1(values, name):
    import ROOT

    n = len(values)
    h = ROOT.TH1D(name, "", n, 0.0, float(n))
    for i, v in enumerate(values):
        h.SetBinContent(i + 1, float(v))
        h.SetBinError(i + 1, float(np.sqrt(abs(v))))
    return h


def _np_to_th2(matrix, name):
    import ROOT

    n_reco, n_true = matrix.shape
    h = ROOT.TH2D(name, "", n_reco, 0.0, float(n_reco), n_true, 0.0, float(n_true))
    for r in range(n_reco):
        for t in range(n_true):
            v = float(matrix[r, t])
            h.SetBinContent(r + 1, t + 1, v)
            h.SetBinError(r + 1, t + 1, float(np.sqrt(abs(v))))
    return h


def bayes_unfold(
    response_np,
    measured_flat,
    truth_flat,
    n_iter=4,
    *,
    with_covariance=False,
    tag="rufold",
):
    """Iterative-Bayes unfold one spectrum.

    Parameters
    ----------
    response_np : (n_reco, n_true) array -- reco x truth migration counts.
    measured_flat : (n_reco,) array -- measured data, already fake-corrected.
    truth_flat : (n_true,) array -- MC truth projection including misses
        (response_np.sum(axis=0) + misses); sets the efficiency and the prior.
    n_iter : int -- number of D'Agostini iterations (the regularization).
    with_covariance : bool -- if True also return the n_true x n_true covariance.

    Returns
    -------
    (y_unf, ye_unf) or (y_unf, ye_unf, cov) -- the unfolded truth vector, its
    per-bin error, and optionally the full covariance matrix.
    """
    ROOT = load_roounfold()

    response_np = np.asarray(response_np, dtype=float)
    measured_flat = np.asarray(measured_flat, dtype=float)
    truth_flat = np.asarray(truth_flat, dtype=float)
    n_reco, n_true = response_np.shape

    # Unique names so repeated calls (jackknife) do not collide in gDirectory.
    bayes_unfold._counter = getattr(bayes_unfold, "_counter", 0) + 1
    sfx = f"{tag}_{bayes_unfold._counter}"

    h_resp = _np_to_th2(response_np, f"hResp_{sfx}")
    h_truth = _np_to_th1(truth_flat, f"hTruth_{sfx}")
    h_meas_train = _np_to_th1(response_np.sum(axis=1), f"hMeasTrain_{sfx}")
    h_data = _np_to_th1(measured_flat, f"hData_{sfx}")

    response = ROOT.RooUnfoldResponse(h_meas_train, h_truth, h_resp)
    response.UseOverflow(False)

    bayes = ROOT.RooUnfoldBayes(response, h_data, int(n_iter))
    bayes.SetVerbose(-1)

    err_mode = ROOT.RooUnfold.kCovariance if with_covariance else ROOT.RooUnfold.kErrors
    h_unf = bayes.Hunfold(err_mode)

    y_unf = np.array([h_unf.GetBinContent(i + 1) for i in range(n_true)])
    ye_unf = np.array([h_unf.GetBinError(i + 1) for i in range(n_true)])

    if not with_covariance:
        return y_unf, ye_unf

    m = bayes.Eunfold(ROOT.RooUnfold.kCovariance)
    cov = np.array([[m(i, j) for j in range(n_true)] for i in range(n_true)])
    return y_unf, ye_unf, cov
