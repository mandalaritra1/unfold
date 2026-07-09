"""Shared helpers for the independent-methods cross-check study (study2_indep).

Two genuinely independent central-value cross-checks of the dijet rho
D'Agostini (iterative Bayes) unfolding, plus a forensic audit of the Bayes
tag's quoted statistical uncertainties.

Everything here reuses the read-only foundation modules
``scripts.dijet_methods.{loader, methods, result}`` and NEVER floors
efficiencies.
"""
from __future__ import annotations

import numpy as np

from scripts.dijet_methods.loader import Binning

# --------------------------------------------------------------------------
# The study binning (smoke-tested; see task spec). pt<200 is a migration sink.
# --------------------------------------------------------------------------
PT_EDGES = np.array([0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0])
RHO_EDGES_RECO = np.array(
    [-10.0, -8.0, -7.0, -6.0, -5.5, -5.0, -4.75, -4.5, -4.25, -4.0, -3.75,
     -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0,
     -0.75, -0.5, -0.25, 0.0]
)
RHO_EDGES_GEN = np.array(
    [-10.0, -6.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
     -0.5, 0.0]
)

# pt bins actually reported (drop the 0-200 sink at index 0 and the last
# reference open bin is kept; report bins 1..5 per the spec).
REPORTED_PT = [1, 2, 3, 4, 5]

# 2% model-systematic floor added in quadrature to the fit chi2 so the
# forward-fold fit does not chase sub-percent reco structure (documented in
# the README).
MODEL_FLOOR_FRAC = 0.02


def study_binning() -> Binning:
    return Binning(PT_EDGES.copy(), RHO_EDGES_RECO.copy(), RHO_EDGES_GEN.copy())


def gen_rho_centers():
    """Log10(rho^2) bin centers for the gen rho binning."""
    e = RHO_EDGES_GEN
    return 0.5 * (e[:-1] + e[1:])


def slice_gen(prob, pt_bin):
    """Indices of gen bins belonging to a given pt bin (in unroll order)."""
    return np.where(prob.gen_pt_idx == pt_bin)[0]


def slice_reco(prob, pt_bin):
    return np.where(prob.reco_pt_idx == pt_bin)[0]


def normalized_ratio_metric(a, b, mask=None):
    """median/max |a/b - 1| over a (optionally masked) set of bins, after
    normalizing a and b to unit area over that set. Returns (median, max)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if mask is None:
        mask = np.ones_like(a, bool)
    mask = mask & (a > 0) & (b > 0)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    an = a[mask] / a[mask].sum()
    bn = b[mask] / b[mask].sum()
    r = np.abs(an / bn - 1.0)
    return float(np.median(r)), float(np.max(r))


def per_pt_normalized_ratio(prob, a, b):
    """Per reported pt slice: normalized median/max |ratio-1|. Returns dict."""
    out = {}
    for pt in REPORTED_PT:
        idx = slice_gen(prob, pt)
        med, mx = normalized_ratio_metric(a[idx], b[idx])
        out[pt] = {"median_pct": 100 * med, "max_pct": 100 * mx}
    return out
