"""Method 2: POSITIVE REGULARIZED FIT (NNLS-type, independent of Bayes).

min || (P x - m) / sigma ||^2 + tau^2 || L_ratio x ||^2   s.t.  x >= 0

L_ratio is a per-pt-slice second-difference operator scaled by the Pythia prior
x0 so it penalizes curvature of the *ratio* x/x0 (rows
(1/x0[i-1], -2/x0[i], 1/x0[i+1])). Solved by stacking the weighted fold and the
penalty and calling scipy.optimize.nnls on the stacked system.

Numerical note (documented in the README): the raw measured counts span ~8
orders of magnitude (200 -> 4e8), so weighting rows by 1/sigma with the 2% model
floor makes the stacked system catastrophically ill-conditioned
(cond ~ 1e21) and NNLS collapses many bins to the positivity boundary. We
therefore row-normalize by the data (fit fold/m ~ 1 with weight m/sigma ~ 1/0.02),
which keeps cond ~ 1e6 and eliminates the spurious zeros -- P itself is full-rank
with cond ~ 500, so this is purely a scaling fix, not a change to the physics.

tau scanned over logspace(-2, 2, 9); we pick the smallest tau whose oscillation
metric (median |second difference of x/x0| over interior gen bins) is below
threshold, then report its central values vs D'Agostini n=4.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from scripts.dijet_methods.study2_indep_common import MODEL_FLOOR_FRAC
from scripts.dijet_methods import methods as M


TAU_GRID = np.logspace(-2, 2, 9)


def _ratio_curvature_matrix(prob, x0):
    """L_ratio rows within each pt slice, scaled by the Pythia prior x0."""
    Ngen = prob.P.shape[1]
    rows = []
    for ptg in np.unique(prob.gen_pt_idx):
        idx = np.where(prob.gen_pt_idx == ptg)[0]
        for k in range(1, len(idx) - 1):
            im, i0, ip = idx[k - 1], idx[k], idx[k + 1]
            if x0[im] <= 0 or x0[i0] <= 0 or x0[ip] <= 0:
                continue
            row = np.zeros(Ngen)
            row[im] = 1.0 / x0[im]
            row[i0] = -2.0 / x0[i0]
            row[ip] = 1.0 / x0[ip]
            rows.append(row)
    return np.array(rows) if rows else np.zeros((0, Ngen))


def _oscillation_metric(prob, x, x0):
    """median |second difference of the ratio x/x0| over interior gen bins.

    A smooth solution that tracks the prior has |d2(r)| ~ 0; genuine ringing
    (the failure mode positivity clips into zeros) drives it up. Robust to a
    ratio that is merely non-monotonic but smooth."""
    r = np.divide(x, x0, out=np.ones_like(x), where=x0 > 0)
    vals = []
    for ptg in np.unique(prob.gen_pt_idx):
        idx = np.where(prob.gen_pt_idx == ptg)[0]
        rr = r[idx]
        if len(rr) < 3:
            continue
        d2 = rr[2:] - 2 * rr[1:-1] + rr[:-2]
        vals.extend(np.abs(d2).tolist())
    return float(np.median(vals)) if vals else 0.0


def _weighted_system(prep):
    """Row-normalized weighted fold: A = (P/m)*w, b = 1*w, w = m/sigma."""
    prob = prep.prob
    P = prob.P
    m = prep.data_matched
    var = prep.data_var
    f = M.fake_fraction(prob)
    var_matched = var * (1.0 - f) ** 2
    sigma = np.sqrt(np.clip(var_matched + (MODEL_FLOOR_FRAC * m) ** 2, 1e-30, None))
    ok = m > 0
    w = np.where(ok, m / sigma, 0.0)
    A = np.where(ok[:, None], P / np.where(ok, m, 1.0)[:, None], 0.0) * w[:, None]
    b = np.where(ok, 1.0, 0.0) * w
    return A, b


def solve_posreg(prep, tau):
    prob = prep.prob
    x0 = prob.gen.astype(float).copy()
    A_fold, b_fold = _weighted_system(prep)
    L = _ratio_curvature_matrix(prob, x0)
    A = np.vstack([A_fold, tau * L])
    b = np.concatenate([b_fold, np.zeros(L.shape[0])])
    x, _ = nnls(A, b, maxiter=100 * A.shape[1])
    return x


def run_posreg(prep, osc_thresh=0.05):
    """Scan tau; pick the smallest tau with oscillation metric < osc_thresh.
    Returns (x_chosen, chosen_tau, scan)."""
    prob = prep.prob
    x0 = prob.gen.astype(float)
    scan = []
    for tau in TAU_GRID:
        x = solve_posreg(prep, tau)
        osc = _oscillation_metric(prob, x, x0)
        neg = int(np.sum(x < 0))  # nnls guarantees >= 0; kept as a record
        scan.append({"tau": float(tau), "osc": float(osc), "neg": neg, "x": x})
    chosen = None
    for s in scan:
        if s["osc"] < osc_thresh:
            chosen = s
            break
    if chosen is None:  # fall back to the smoothest available (largest tau ok)
        chosen = min(scan, key=lambda s: s["osc"])
    return chosen["x"], chosen["tau"], scan
