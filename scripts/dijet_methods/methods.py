"""Unfolding methods (pure numpy/scipy) for the dijet rho bake-off.

All methods operate on the unrolled Problem from loader.build_problem and the
folding probability matrix P[reco, gen] = R[reco,gen] / gen_total[gen]
(column sums = efficiency). They take a *matched* reco vector (fakes already
subtracted) and return an efficiency-corrected truth estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


# --------------------------------------------------------------------------
# Data / fakes preparation
# --------------------------------------------------------------------------
def fake_fraction(prob):
    """Per reco bin fake fraction from MC = fakes / (matched + fakes)."""
    tot = prob.R.sum(axis=1) + prob.fakes
    return np.divide(prob.fakes, tot, out=np.zeros_like(prob.fakes), where=tot > 0)


def subtract_fakes(measured, prob):
    """Return matched reco = measured * (1 - fake_fraction_MC)."""
    return measured * (1.0 - fake_fraction(prob))


def fold(prob, truth):
    """Fold a truth vector to expected matched reco."""
    return prob.P @ truth


# --------------------------------------------------------------------------
# 1. D'Agostini iterative Bayesian unfolding (with efficiency)
# --------------------------------------------------------------------------
def dagostini(prob, measured_matched, n_iter=4, prior=None, return_all=False,
              eff_floor=0.0):
    """Iterative Bayes. prior defaults to the MC gen truth spectrum.

    Never inverts P, so it is robust to rank-deficiency / empty bins. With
    eff_floor=0 the true-prior fixed point is preserved exactly (machine-
    precision self-closure at every iteration); empty (eff==0) bins carry no
    information and are returned as 0. Low-eff bins amplify noise honestly (no
    bias) -- tame them by *binning*, not by flooring the efficiency.
    """
    P = prob.P                       # (Nreco, Ngen), col sum = eff
    eff = prob.eff.copy()
    Ngen = P.shape[1]
    if prior is None:
        prior = prob.gen.copy()
    prior = np.where(prior > 0, prior, prior[prior > 0].min() * 1e-6 if np.any(prior > 0) else 1.0)
    n = prior.copy()
    history = [n.copy()]
    eff_use = np.clip(eff, eff_floor, None) if eff_floor > 0 else eff
    safe_eff = np.where(eff > 0, eff_use, 1.0)
    for _ in range(n_iter):
        # expected matched reco from current estimate
        expected = P @ n                                   # (Nreco,)
        expected = np.where(expected > 0, expected, np.inf)
        # Bayes unfolding matrix M[gen, reco] = P[reco,gen] n[gen] / expected[reco]
        # propagate measured through it, then efficiency-correct
        ratio = measured_matched / expected                # (Nreco,)
        update = (P.T @ ratio) * n                          # (Ngen,)
        n = update / safe_eff
        n = np.where(eff > 0, n, 0.0)
        history.append(n.copy())
    return history if return_all else n


# --------------------------------------------------------------------------
# 2. Tikhonov-regularized least squares (2nd-derivative / curvature penalty)
# --------------------------------------------------------------------------
def _curvature_matrix(prob):
    """Second-difference operator acting within each pt slice along rho."""
    Ngen = prob.P.shape[1]
    rows = []
    for ptg in np.unique(prob.gen_pt_idx):
        idx = np.where(prob.gen_pt_idx == ptg)[0]
        for k in range(1, len(idx) - 1):
            row = np.zeros(Ngen)
            row[idx[k - 1]] = 1.0
            row[idx[k]] = -2.0
            row[idx[k + 1]] = 1.0
            rows.append(row)
    return np.array(rows) if rows else np.zeros((0, Ngen))


def tikhonov(prob, measured_matched, tau=1e-3, prior_scale=True):
    """min || P x - m ||^2 + tau^2 || L (x - x0) ||^2 ; x0 = MC truth prior.

    Solved as a linear least-squares (no positivity). Efficiency is in P.
    """
    P = prob.P
    L = _curvature_matrix(prob)
    x0 = prob.gen.copy()
    # scale rows so tau is roughly dimensionless relative to spectrum size
    A = np.vstack([P, tau * L])
    b = np.concatenate([measured_matched, tau * (L @ x0)])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


# --------------------------------------------------------------------------
# 3. SVD unfolding (Hocker-Kartvelishvili style, curvature-rotated)
# --------------------------------------------------------------------------
def svd_unfold(prob, measured_matched, k=None, tau=None):
    """Truncated/Tikhonov SVD on the curvature-preconditioned system.

    Either truncate at k singular values (k given) or apply Tikhonov damping
    1/(s^2 + tau^2) (tau given). Operates on x' = L(x - x0).
    """
    P = prob.P
    L = _curvature_matrix(prob)
    x0 = prob.gen.copy()
    # use curvature as preconditioner: regularize towards smooth deviations
    # Solve P x = m with x = x0 + Lpinv y, damping y.
    Lpinv = np.linalg.pinv(L) if L.shape[0] else np.zeros((P.shape[1], 0))
    A = P @ Lpinv
    r = measured_matched - P @ x0
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    d = U.T @ r
    if k is not None:
        z = np.zeros_like(s)
        z[:k] = d[:k] / s[:k]
    else:
        z = s * d / (s**2 + (tau if tau is not None else 0.0) ** 2)
    y = Vt.T @ z
    return x0 + Lpinv @ y


# --------------------------------------------------------------------------
# 4. Non-negative regularized least squares (positivity + curvature)
# --------------------------------------------------------------------------
def nnls_unfold(prob, measured_matched, tau=1e-2):
    """min ||P x - m||^2 + tau^2 ||L(x - x0)||^2  s.t. x >= 0."""
    P = prob.P
    L = _curvature_matrix(prob)
    x0 = prob.gen.copy()
    A = np.vstack([P, tau * L])
    b = np.concatenate([measured_matched, tau * (L @ x0)])
    x, _ = nnls(A, b, maxiter=10 * A.shape[1])
    return x


# --------------------------------------------------------------------------
# 5. Bin-by-bin (reference only; ignores migration shape changes)
# --------------------------------------------------------------------------
def bin_by_bin(prob, measured_matched):
    """Truth = measured * (MC truth / MC matched reco), gen-binned.

    Only valid where reco and gen binnings match; here we map via the column
    correction factor C[gen] = gen_total / (folded MC). Reference baseline.
    """
    mc_reco = fold(prob, prob.gen)
    # collapse measured to gen binning is not 1:1; use a simple ratio per gen
    # bin from the diagonal-dominant mapping (reco rho finer than gen rho).
    # Here we approximate by folding correction on the truth directly:
    c = np.divide(prob.gen, mc_reco_to_gen(prob), out=np.ones_like(prob.gen),
                  where=mc_reco_to_gen(prob) > 0)
    return measured_to_gen(prob, measured_matched) * c


def mc_reco_to_gen(prob):
    """Project MC matched reco onto gen binning by argmax responsibility."""
    return _project_reco_to_gen(prob, fold(prob, prob.gen))


def measured_to_gen(prob, measured_matched):
    return _project_reco_to_gen(prob, measured_matched)


def _project_reco_to_gen(prob, reco_vec):
    """Distribute a reco vector onto gen bins using P responsibilities."""
    P = prob.P
    resp = np.divide(P, P.sum(axis=1, keepdims=True),
                     out=np.zeros_like(P), where=P.sum(axis=1, keepdims=True) > 0)
    return resp.T @ reco_vec
