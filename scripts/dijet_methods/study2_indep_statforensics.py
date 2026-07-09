"""Part 3: STAT-ERROR FORENSICS of the D'Agostini (roounfold_bayes) tag.

(a) Reproduce the HONEST data-stat spread with Gaussian toys through D'Agostini
    n=4 (data-only, then +response-matrix), using the STORED sumw2 variance with a
    Poisson floor -- never sqrt(N). Compare the median/p90 against the tag's quoted
    `unfolded_input_errors` from the npz artifact.
(b) Quantify the defect at its SOURCE: the tag feeds RooUnfold a data histogram
    whose bin errors are sqrt(|content|)=sqrt(N) (roounfold_backend.py:63), so it
    discards the stored sumw2. sumw2/N ~ 247 for this weighted data => an
    underestimate factor sqrt(247) ~ 15.7x. (File:line chain documented in README.)
(c) n_iter regularization systematic the honest way: |u(2)-u(4)|, |u(6)-u(4)|.
"""
from __future__ import annotations

import numpy as np

from scripts.dijet_methods import result as R
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.study2_indep_common import REPORTED_PT


def _rel_spread(std, central):
    rel = np.divide(std, central, out=np.zeros_like(std), where=central > 0) * 100
    return rel, central > 0


def _floor_var(prep):
    """Poisson-floored fake-corrected data variance: var = max(sumw2, N, 1)."""
    prob = prep.prob
    f = M.fake_fraction(prob)
    var_matched = prep.data_var * (1.0 - f) ** 2
    return np.maximum.reduce([var_matched,
                              np.clip(prep.data_matched, 0, None),
                              np.ones_like(prep.data_matched)])


def toy_stat_spread(prep, n_toys=200, response_toys=False, n_iter=4, seed=0):
    """Honest toy spread of the D'Agostini n=4 unfold using stored (floored)
    variances. Returns (central, std, rel%, mask). We call the foundation
    unfold_with_unc with measured = matched data and measured_var = floored var
    so the Gaussian toys carry the real prescale-weighted stat, not sqrt(N)."""
    floored = _floor_var(prep)
    # unfold_with_unc subtracts fakes itself, so pass raw measured + a variance
    # that already reflects the fake-corrected floor scaled back up by 1/(1-f)^2
    # is unnecessary: instead pass the fake-corrected matched spectrum directly
    # as "measured" with fake fraction ~0 effect. Simplest correct route: feed
    # the matched spectrum and floored matched variance, and disable the internal
    # fake subtraction by using a fake-free problem view. We instead pass raw
    # measured + raw stored var (the foundation applies (1-f) inside subtract),
    # matching how the tag would honestly propagate.
    central, std, cov, toys = R.unfold_with_unc(
        prep, M.dagostini, measured=prep.data_meas,
        measured_var=np.maximum.reduce([prep.data_var,
                                        np.clip(prep.data_meas, 0, None),
                                        np.ones_like(prep.data_meas)]),
        n_toys=n_toys, response_toys=response_toys, seed=seed, n_iter=n_iter,
    )
    rel, m = _rel_spread(std, central)
    return central, std, rel, m


def reported_mask(prob):
    m = np.zeros(prob.P.shape[1], bool)
    for pt in REPORTED_PT:
        m[prob.gen_pt_idx == pt] = True
    return m


def summarize_rel(rel, mask):
    v = rel[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median_pct": float("nan"), "p90_pct": float("nan"),
                "max_pct": float("nan")}
    return {"median_pct": float(np.median(v)),
            "p90_pct": float(np.percentile(v, 90)),
            "max_pct": float(np.max(v))}


def tag_quoted_rel(mode):
    """Median/p90/max of the tag's quoted unfolded_input_errors as % of the
    tag's unfolded, over all populated tag bins.

    The tag uses a per-pt VARIABLE rho binning (unfolded length is not
    n_pt*n_rho), so we cannot cleanly project onto our pt slices; the pt<200 sink
    is a small fraction of bins and the median over all populated bins is a fair
    summary of the quoted per-bin stat."""
    path = f"outputs/dijet/2018/rho/unfolding_bayes/artifacts/{mode}_results.npz"
    d = np.load(path, allow_pickle=True)
    unf = d["unfolded"]
    err = d["unfolded_input_errors"]
    rel, m = _rel_spread(err, unf)
    return summarize_rel(rel, m), path


def sqrtN_vs_sumw2(mode):
    """Defect at the SOURCE: sqrt(N)/N (what RooUnfold sees) vs sqrt(sumw2)/N (the
    honest per-bin data stat). Returns the ratio sumw2/N and the underestimate
    factor sqrt(sumw2/N)."""
    path = f"outputs/dijet/2018/rho/unfolding_bayes/artifacts/{mode}_results.npz"
    d = np.load(path, allow_pickle=True)
    meas = d["measured"]
    mv_raw = d["measured_variances_raw"]
    m = meas > 0
    rel_sqrtN = np.sqrt(np.abs(meas)) / np.where(meas > 0, meas, 1) * 100
    rel_sumw2 = np.sqrt(np.clip(mv_raw, 0, None)) / np.where(meas > 0, meas, 1) * 100
    ratio = np.divide(mv_raw, np.where(meas > 0, meas, 1))[m]
    return {
        "implied_rel_from_sqrtN_median_pct": float(np.median(rel_sqrtN[m])),
        "honest_rel_from_sumw2_median_pct": float(np.median(rel_sumw2[m])),
        "sumw2_over_N_median": float(np.median(ratio)),
        "underestimate_factor_median": float(np.sqrt(np.median(ratio))),
    }


def n_iter_systematic(prep, n_iters=(2, 4, 6)):
    """Honest regularization systematic: |u(2)-u(4)|, |u(6)-u(4)| as % of u(4)
    over reported pt slices."""
    prob = prep.prob
    u = {n: M.dagostini(prob, prep.data_matched, n_iter=n) for n in n_iters}
    mask = reported_mask(prob)
    u4 = u[4]
    out = {}
    for n in n_iters:
        if n == 4:
            continue
        rel = np.divide(np.abs(u[n] - u4), u4,
                        out=np.zeros_like(u4), where=u4 > 0) * 100
        out[f"n{n}_vs_n4"] = summarize_rel(rel, mask & (u4 > 0))
    return out, u
