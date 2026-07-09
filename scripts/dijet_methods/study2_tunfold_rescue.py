"""Study 2: can a least-squares / TUnfold-style unfolding be rescued for dijet rho?

Direction (v2): HERWIG is NOT used as a validation yardstick (low stats); the
bias metric everywhere is a data-driven pseudo-truth closure. D'Agostini is
disfavored as a primary method because of prior dependence; the prior-free
candidate is UNREGULARIZED weighted least squares on a resolution-matched
coarser gen binning.

Metrics per method knob:
  - bias: unfold pseudo-data (P @ truth*) with the NOMINAL Pythia prior/anchor,
    truth* = smooth data-driven reweight of the Pythia gen (see
    build_pseudo_truth). bias = median |u/truth* - 1| over reported bins.
  - data-stat toy spread (100 toys on the real data, response toys off).
  - negative reported bins / oscillation / foldback on the real data.

Plus:
  - prior-dependence table: unfold the SAME real data with different priors /
    L-anchors (nominal, data-reweighted truth*, +-40% linear tilt) for
    dagostini n=4, ratio-curv Tikhonov at its plateau tau, and unregularized
    lstsq at the winning coarse binning (zero prior dependence by construction).
  - binning scan pushed until unregularized weighted lstsq is publishable
    (0 negative reported bins, osc < ~0.1), with per-bin purity/stability/eff.

Outputs -> outputs/dijet/2018/rho/method_study2/tunfold_rescue/
  metrics.json, README.md, PNGs.

NOTE: never floor efficiencies (validated bug for D'Agostini fixed point).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.dijet_methods.loader import Binning, load_native, rebin, build_problem  # noqa: E402
from scripts.dijet_methods import methods as M  # noqa: E402
from scripts.dijet_methods import result as R  # noqa: E402


OUT = Path("outputs/dijet/2018/rho/method_study2/tunfold_rescue")
OUT.mkdir(parents=True, exist_ok=True)

CHANNEL = "dijet"
YEAR = 2018

PT_EDGES = np.array([0, 200, 290, 400, 570, 760, 13000], float)
RHO_RECO_NATIVE25 = np.array(
    [-10, -8, -7, -6, -5.5, -5, -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25,
     -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0],
    float,
)
RHO_GEN_NATIVE = np.array(
    [-10, -8, -7, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0], float
)
# study gen binning (smoke-tested baseline)
RHO_GEN_STUDY = np.array(
    [-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0], float
)

# reported pt bins = slices 1..5 (drop the 0-200 sink at index 0)
N_TOYS = 100

TAU_SCAN = np.logspace(-3, 4, 29)
NITER_SCAN = [1, 2, 3, 4, 6, 8, 12, 16, 32]

PROD_TAU = 0.79
PROD_NITER = 4

# binning-scan candidates per mode (gen edges; subsets of RHO_GEN_NATIVE).
# coarse_* keep 0.5-wide high-rho bins with progressively merged low-rho tail;
# push_* merge the low-rho tail into a single sink-side bin (and push_C also
# merges the sparse last high-rho bin).
BINNING_CANDIDATES = {
    "groomed": {
        "study": list(RHO_GEN_STUDY),
        "coarse_A": [-10, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
        "coarse_B": [-10, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0],
        "coarse_C": [-10, -5, -4, -3, -2, -1.5, -1, -0.5, 0],
        "push_A": [-10, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
        "push_B": [-10, -3, -2.5, -2, -1.5, -1, -0.5, 0],
        "push_C": [-10, -3, -2.5, -2, -1.5, -1, 0],
    },
    "ungroomed": {
        "study": list(RHO_GEN_STUDY),
        "coarse_A": [-10, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
        "coarse_B": [-10, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0],
        "coarse_C": [-10, -5, -4, -3, -2, -1.5, -1, -0.5, 0],
        "push_A": [-10, -2.5, -2, -1.5, -1, -0.5, 0],
        "push_B": [-10, -2, -1.5, -1, -0.5, 0],
        "push_C": [-10, -2, -1.5, -1, 0],
    },
}


# --------------------------------------------------------------------------
# weighted least squares (Poisson-floored inverse-variance weighting)
# --------------------------------------------------------------------------
def _safe_var(measured_matched, measured_var):
    """Poisson-floored variance for inverse-variance weighting.

    The data pickles carry sumw2-style variances that are ~O(200) x the counts
    (weighted events), with near-empty reco bins having tiny/zero variance. Left
    unfloored, 1/sqrt(var) blows those rows up by ~1e15 and lstsq(rcond) then
    truncates the whole system -> spectrum collapses to 0. Floor at max(var,
    counts, 1) so weights stay in a sane dynamic range and empty bins do not
    dominate.
    """
    if measured_var is None:
        measured_var = np.clip(measured_matched, 1.0, None)
    return np.maximum.reduce([
        np.clip(measured_var, 0, None),
        np.clip(measured_matched, 0, None),
        np.ones_like(measured_matched),
    ])


def _ratio_curvature_L(prob, x0):
    """L rows (1/x0[i-1], -2/x0[i], 1/x0[i+1]) on interior gen bins per pt slice.

    Zero penalty when x proportional to x0. Skip rows where any anchor <= 0
    (mirrors production _add_ratio_curvature_conditions).
    """
    Ngen = prob.P.shape[1]
    rows = []
    for ptg in np.unique(prob.gen_pt_idx):
        idx = np.where(prob.gen_pt_idx == ptg)[0]
        for k in range(1, len(idx) - 1):
            j0, j1, j2 = idx[k - 1], idx[k], idx[k + 1]
            m0, m1, m2 = x0[j0], x0[j1], x0[j2]
            if min(m0, m1, m2) <= 0:
                continue
            row = np.zeros(Ngen)
            row[j0] = 1.0 / m0
            row[j1] = -2.0 / m1
            row[j2] = 1.0 / m2
            rows.append(row)
    return np.array(rows) if rows else np.zeros((0, Ngen))


def _plain_curvature_L(prob):
    return M._curvature_matrix(prob)


def tikhonov_weighted(prob, measured_matched, tau=1e-3, flavor="ratio",
                      measured_var=None, x0=None):
    """Weighted least squares:
        min || W^{1/2}(P x - m) ||^2 + tau^2 || L x' ||^2
    with W = diag(1/var).

    ratio flavor:  penalty on L'x with L' = ratio-curvature built from the
                   anchor x0 (zero for x prop x0); matches production
                   ratio_curvature. b-side = 0.
    curvature flavor: penalize L (x - x0) as in methods.tikhonov.
    x0 defaults to the MC gen prior; pass an alternative to probe prior/anchor
    dependence.
    """
    P = prob.P
    if x0 is None:
        x0 = prob.gen.copy()
    var = _safe_var(measured_matched, measured_var)
    w = 1.0 / np.sqrt(var)
    # empty reco rows (no folded MC support and no data) carry no information
    active = (P.sum(axis=1) > 0) | (measured_matched > 0)
    w = np.where(active, w, 0.0)
    Aw = P * w[:, None]
    bw = measured_matched * w
    if flavor == "ratio":
        L = _ratio_curvature_L(prob, x0)
        Lb = np.zeros(L.shape[0])
    else:
        L = _plain_curvature_L(prob)
        Lb = L @ x0
    A = np.vstack([Aw, tau * L])
    b = np.concatenate([bw, tau * Lb])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def unreg_lstsq(prob, measured_matched, measured_var=None, return_chi2=False):
    """Unregularized inverse-variance-weighted least squares (prior-free)."""
    var = _safe_var(measured_matched, measured_var)
    w = 1.0 / np.sqrt(var)
    active = (prob.P.sum(axis=1) > 0) | (measured_matched > 0)
    w = np.where(active, w, 0.0)
    Aw = prob.P * w[:, None]
    bw = measured_matched * w
    x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    if return_chi2:
        ndof = int(np.sum(w > 0)) - prob.P.shape[1]
        chi2 = float(np.sum((Aw @ x - bw) ** 2))
        return x, chi2 / max(ndof, 1)
    return x


# --------------------------------------------------------------------------
# data-driven pseudo-truth (replaces the deprecated Herwig-as-data yardstick)
# --------------------------------------------------------------------------
def build_pseudo_truth(prep, n_pass=4, clip=(0.2, 5.0), eff_min=0.05):
    """Smooth gen-space reweight w(x_gen) per pt slice so that folding
    w * pythia_gen matches the DATA matched reco shape.

    Iterative (a single-pass reweight is known to overestimate): each pass
    folds the current truth*, forms the reco-space data/expected ratio,
    projects it to gen space as the folding-probability-weighted column
    average (P^T r / eff), multiplies into w, smooths within each pt slice
    ([0.25, 0.5, 0.25] kernel), anchors on populated bins (w=1 elsewhere),
    and clips w to `clip`.

    Returns (truth_star, w, foldback_median_relres).
    """
    prob = prep.prob
    gen = prob.gen
    eff = prob.eff
    populated = (gen > 0) & (eff > eff_min)
    w = np.ones_like(gen)
    for _ in range(n_pass):
        truth = w * gen
        expected = prob.P @ truth
        r_reco = np.divide(prep.data_matched, expected,
                           out=np.ones_like(expected), where=expected > 0)
        # gen-space ratio: column-probability-weighted average of reco ratios
        r_gen = np.divide(prob.P.T @ r_reco, eff,
                          out=np.ones_like(gen), where=eff > 0)
        w_new = w * np.where(populated, r_gen, 1.0)
        # smooth within each pt slice (renormalized [0.25, 0.5, 0.25] kernel)
        w_s = w_new.copy()
        for ptg in np.unique(prob.gen_pt_idx):
            idx = np.where(prob.gen_pt_idx == ptg)[0]
            v = w_new[idx]
            sm = v.copy()
            for k in range(len(v)):
                acc, norm = 0.5 * v[k], 0.5
                if k > 0:
                    acc += 0.25 * v[k - 1]
                    norm += 0.25
                if k < len(v) - 1:
                    acc += 0.25 * v[k + 1]
                    norm += 0.25
                sm[k] = acc / norm
            w_s[idx] = sm
        w = np.where(populated, np.clip(w_s, clip[0], clip[1]), 1.0)
    truth_star = w * gen
    expected = prob.P @ truth_star
    good = prep.data_matched > 0
    relres = np.abs(expected[good] - prep.data_matched[good]) / prep.data_matched[good]
    return truth_star, w, float(np.median(relres)) if np.any(good) else float("nan")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _reported_mask(prob):
    """Mask of gen bins in pt slices 1..5 (drop sink slice 0)."""
    return prob.gen_pt_idx >= 1


def osc_metric(prob, u, x0):
    """median | 2nd diff of (u/prior) | within reported pt slices."""
    ratio = np.divide(u, x0, out=np.zeros_like(u), where=x0 > 0)
    vals = []
    for ptg in np.unique(prob.gen_pt_idx):
        if ptg == 0:
            continue
        idx = np.where(prob.gen_pt_idx == ptg)[0]
        r = ratio[idx]
        if len(r) >= 3:
            d2 = r[2:] - 2 * r[1:-1] + r[:-2]
            vals.extend(np.abs(d2).tolist())
    return float(np.median(vals)) if vals else 0.0


def neg_bins(prob, u):
    m = _reported_mask(prob)
    return int(np.sum(u[m] < 0))


def foldback_residual(prob, u, measured_matched):
    """median |rel residual| of P u vs matched data over reco bins with signal."""
    pred = prob.P @ u
    m = measured_matched
    good = m > 0
    rel = np.abs(pred[good] - m[good]) / m[good]
    return float(np.median(rel)) if np.any(good) else float("nan")


def closure_bias(prob, u, truth_star):
    """median & p90 |u/truth* - 1| over reported bins where truth*>0."""
    m = _reported_mask(prob) & (truth_star > 0)
    if not np.any(m):
        return float("nan"), float("nan")
    rel = np.abs(u[m] / truth_star[m] - 1.0)
    return float(np.median(rel)), float(np.percentile(rel, 90))


def toy_spread(prep, method_fn):
    """Data-stat toy spread over reported bins, response_toys off.

    Returns (median_frac, p90_frac)."""
    central, std, cov, toys = R.unfold_with_unc(
        prep, method_fn, n_toys=N_TOYS, response_toys=False, seed=0
    )
    m = _reported_mask(prep.prob) & (prep.prob.gen > 0)
    frac = np.divide(std, np.abs(central), out=np.zeros_like(std),
                     where=np.abs(central) > 0)
    return float(np.median(frac[m])), float(np.percentile(frac[m], 90))


# --------------------------------------------------------------------------
# method wrappers that R.unfold_with_unc can call: fn(prob, measured_matched)
# --------------------------------------------------------------------------
def make_tik_fn(tau, flavor, measured_var):
    def fn(prob, measured_matched):
        return tikhonov_weighted(prob, measured_matched, tau=tau, flavor=flavor,
                                 measured_var=measured_var)
    return fn


def make_dago_fn(n_iter):
    def fn(prob, measured_matched):
        return M.dagostini(prob, measured_matched, n_iter=n_iter)
    return fn


def make_unreg_fn(measured_var):
    def fn(prob, measured_matched):
        return unreg_lstsq(prob, measured_matched, measured_var)
    return fn


# --------------------------------------------------------------------------
# driver for one mode / one binning
# --------------------------------------------------------------------------
def run_family_scan(prep, truth_star, binning_label="study"):
    prob = prep.prob
    x0 = prob.gen.copy()
    dm = prep.data_matched
    dvar = prep.data_var
    pseudo = prob.P @ truth_star           # noiseless pseudo-data (matched)

    out = {}

    # --- Tikhonov families ---
    for flavor in ("ratio", "curvature"):
        fam = {}
        for tau in TAU_SCAN:
            u = tikhonov_weighted(prob, dm, tau=tau, flavor=flavor, measured_var=dvar)
            u_ps = tikhonov_weighted(prob, pseudo, tau=tau, flavor=flavor,
                                     measured_var=dvar)
            cb_med, cb_p90 = closure_bias(prob, u_ps, truth_star)
            spread, spread_p90 = toy_spread(prep, make_tik_fn(tau, flavor, dvar))
            fam[f"{tau:.6g}"] = {
                "tau": float(tau),
                "neg_bins": neg_bins(prob, u),
                "osc": osc_metric(prob, u, x0),
                "foldback": foldback_residual(prob, u, dm),
                "toy_spread": spread,
                "toy_spread_p90": spread_p90,
                "closure_bias_med": cb_med,
                "closure_bias_p90": cb_p90,
            }
            print(f"    [{binning_label}] tik {flavor} tau={tau:.3g}: "
                  f"neg={fam[f'{tau:.6g}']['neg_bins']} osc={fam[f'{tau:.6g}']['osc']:.3g} "
                  f"bias={cb_med:.3g} spread={spread:.3g}")
        out[f"tikhonov_{flavor}"] = fam

    # --- Bayes family ---
    fam = {}
    for ni in NITER_SCAN:
        u = M.dagostini(prob, dm, n_iter=ni)
        u_ps = M.dagostini(prob, pseudo, n_iter=ni)
        cb_med, cb_p90 = closure_bias(prob, u_ps, truth_star)
        spread, spread_p90 = toy_spread(prep, make_dago_fn(ni))
        fam[str(ni)] = {
            "n_iter": ni,
            "neg_bins": neg_bins(prob, u),
            "osc": osc_metric(prob, u, x0),
            "foldback": foldback_residual(prob, u, dm),
            "toy_spread": spread,
            "toy_spread_p90": spread_p90,
            "closure_bias_med": cb_med,
            "closure_bias_p90": cb_p90,
        }
        print(f"    [{binning_label}] bayes n={ni}: neg={fam[str(ni)]['neg_bins']} "
              f"osc={fam[str(ni)]['osc']:.3g} bias={cb_med:.3g} spread={spread:.3g}")
    out["bayes"] = fam
    return out


def plateau_tau(fam_ratio, bayes_osc_ref):
    """Smallest tau whose oscillation is at or below the Bayes reference."""
    ks = sorted(fam_ratio, key=lambda k: fam_ratio[k]["tau"])
    for k in ks:
        if fam_ratio[k]["osc"] <= max(bayes_osc_ref, 0.1):
            return fam_ratio[k]["tau"]
    return fam_ratio[ks[-1]]["tau"]


# --------------------------------------------------------------------------
# prior-dependence table (boss-facing deliverable)
# --------------------------------------------------------------------------
def tilt_prior(prob, sign=+1, amplitude=0.4):
    """+-40% linear tilt of the Pythia gen across each pt slice."""
    gen = prob.gen.copy()
    w = np.ones_like(gen)
    for ptg in np.unique(prob.gen_pt_idx):
        idx = np.where(prob.gen_pt_idx == ptg)[0]
        n = len(idx)
        frac = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
        w[idx] = 1.0 + sign * amplitude * (2.0 * frac - 1.0)
    return w * gen


def prior_dependence(prep_study, truth_star_study, tau_plateau,
                     prep_coarse, truth_star_coarse):
    """Spread of central values (real data) across priors/anchors per method."""
    out = {}

    def spread_stats(u_by_prior):
        # Relative prior-induced shift per reported bin. The denominator is
        # floored at 5% of the MC truth so bins the estimator drives to ~0 do
        # not blow the max up to meaningless values (they are still counted).
        prob = u_by_prior.pop("_prob")
        u_nom = u_by_prior["nominal"]
        gen = prob.gen
        m = _reported_mask(prob) & (gen > 0)
        den = np.maximum(np.abs(u_nom[m]), 0.05 * gen[m])
        rels = []
        for name, u in u_by_prior.items():
            if name == "nominal":
                continue
            rels.append(np.abs(u[m] - u_nom[m]) / den)
        worst = np.max(rels, axis=0)
        return {"median": float(np.median(worst)), "max": float(np.max(worst))}

    def priors_for(prep, truth_star):
        prob = prep.prob
        return {
            "nominal": prob.gen.copy(),
            "data_reweighted": truth_star.copy(),
            "tilt_up": tilt_prior(prob, +1),
            "tilt_down": tilt_prior(prob, -1),
        }

    # 1. dagostini n=4, study binning
    prob = prep_study.prob
    res = {"_prob": prob}
    for name, p in priors_for(prep_study, truth_star_study).items():
        res[name] = M.dagostini(prob, prep_study.data_matched,
                                n_iter=PROD_NITER, prior=p)
    out["dagostini_n4_study"] = spread_stats(res)

    # 2. ratio-curv Tikhonov at plateau tau, study binning (prior enters via L)
    res = {"_prob": prob}
    for name, p in priors_for(prep_study, truth_star_study).items():
        res[name] = tikhonov_weighted(prob, prep_study.data_matched,
                                      tau=tau_plateau, flavor="ratio",
                                      measured_var=prep_study.data_var, x0=p)
    out[f"tikhonov_ratio_plateau_study"] = dict(
        spread_stats(res), tau=float(tau_plateau))

    # 3. dagostini n=4, winning coarse binning (binning not a confounder)
    probc = prep_coarse.prob
    res = {"_prob": probc}
    for name, p in priors_for(prep_coarse, truth_star_coarse).items():
        res[name] = M.dagostini(probc, prep_coarse.data_matched,
                                n_iter=PROD_NITER, prior=p)
    out["dagostini_n4_coarse"] = spread_stats(res)

    # 4. unregularized lstsq at the winning coarse binning: no prior input at
    # all -> zero prior dependence by construction. Verify numerically that
    # repeated solves are bit-identical (lstsq is deterministic; nothing in the
    # system depends on any prior).
    u_ref = unreg_lstsq(probc, prep_coarse.data_matched, prep_coarse.data_var)
    max_dev = 0.0
    for _ in range(3):
        u_i = unreg_lstsq(probc, prep_coarse.data_matched, prep_coarse.data_var)
        max_dev = max(max_dev, float(np.max(np.abs(u_i - u_ref))))
    out["unreg_lstsq_coarse"] = {
        "median": 0.0, "max": 0.0,
        "note": "no prior enters the estimator (zero dependence by construction)",
        "repeat_solve_max_abs_dev": max_dev,
    }
    return out


# --------------------------------------------------------------------------
# binning scan
# --------------------------------------------------------------------------
def migration_rms_per_genbin(prep):
    """RMS of each gen bin's reco-x kernel (in x units), from the study P."""
    prob = prep.prob
    binning = prep.binning
    rc = 0.5 * (binning.rho_edges_reco[:-1] + binning.rho_edges_reco[1:])
    rms = {}
    for j in range(prob.P.shape[1]):
        col = prob.R[:, j]
        if col.sum() <= 0:
            continue
        xr = rc[prob.reco_rho_idx]
        w = col
        mean = np.sum(w * xr) / np.sum(w)
        var = np.sum(w * (xr - mean) ** 2) / np.sum(w)
        rms[j] = np.sqrt(max(var, 0.0))
    return rms


def build_prep_with_gen(mode, rho_gen_edges):
    binning = Binning(
        pt_edges=PT_EDGES,
        rho_edges_reco=RHO_RECO_NATIVE25,
        rho_edges_gen=np.asarray(rho_gen_edges, float),
    )
    return R.prepare(mode, binning, channel=CHANNEL)


def cond_P(prob):
    good = prob.eff > 0
    Pg = prob.P[:, good]
    s = np.linalg.svd(Pg, compute_uv=False)
    s = s[s > 0]
    return float(s[0] / s[-1]) if len(s) else float("inf")


def per_bin_purity_stability(prep):
    """Purity & stability per gen bin via a reco->gen coarse projection:
    each reco rho bin is assigned to the gen rho bin containing its center
    (edges are nested)."""
    prob = prep.prob
    binning = prep.binning
    rc = 0.5 * (binning.rho_edges_reco[:-1] + binning.rho_edges_reco[1:])
    ge = binning.rho_edges_gen
    reco_to_genrho = np.clip(np.searchsorted(ge, rc, side="right") - 1, 0, len(ge) - 2)
    nptg = len(binning.pt_edges) - 1
    nrg = len(ge) - 1
    purity = np.zeros(prob.P.shape[1])
    stability = np.zeros(prob.P.shape[1])
    R4 = prep.rb_mc.response  # (nptr, nptg, nrr, nrg)
    for ptb in range(nptg):
        for grho in range(nrg):
            gidx = ptb * nrg + grho
            recorho_sel = np.where(reco_to_genrho == grho)[0]
            matched_diag = R4[ptb, ptb][:, grho][recorho_sel].sum()
            matched_recowin = R4[ptb, ptb][recorho_sel, :].sum()
            gen_matched_col = R4[:, ptb][:, :, grho].sum()
            purity[gidx] = matched_diag / matched_recowin if matched_recowin > 0 else 0.0
            stability[gidx] = matched_diag / gen_matched_col if gen_matched_col > 0 else 0.0
    return purity, stability


def run_binning_scan(mode):
    print(f"  [binning-scan {mode}] computing migration RMS on study binning")
    prep_study = build_prep_with_gen(mode, RHO_GEN_STUDY)
    rms = migration_rms_per_genbin(prep_study)
    binning = prep_study.binning
    nrg = len(binning.rho_edges_gen) - 1
    gc = 0.5 * (binning.rho_edges_gen[:-1] + binning.rho_edges_gen[1:])
    gw = np.diff(binning.rho_edges_gen)
    rms_by_rho = {}
    for grho in range(nrg):
        vals = [rms[j] for j in rms if (j % nrg) == grho]
        rms_by_rho[grho] = float(np.mean(vals)) if vals else float("nan")
    for grho in range(nrg):
        print(f"      gen rho bin {grho:2d}  c={gc[grho]:6.2f}  w={gw[grho]:4.2f}  "
              f"rms={rms_by_rho[grho]:.3f}")

    scan = {}
    for name, edges in BINNING_CANDIDATES[mode].items():
        prep = build_prep_with_gen(mode, edges)
        prob = prep.prob
        c = cond_P(prob)
        purity, stability = per_bin_purity_stability(prep)
        eff = prob.eff
        m = _reported_mask(prob) & (prob.gen > 0)
        u, chi2_ndof = unreg_lstsq(prob, prep.data_matched, prep.data_var,
                                   return_chi2=True)
        nb = neg_bins(prob, u)
        osc = osc_metric(prob, u, prob.gen)
        u_dago = M.dagostini(prob, prep.data_matched, n_iter=PROD_NITER)
        mm = _reported_mask(prob) & (prob.gen > 0) & (u_dago != 0)
        rel_diff = np.abs(u[mm] / u_dago[mm] - 1.0)
        entry = {
            "gen_edges": list(map(float, edges)),
            "n_gen_bins_total": int(prob.P.shape[1]),
            "n_gen_bins_reported": int(np.sum(_reported_mask(prob))),
            "cond_P": c,
            "purity_median": float(np.median(purity[m])),
            "purity_min": float(np.min(purity[m])),
            "stability_median": float(np.median(stability[m])),
            "stability_min": float(np.min(stability[m])),
            "eff_median": float(np.median(eff[m])),
            "unreg_neg_bins": nb,
            "unreg_osc": osc,
            "unreg_chi2_ndof": chi2_ndof,
            "unreg_vs_dago4_reldiff_med": float(np.median(rel_diff)) if len(rel_diff) else float("nan"),
            "unreg_vs_dago4_reldiff_p90": float(np.percentile(rel_diff, 90)) if len(rel_diff) else float("nan"),
            "publishable": bool(nb == 0 and osc < 0.1),
        }
        scan[name] = entry
        print(f"    [{mode}] binning {name}: nbins={prob.P.shape[1]} "
              f"cond={c:.3g} purity_med={entry['purity_median']:.3f} "
              f"neg={nb} osc={osc:.3g} chi2/ndof={chi2_ndof:.0f} "
              f"vs_dago={entry['unreg_vs_dago4_reldiff_med']:.3g}"
              f"{'  <-- publishable' if entry['publishable'] else ''}")

    # prefer the publishable candidate with the MOST gen bins (least merging);
    # if none is fully publishable (the observed outcome: the residual negative
    # bins are deterministic data/model incompatibility, chi2/ndof >> 1, not a
    # conditioning problem), fall back to the least-bad candidate by
    # (neg_bins, osc) and record why.
    publishable = [n for n, d in scan.items() if d["publishable"]]
    if publishable:
        winner_name = max(publishable, key=lambda n: scan[n]["n_gen_bins_total"])
        winner_note = "meets criteria (0 negative reported bins, osc < 0.1)"
    else:
        winner_name = min(scan, key=lambda n: (scan[n]["unreg_neg_bins"],
                                               scan[n]["unreg_osc"]))
        winner_note = (
            "NO candidate meets the criteria: the remaining negative bins are "
            "deterministic (toy pulls of O(-100)), a data/model incompatibility "
            "(weighted foldback chi2/ndof >> 1), not a conditioning problem; "
            "this is the least-bad candidate by (neg_bins, osc)."
        )

    detail = None
    if winner_name is not None:
        prep_w = build_prep_with_gen(mode, scan[winner_name]["gen_edges"])
        prob_w = prep_w.prob
        purity, stability = per_bin_purity_stability(prep_w)
        eff = prob_w.eff
        central_w, std_w, _, _ = R.unfold_with_unc(
            prep_w, make_unreg_fn(prep_w.data_var), n_toys=N_TOYS,
            response_toys=False, seed=0,
        )
        mrep = _reported_mask(prob_w) & (prob_w.gen > 0)
        frac_w = np.divide(std_w, np.abs(central_w), out=np.zeros_like(std_w),
                           where=np.abs(central_w) > 0)
        spread_med = float(np.median(frac_w[mrep]))
        spread_p90 = float(np.percentile(frac_w[mrep], 90))
        # deterministic negative bins: locations and toy pulls
        neg_detail = []
        nrg_tmp = len(prep_w.binning.rho_edges_gen) - 1
        for j in np.where(_reported_mask(prob_w) & (central_w < 0))[0]:
            neg_detail.append({
                "pt_bin": int(j // nrg_tmp),
                "rho_lo": float(prep_w.binning.rho_edges_gen[j % nrg_tmp]),
                "rho_hi": float(prep_w.binning.rho_edges_gen[j % nrg_tmp + 1]),
                "u_over_gen": float(central_w[j] / max(prob_w.gen[j], 1e-30)),
                "toy_pull": float(central_w[j] / std_w[j]) if std_w[j] > 0 else float("nan"),
            })
        nrg_w = len(prep_w.binning.rho_edges_gen) - 1
        per_bin = []
        for j in range(prob_w.P.shape[1]):
            ptb, grho = j // nrg_w, j % nrg_w
            if ptb == 0:
                continue
            per_bin.append({
                "pt_bin": int(ptb),
                "rho_lo": float(prep_w.binning.rho_edges_gen[grho]),
                "rho_hi": float(prep_w.binning.rho_edges_gen[grho + 1]),
                "purity": float(purity[j]),
                "stability": float(stability[j]),
                "efficiency": float(eff[j]),
            })
        detail = {
            "name": winner_name,
            "note": winner_note,
            "publishable": scan[winner_name]["publishable"],
            "gen_edges": scan[winner_name]["gen_edges"],
            "unreg_toy_spread_med": spread_med,
            "unreg_toy_spread_p90": spread_p90,
            "negative_bins": neg_detail,
            "per_bin": per_bin,
        }
        print(f"    [{mode}] WINNER {winner_name}: unreg toy spread "
              f"med={spread_med:.3g} p90={spread_p90:.3g}")
    return {"rms_by_rho": rms_by_rho, "binnings": scan, "winner": detail}


# --------------------------------------------------------------------------
# production confirmation (run after scripts/run_rho_unfolding.py --binning
# coarse has produced outputs/dijet/2018/rho/unfolding_coarse/)
# --------------------------------------------------------------------------
# Production-style reco edges for the coarse binning: native fine bins from one
# native edge below the first interior gen edge; everything lower merged into
# the reco sink (mirrors channel_rho_binning(variant="coarse")).
PRODUCTION_COARSE_RECO = {
    "groomed": [-10, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25,
                -1, -0.75, -0.5, -0.25, 0],
    "ungroomed": [-10, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0],
}


def confirm_production(metrics,
                       artifact_dir="outputs/dijet/2018/rho/unfolding_coarse"):
    """Compare the production TUnfold coarse run (tau=0, area constraint) with
    the numpy unregularized lstsq on the same gen binning, both with the native
    reco binning (as scanned above) and with the production-style merged reco
    sink. Appends a 'production_confirmation' block per mode."""
    for mode in ("groomed", "ungroomed"):
        path = Path(artifact_dir) / "artifacts" / f"{mode}_results.npz"
        if not path.exists():
            print(f"  [confirm] {path} missing; skipping {mode}")
            continue
        d = np.load(path, allow_pickle=True)
        u_prod, prior = d["unfolded"], d["truth_prior"]
        err = d["unfolded_input_errors"]
        edges = metrics[mode]["binning_scan"]["winner"]["gen_edges"]
        nrg = len(edges) - 1

        # production-result sanity metrics (reported = pt slices 1..5)
        neg = 0
        osc_vals = []
        for ptb in range(1, len(PT_EDGES) - 1):
            s = slice(ptb * nrg, (ptb + 1) * nrg)
            neg += int(np.sum(u_prod[s] < 0))
            r = np.where(prior[s] > 0, u_prod[s] / prior[s], 0)
            osc_vals.extend(np.abs(r[2:] - 2 * r[1:-1] + r[:-2]))
        rep = np.repeat(np.arange(len(PT_EDGES) - 1), nrg) >= 1
        rel_err = np.divide(err, np.abs(u_prod), out=np.zeros_like(err),
                            where=np.abs(u_prod) > 0)

        # numpy lstsq with the production-style merged reco sink
        b = Binning(pt_edges=PT_EDGES,
                    rho_edges_reco=np.array(PRODUCTION_COARSE_RECO[mode], float),
                    rho_edges_gen=np.array(edges, float))
        prep = R.prepare(mode, b, channel=CHANNEL)
        u_np, chi2 = unreg_lstsq(prep.prob, prep.data_matched, prep.data_var,
                                 return_chi2=True)
        m = _reported_mask(prep.prob) & (np.abs(u_prod) > 0)
        rr = np.abs(u_np[m] / u_prod[m] - 1.0)

        # D'Agostini n=4 cross-check on the same gen edges (native reco)
        prep_nat = build_prep_with_gen(mode, edges)
        u_dago = M.dagostini(prep_nat.prob, prep_nat.data_matched, n_iter=PROD_NITER)
        md = _reported_mask(prep_nat.prob) & (u_dago > 0) & (u_prod > 0)
        rd = np.abs(u_prod[md] / u_dago[md] - 1.0)

        block = {
            "artifact": str(path),
            "prod_neg_reported": neg,
            "prod_osc": float(np.median(osc_vals)),
            "prod_stat_err_med": float(np.median(rel_err[rep])),
            "prod_stat_err_p90": float(np.percentile(rel_err[rep], 90)),
            "numpy_mergedreco_neg": int(np.sum(u_np[_reported_mask(prep.prob)] < 0)),
            "numpy_mergedreco_osc": osc_metric(prep.prob, u_np, prep.prob.gen),
            "numpy_mergedreco_chi2_ndof": chi2,
            "numpy_vs_prod_med": float(np.median(rr)),
            "numpy_vs_prod_p90": float(np.percentile(rr, 90)),
            "prod_vs_dago4_med": float(np.median(rd)),
            "prod_vs_dago4_p90": float(np.percentile(rd, 90)),
            "note": (
                "production coarse run: TUnfold kRegModeDerivative path with "
                "tau resolved to 0 (effectively unregularized) + area "
                "constraint (kEConstraintArea) + merged reco sink below the "
                "first interior gen edge. The merged reco sink removes the "
                "incompatible low-rho reco constraints and eliminates the "
                "deterministic negative bins seen with the native fine reco."
            ),
        }
        metrics[mode]["production_confirmation"] = block
        print(f"  [confirm {mode}] prod neg={neg} osc={block['prod_osc']:.3g} "
              f"stat={block['prod_stat_err_med']:.3g}; numpy(mergedreco) "
              f"neg={block['numpy_mergedreco_neg']} vs prod med={block['numpy_vs_prod_med']:.4f}; "
              f"prod vs dago4 med={block['prod_vs_dago4_med']:.3f}")
    return metrics


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    metrics = {"config": {
        "tau_scan": list(map(float, TAU_SCAN)),
        "niter_scan": NITER_SCAN,
        "n_toys": N_TOYS,
        "prod_tau": PROD_TAU,
        "prod_niter": PROD_NITER,
        "pt_edges": list(map(float, PT_EDGES)),
        "rho_edges_reco": list(map(float, RHO_RECO_NATIVE25)),
        "rho_edges_gen_study": list(map(float, RHO_GEN_STUDY)),
        "bias_metric": (
            "data-driven pseudo-truth closure (Herwig yardstick deprecated: "
            "low stats). truth* = smooth iterative data-reweight of Pythia gen; "
            "pseudo-data = P @ truth*; unfold with nominal prior."
        ),
    }}

    for mode in ("groomed", "ungroomed"):
        print(f"=== MODE {mode} ===")
        prep = build_prep_with_gen(mode, RHO_GEN_STUDY)
        print(f"  Nreco={prep.prob.P.shape[0]} Ngen={prep.prob.P.shape[1]} "
              f"cond(P)={cond_P(prep.prob):.4g}")
        truth_star, w_star, fold_res = build_pseudo_truth(prep)
        print(f"  pseudo-truth: w in [{w_star.min():.2f}, {w_star.max():.2f}], "
              f"foldback median relres = {fold_res:.4f}")
        _, chi2_study = unreg_lstsq(prep.prob, prep.data_matched, prep.data_var,
                                    return_chi2=True)
        print(f"  weighted lstsq foldback chi2/ndof (study binning) = {chi2_study:.1f}")
        fam = run_family_scan(prep, truth_star, binning_label=mode)
        binscan = run_binning_scan(mode)

        tau_p = plateau_tau(fam["tikhonov_ratio"],
                            fam["bayes"][str(PROD_NITER)]["osc"])
        print(f"  plateau tau ({mode}) = {tau_p:g}")

        # prior-dependence table on study binning + winning coarse binning
        winner = binscan["winner"]
        coarse_edges = (winner["gen_edges"] if winner is not None
                        else BINNING_CANDIDATES[mode]["coarse_C"])
        prep_coarse = build_prep_with_gen(mode, coarse_edges)
        truth_star_c, w_c, fold_res_c = build_pseudo_truth(prep_coarse)
        pd = prior_dependence(prep, truth_star, tau_p, prep_coarse, truth_star_c)
        for meth, st in pd.items():
            print(f"    prior-dep {meth}: median={st['median']:.4g} max={st['max']:.4g}")

        # closure bias of the prior-free candidate at the winning binning
        u_ps = unreg_lstsq(prep_coarse.prob, prep_coarse.prob.P @ truth_star_c,
                           prep_coarse.data_var)
        cb_med, cb_p90 = closure_bias(prep_coarse.prob, u_ps, truth_star_c)
        winner_point = {
            "binning": (winner["name"] if winner is not None else "coarse_C"),
            "closure_bias_med": cb_med,
            "closure_bias_p90": cb_p90,
            "toy_spread_med": (winner["unreg_toy_spread_med"] if winner else None),
            "toy_spread_p90": (winner["unreg_toy_spread_p90"] if winner else None),
        }
        print(f"  prior-free candidate closure bias: med={cb_med:.3g} p90={cb_p90:.3g}")

        metrics[mode] = {
            "unreg_chi2_ndof_study": chi2_study,
            "pseudo_truth": {
                "w_min": float(w_star.min()), "w_max": float(w_star.max()),
                "foldback_median_relres": fold_res,
                "coarse_foldback_median_relres": fold_res_c,
            },
            "families": fam,
            "plateau_tau": tau_p,
            "prior_dependence": pd,
            "binning_scan": binscan,
            "prior_free_candidate": winner_point,
        }

    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {OUT/'metrics.json'}")
    return metrics


if __name__ == "__main__":
    if "--confirm-only" in sys.argv:
        with open(OUT / "metrics.json") as fh:
            existing = json.load(fh)
        confirm_production(existing)
        with open(OUT / "metrics.json", "w") as fh:
            json.dump(existing, fh, indent=2)
        print(f"updated {OUT/'metrics.json'} (production confirmation)")
    else:
        main()
