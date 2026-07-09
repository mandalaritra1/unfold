"""Method 1: FORWARD-FOLDING SPLINE FIT (primary independent cross-check).

Per reported pt slice we fit the reco data in that reco-pt block by folding a
*parametrized gen truth* shape through the intra-slice response block. No matrix
inversion, no iterative Bayes -- a genuinely independent forward-fold. The gen
shape per pt slice is

    x_pt(theta) = s_pt * pythia_gen_pt * exp( spline_pt(x) )

where x = log10(rho^2), spline_pt is a natural cubic spline with 4 interior knots
across the populated gen range of that slice (its coefficients are free
parameters), and s_pt is a free per-slice normalization. HERWIG IS NOT USED as a
template anywhere here -- the parametrization is Pythia * exp(free smooth
correction), so the fitted shape is data-driven, not tied to a distrusted
generator.

pt migration: only ~70-81% of the folded reco in a given reco-pt block comes from
the intra-slice gen block (measured); the rest migrates in from other gen-pt
slices (dominated by the pt<200 sink leaking into pt1). We handle this by
SUBTRACTING the MC (Pythia) out-of-slice prediction from the data reco block
before fitting, and by weighting each reco bin by its intra-slice purity so bins
dominated by the untrusted low-rho sink migration do not drive the fit. This
decouples the slices without a global degenerate fit. Consequence: the lowest
reported slices (pt1, and ungroomed pt2) sit under heavy sink contamination and
cannot be cleanly cross-checked this way -- reported honestly as a limitation.

chi2 = sum_r (P_block x - m_block)_r^2 / (var_data_floored + (0.02 m)^2)_r,
data variance Poisson-floored (var = max(sumw2, N, 1)). The 2% floor represents a
model-systematic scale so the fit does not chase sub-percent reco structure
(documented in the README). Minimized with scipy.optimize.least_squares (TRF); a
mild ridge keeps the spline a shape term so s_pt carries the normalization.
Per-slice normalization errors are read off the Jacobian (J^T J)^-1 scaled by the
reduced chi2 -- a linearized/Gaussian estimate (caveat noted in the README).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from scripts.dijet_methods.study2_indep_common import (
    REPORTED_PT, slice_gen, slice_reco, gen_rho_centers, MODEL_FLOOR_FRAC,
)
from scripts.dijet_methods import methods as M


# number of interior spline knots per slice (task: 3-5). 4 knots -> a natural
# cubic spline with enough freedom to track a smooth shape correction without
# ringing on the ~10-12 populated gen bins per slice.
N_KNOTS = 4
# the smooth log-correction is squashed through tanh into [-CORR_CAP, CORR_CAP]
# so the optimizer cannot run a coefficient off to infinity on a low-population
# bin; +-1 in log => factor exp(+-1) ~ [0.37, 2.7], comfortably bracketing any
# physical Pythia/data shape difference.
CORR_CAP = 1.5


def _prep_arrays(prep):
    prob = prep.prob
    meas = prep.data_matched
    var = prep.data_var
    f = M.fake_fraction(prob)
    var_matched = var * (1.0 - f) ** 2
    # Poisson-floor the data variance (var = max(var, counts, 1)) before adding
    # the model floor -- prescale-weighted triggers give sumw2 >> N, but empty /
    # sparse bins must not get a zero error.
    var_floor = np.maximum.reduce([var_matched, np.clip(meas, 0, None),
                                   np.ones_like(meas)])
    sigma = np.sqrt(var_floor + (MODEL_FLOOR_FRAC * meas) ** 2)
    inv_sigma = 1.0 / np.where(sigma > 0, sigma, np.inf)
    pyth = prob.gen.astype(float)
    return prob, meas, inv_sigma, pyth


def _natural_cubic_basis(x, knots):
    """Design matrix for a natural cubic spline with the given interior knots.

    Columns: [1, x, then the natural-spline truncated-power basis functions].
    A natural cubic spline is linear beyond the boundary knots, so it will not
    blow up in the sparse tails. Returns (npt, nbasis)."""
    x = np.asarray(x, float)
    kn = np.asarray(knots, float)
    K = len(kn)
    # truncated cubic power basis with the natural boundary reduction
    def d(k):
        num = np.clip(x - kn[k], 0, None) ** 3 - np.clip(x - kn[K - 1], 0, None) ** 3
        return num / (kn[K - 1] - kn[k])
    # NOTE: no constant (intercept) column -- normalization is carried solely by
    # the free per-slice scale s_pt, otherwise s_pt and the spline intercept are
    # degenerate. We also center the linear and cubic columns to zero mean so the
    # spline correction is (to first order) shape-only, leaving s_pt as the norm.
    cols = [x - x.mean()]
    for k in range(K - 2):
        c = d(k) - d(K - 2)
        cols.append(c - c.mean())
    return np.vstack(cols).T


def _knots_for_slice(xc, gpop):
    """Interior knots spread across the *populated* gen range of a slice."""
    xs = xc[gpop]
    lo, hi = xs.min(), xs.max()
    # place knots strictly inside [lo, hi]
    return np.linspace(lo, hi, N_KNOTS + 2)[1:-1]


def run_forward_fold(prep):
    """Per-pt-slice forward-fold spline fit.

    For each reported pt slice we fit the reco data *in that reco-pt block* by
    folding the parametrized slice gen shape through the intra-slice response
    block P[reco_slice, gen_slice], after subtracting the MC (Pythia) prediction
    of migration into that reco block from OTHER gen pt slices. This decouples
    the slices (no sink degeneracy) while still accounting for cross-pt migration
    through a small, MC-based additive correction (<=15% of the block, dominated
    by the pt<200 sink leaking into pt1). Returns (x_full, per-slice-info)."""
    prob, meas, inv_sigma, pyth = _prep_arrays(prep)
    P = prob.P
    xc = gen_rho_centers()

    x_full = pyth.copy()
    info = {}
    chi2_tot = 0.0
    ndf_tot = 0
    for pt in REPORTED_PT:
        gidx = slice_gen(prob, pt)
        ridx = slice_reco(prob, pt)
        gpop = pyth[gidx] > 0
        knots = _knots_for_slice(xc, gpop)
        basis_full = _natural_cubic_basis(xc, knots)   # (nrho, ncoef)
        basis = basis_full                              # aligns row-for-row with gidx
        ncoef = basis.shape[1]

        Pblock = P[np.ix_(ridx, gidx)]                  # intra-slice response
        # MC-predicted migration into these reco rows from all OTHER gen bins:
        other = np.ones(P.shape[1], bool); other[gidx] = False
        m_other = P[np.ix_(ridx, np.where(other)[0])] @ pyth[other]
        m_in = Pblock @ pyth[gidx]                       # MC intra-slice reco pred
        m_block = meas[ridx] - m_other                  # data minus MC out-of-slice
        # Down-weight reco bins whose MC prediction is dominated by out-of-slice
        # migration (purity = intra / (intra+other)): those bins carry little
        # intra-slice information and their (data - MC_other) subtraction is
        # driven by the untrusted out-of-slice modelling. Weight ~ purity keeps
        # the fit anchored to the bins the slice actually populates. Also clip
        # the subtracted spectrum at 0 (negative = MC_other over-predicts data).
        purity = np.divide(m_in, m_in + m_other,
                           out=np.zeros_like(m_in), where=(m_in + m_other) > 0)
        m_block = np.clip(m_block, 0.0, None)
        inv_block = inv_sigma[ridx] * purity

        # starting scale to match the intra-slice reco integral
        den = max(m_in.sum(), 1e-30)
        s0 = max(m_block.sum(), 1e-30) / den

        # Mild ridge on the spline coefficients so the smooth correction stays a
        # shape term and does not do the normalization's job -- keeps s_pt near 1
        # (physical) and stops the spline saturating tanh to fake a rescaling.
        RIDGE = 2.0

        def residuals(theta, Pblock=Pblock, gidx=gidx, basis=basis,
                      m_block=m_block, inv_block=inv_block):
            scale = np.exp(np.clip(theta[0], -50, 50))
            coefs = theta[1:]
            g = CORR_CAP * np.tanh((basis @ coefs) / CORR_CAP)
            xg = scale * pyth[gidx] * np.exp(g)
            xg = np.where(pyth[gidx] > 0, xg, 0.0)
            data_res = (Pblock @ xg - m_block) * inv_block
            return np.concatenate([data_res, RIDGE * coefs])

        theta0 = np.concatenate([[np.log(s0)], np.zeros(ncoef)])
        res = least_squares(residuals, theta0, method="trf",
                            max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        scale = float(np.exp(np.clip(res.x[0], -50, 50)))
        coefs = res.x[1:]
        g = CORR_CAP * np.tanh((basis @ coefs) / CORR_CAP)
        xg = scale * pyth[gidx] * np.exp(g)
        xg = np.where(pyth[gidx] > 0, xg, 0.0)
        x_full[gidx] = xg

        npar = res.x.size
        # data-only chi2 (exclude the ridge rows appended to the residual)
        data_res = res.fun[: len(ridx)]
        chi2 = float(data_res @ data_res)
        ndf = max(len(ridx) - npar, 1)
        chi2_tot += chi2; ndf_tot += ndf
        # scale error from the Jacobian (linearized; README caveat)
        try:
            cov = (chi2 / ndf) * np.linalg.pinv(res.jac.T @ res.jac)
            scale_rel_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        except Exception:
            scale_rel_err = float("nan")
        info[pt] = {"scale": scale, "scale_rel_err": scale_rel_err,
                    "n_knots": N_KNOTS, "ncoef": int(ncoef),
                    "chi2_ndf": float(chi2 / ndf), "success": bool(res.success)}

    info["_global"] = {"chi2": float(chi2_tot), "ndf": int(ndf_tot),
                       "chi2_ndf": float(chi2_tot / max(ndf_tot, 1))}
    return x_full, info
