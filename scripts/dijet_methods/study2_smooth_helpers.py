"""Response-matrix smoothing / parametrization helpers for the dijet rho
stabilization study (study2).

Two ways to rebuild the NATIVE 4D response (nptreco, nptgen, nrhoreco, nrhogen)
so that the reco-rho migration kernel is smooth in reco-rho, WITHOUT touching:
  - the pT-migration structure (ptreco<->ptgen block sums), or
  - the efficiency (per gen column total).

The atomic object we smooth is, for a fixed (ptgen jg, rhogen kg, ptreco ir),
the vector over reco-rho bins:
    col = R[ir, jg, :, kg]        (length nrhoreco)
Its SUM over reco-rho (the amount of that gen-column landing in ptreco-slice ir)
is a physical migration/efficiency quantity and is preserved exactly. Only the
*shape* of the reco-rho distribution within that slice is smoothed.

approach (i)  smooth_response_nonparametric  -- Gaussian kernel smoothing in
              reco-bin-center space (variable-width bins handled).
approach (ii) smooth_response_parametric     -- fit a (possibly two-piece)
              Gaussian in x=log10(rho^2)=reco-bin-center per gen-rho bin, smooth
              mu(x_gen), sigma(x_gen) with low-order polynomials per
              (ptgen->ptreco) pair, and re-integrate over reco bins.

Both return a new response array of the same shape; feed it through
loader.rebin + build_problem exactly like the nominal matrix.
"""

from __future__ import annotations

import numpy as np


def _centers(edges):
    edges = np.asarray(edges, float)
    return 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------
# approach (i): nonparametric Gaussian kernel smoothing
# --------------------------------------------------------------------------
def smooth_response_nonparametric(response, rho_edges_reco, sigma_bins=1.2,
                                  min_counts=1.0):
    """Gaussian-kernel-smooth the reco-rho shape of every (ir, jg, kg) column.

    Parameters
    ----------
    response : (nptreco, nptgen, nrhoreco, nrhogen)
    rho_edges_reco : (nrhoreco+1,) reco-rho bin edges (variable width)
    sigma_bins : Gaussian kernel width, in units of *median reco bin width*
                 (so the smoothing scale is physical in x, not per-bin).
    min_counts : slices whose total < min_counts are left untouched (noise).

    The kernel is built in x=center space, so unequal bin widths are handled by
    evaluating the Gaussian at true centers. Each column is renormalized to its
    original sum -> per-slice pT-migration/efficiency preserved exactly.
    """
    centers = _centers(rho_edges_reco)
    widths = np.diff(rho_edges_reco)
    scale = sigma_bins * np.median(widths)
    # kernel weight matrix W[a, b] = Gauss(center_a - center_b) * width_b
    dx = centers[:, None] - centers[None, :]
    W = np.exp(-0.5 * (dx / scale) ** 2) * widths[None, :]
    # column-normalize the kernel so a flat input -> flat output (density aware)
    W = W / W.sum(axis=1, keepdims=True)

    nptr, nptg, nrr, nrg = response.shape
    out = np.array(response, dtype=float, copy=True)
    for ir in range(nptr):
        for jg in range(nptg):
            block = response[ir, jg]           # (nrr, nrg)
            tot = block.sum(axis=0)            # per rhogen slice total
            # smooth each rhogen column (over reco axis)
            sm = W @ block                     # (nrr, nrg)
            sm_tot = sm.sum(axis=0)
            # renormalize to preserve exact column sums; guard empties
            with np.errstate(divide="ignore", invalid="ignore"):
                fac = np.where(sm_tot > 0, tot / sm_tot, 0.0)
            sm = sm * fac[None, :]
            # only replace columns with enough counts to be worth smoothing
            keep = tot >= min_counts
            out[ir, jg][:, keep] = sm[:, keep]
    return out


# --------------------------------------------------------------------------
# approach (ii): parametric Gaussian kernel per gen bin
# --------------------------------------------------------------------------
def _weighted_moments(counts, centers):
    """mean, std of a histogram (counts over centers). Returns (n, mean, std)."""
    n = counts.sum()
    if n <= 0:
        return 0.0, np.nan, np.nan
    mean = np.sum(counts * centers) / n
    var = np.sum(counts * (centers - mean) ** 2) / n
    return n, mean, max(var, 0.0) ** 0.5


def _polyfit_safe(x, y, w, deg):
    """Weighted polyfit with graceful degree fallback; returns poly1d."""
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if good.sum() == 0:
        return np.poly1d([np.nan])
    xg, yg, wg = x[good], y[good], w[good]
    d = min(deg, good.sum() - 1)
    if d < 0:
        return np.poly1d([np.nan])
    if d == 0:
        return np.poly1d([np.average(yg, weights=wg)])
    coef = np.polyfit(xg, yg, d, w=np.sqrt(wg))
    return np.poly1d(coef)


def _gauss_integral(edges, mu, sigma):
    """Integral of a unit-area Gaussian(mu,sigma) over each bin in edges."""
    from math import sqrt
    from scipy.special import erf
    z = (edges - mu) / (sigma * sqrt(2.0))
    cdf = 0.5 * (1.0 + erf(z))
    p = np.diff(cdf)
    s = p.sum()
    return p / s if s > 0 else p


def _twopiece_integral(edges, mu, sig_lo, sig_hi):
    """Two-piece (split-normal) Gaussian bin integrals: different sigma below /
    above the mode mu. Continuous, normalized numerically over the bin range."""
    from math import sqrt
    from scipy.special import erf
    c = _centers(edges)
    lo = c < mu
    sig = np.where(lo, sig_lo, sig_hi)
    # evaluate a piecewise density at fine sub-resolution via edge CDF is messy;
    # approximate with per-bin density * width then renormalize (bins are fine).
    dens = np.exp(-0.5 * ((c - mu) / sig) ** 2)
    p = dens * np.diff(edges)
    s = p.sum()
    return p / s if s > 0 else p


def fit_response_parametric(response, rho_edges_reco, rho_edges_gen,
                            pt_edges, deg=2, twopiece=True, min_counts=50.0,
                            skew_ptgen_max=None):
    """Fit a (two-piece) Gaussian reco-rho kernel per (ptgen->ptreco, rhogen).

    Returns (rebuilt_response, fitinfo) where fitinfo holds the smoothed
    mu(x_gen)/sigma(x_gen) polynomials for diagnostics.

    For each (ir=ptreco, jg=ptgen):
      - measure (n, mu, sig) of the reco-rho distribution for each rhogen bin
        that has >= min_counts, at x = reco-bin-center
      - fit mu(x_gen), sigma(x_gen) [and sig_lo/sig_hi if twopiece] as deg-poly
        in x_gen (the gen-bin center), weighted by counts
      - rebuild each rhogen column by integrating the Gaussian over reco bins,
        scaled to the ORIGINAL per-slice column sum (efficiency preserved).
    Columns with < min_counts keep their nominal (unsmoothed) content.
    """
    reco_c = _centers(rho_edges_reco)
    gen_c = _centers(rho_edges_gen)
    nptr, nptg, nrr, nrg = response.shape
    out = np.array(response, dtype=float, copy=True)
    fitinfo = {}

    for ir in range(nptr):
        for jg in range(nptg):
            block = response[ir, jg]              # (nrr, nrg)
            tot = block.sum(axis=0)               # per rhogen slice
            ns = np.zeros(nrg); mus = np.full(nrg, np.nan)
            sigs = np.full(nrg, np.nan)
            siglo = np.full(nrg, np.nan); sighi = np.full(nrg, np.nan)
            for kg in range(nrg):
                col = block[:, kg]
                n, mean, std = _weighted_moments(col, reco_c)
                ns[kg] = n; mus[kg] = mean; sigs[kg] = std
                if twopiece and n > 0 and np.isfinite(mean):
                    lo = reco_c < mean
                    nlo = col[lo].sum(); nhi = col[~lo].sum()
                    if nlo > 0:
                        vlo = np.sum(col[lo] * (reco_c[lo] - mean) ** 2) / nlo
                        siglo[kg] = max(vlo, 1e-6) ** 0.5
                    if nhi > 0:
                        vhi = np.sum(col[~lo] * (reco_c[~lo] - mean) ** 2) / nhi
                        sighi[kg] = max(vhi, 1e-6) ** 0.5

            w = np.where(ns >= min_counts, ns, 0.0)
            pmu = _polyfit_safe(gen_c, mus, w, deg)
            psig = _polyfit_safe(gen_c, sigs, w, deg)
            plo = _polyfit_safe(gen_c, siglo, w, deg) if twopiece else None
            phi = _polyfit_safe(gen_c, sighi, w, deg) if twopiece else None
            fitinfo[(ir, jg)] = dict(pmu=pmu, psig=psig, plo=plo, phi=phi,
                                     gen_c=gen_c, ns=ns, mus=mus, sigs=sigs)

            use_tp = twopiece and (
                skew_ptgen_max is None or jg <= skew_ptgen_max)
            for kg in range(nrg):
                if tot[kg] < min_counts:
                    continue                      # leave nominal
                xg = gen_c[kg]
                mu = float(pmu(xg))
                if not np.isfinite(mu):
                    continue
                if use_tp:
                    sl = float(plo(xg)); sh = float(phi(xg))
                    sl = sl if np.isfinite(sl) and sl > 1e-3 else float(psig(xg))
                    sh = sh if np.isfinite(sh) and sh > 1e-3 else float(psig(xg))
                    if not (np.isfinite(sl) and np.isfinite(sh)
                            and sl > 1e-3 and sh > 1e-3):
                        continue
                    shape = _twopiece_integral(rho_edges_reco, mu, sl, sh)
                else:
                    sg = float(psig(xg))
                    if not np.isfinite(sg) or sg <= 1e-3:
                        continue
                    shape = _gauss_integral(rho_edges_reco, mu, sg)
                out[ir, jg][:, kg] = shape * tot[kg]
    return out, fitinfo


# --------------------------------------------------------------------------
# small metric helpers reused by the study
# --------------------------------------------------------------------------
def foldback_residual(P, truth, measured_matched):
    """median |rel residual| of P@truth vs measured over nonzero-measured bins."""
    fold = P @ truth
    m = measured_matched > 0
    r = np.abs(fold[m] - measured_matched[m]) / measured_matched[m]
    return float(np.median(r)) if r.size else np.nan


def oscillation_metric(u, prior, pt_idx, report_pt):
    """median |second difference| of (u/prior) within each reported pt slice."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prior > 0, u / prior, np.nan)
    vals = []
    for pt in report_pt:
        idx = np.where(pt_idx == pt)[0]
        r = ratio[idx]
        r = r[np.isfinite(r)]
        if r.size >= 3:
            d2 = r[2:] - 2 * r[1:-1] + r[:-2]
            vals.extend(np.abs(d2))
    return float(np.median(vals)) if vals else np.nan
