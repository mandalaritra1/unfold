"""Running-coupling resummed jet-mass curves in the analysis rho convention.

This upgrades the fixed-coupling LL cartoon (``theory_ll``) to the
running-coupling resummation of the soft-drop jet mass in the spirit of
Marzani-Schunk-Soyez (arXiv:1712.05105) and the textbook treatment in
"Looking Inside Jets" (arXiv:1901.10342):

* the Sudakov radiator is integrated **numerically** over the (z, theta) Lund
  plane with a two-loop running coupling in the CMW scheme, evaluated at the
  emission kt = z * theta * pT * R and frozen below ``freeze`` (default 1 GeV);
* **full splitting functions** are used (not just the soft 1/z pole), which
  automatically includes the hard-collinear B_i terms and the finite-z_cut
  corrections to the groomed plateau;
* the NLL **multiple-emission factor** exp(-gammaE R') / Gamma(1 + R') is
  applied (important for the ungroomed Sudakov peak; a small correction for
  soft drop beta = 0, which is single-logarithmic below the transition);
* the perturbative uncertainty is estimated by varying the renormalization
  scale kt -> {kt/2, 2 kt} and taking the envelope of the resulting shapes.

Accuracy, stated honestly: this is NLL-level resummation with finite-z_cut
effects, WITHOUT matching to fixed order (the ρ -> 1 endpoint region is only
LL-accurate) and WITHOUT non-perturbative corrections (the frozen coupling is
a model, not a hadronization correction). It is a genuine calculation -- the
same physics as the central resummed curves in the literature -- but it is not
the NLO+NLL' + NP paper-grade prediction, which requires fixed-order matching.

Conventions: everything is a function of the analysis abscissa
x = log10(rho^2) with rho = m/(pT R), i.e. x = log10(rho_th) for the textbook
rho_th = m^2/(pT R)^2. The soft-drop (beta = 0) transition sits at
x = log10(z_cut) = -1.

Observable phase space: an emission (z, theta) [theta in units of R] gives
rho_th = z theta^2 and kt = z theta pT R. The groomed veto region is
{z theta^2 > rho_th} ∩ {z > z_cut}; ungroomed drops the z_cut condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

CF = 4.0 / 3.0
CA = 3.0
TR = 0.5
NF = 5

JET_R = 0.8
ZCUT = 0.1

ALPHAS_MZ = 0.118
M_Z = 91.1876
_EULER_GAMMA = 0.5772156649015329

_B0 = (33.0 - 2.0 * NF) / (12.0 * np.pi)
_B1 = (153.0 - 19.0 * NF) / (24.0 * np.pi**2)
# CMW / Catani-Marchesini-Webber K factor (soft physical coupling scheme).
_K_CMW = CA * (67.0 / 18.0 - np.pi**2 / 6.0) - 10.0 / 9.0 * NF * TR

_LN10 = np.log(10.0)


def alpha_s(mu, freeze=1.0):
    """Two-loop running alpha_s(mu), nf = 5, frozen below ``freeze`` GeV."""
    mu = np.maximum(np.asarray(mu, dtype=float), freeze)
    t = np.log(mu**2 / M_Z**2)
    w = 1.0 + ALPHAS_MZ * _B0 * t
    # Two-loop iterative form; w > 0 guaranteed for mu > Lambda, which the
    # freeze floor enforces for any sane freeze value.
    return (ALPHAS_MZ / w) * (1.0 - (_B1 / _B0) * ALPHAS_MZ * np.log(w) / w)


def _alpha_cmw(kt, mu_fac=1.0, freeze=1.0):
    """CMW-scheme effective coupling at the (varied) emission scale."""
    a = alpha_s(mu_fac * kt, freeze=freeze)
    return a * (1.0 + a * _K_CMW / (2.0 * np.pi))


def _zp(z, flavor):
    """z * P_i(z): full splitting functions (soft pole + hard collinear).

    Integrating these over (z_min, 1) reproduces 2 C_i ln(1/z_min) + 2 C_i B_i,
    i.e. the hard-collinear B terms come out automatically:
    B_q = -3/4, B_g = -(11 CA - 4 nf TR)/(12 CA).
    """
    if flavor == "quark":
        return CF * (1.0 + (1.0 - z) ** 2)
    if flavor == "gluon":
        return z * (CA * (2.0 * (1.0 - z) / z + z * (1.0 - z))
                    + NF * TR * (z**2 + (1.0 - z) ** 2))
    raise ValueError(flavor)


@dataclass(frozen=True)
class ResummedCurve:
    x: np.ndarray
    y: np.ndarray          # dSigma/dx, unit area over the x grid
    sigma_below: float     # probability below the grid floor (should be ~0)
    alphas_hard: float     # alpha_s(pT R) for labeling
    flavor: str
    groomed: bool
    mu_fac: float


class _RadiatorTable:
    """Numerical radiator R(x) for one (flavor, pT, mu_fac, grooming).

    Grid strategy: on a (u = ln theta^2, v = ln z) grid, precompute
    F(u, a) = int_{ln a}^{0} dv [z P(z)] alpha_cmw(kt)/(2 pi), a reversed
    cumulative integral along v for each u. Then
    R(rho) = int du F(u, max(rho/theta^2, z_cut-or-0)), a 1D interpolation
    per rho -- fast and accurate.
    """

    def __init__(self, flavor, pt, groomed, mu_fac=1.0, jet_r=JET_R,
                 zcut=ZCUT, freeze=1.0, n_u=1200, n_v=6000):
        # n_v sets the smoothness of the final curve: R(x) is piecewise linear
        # in the v grid through the F interpolation, and the density takes two
        # derivatives of it, so a coarse v grid shows up as high-frequency
        # wiggles on dSigma/dx. 6000 points keeps the residual well below the
        # line width.
        self.groomed = groomed
        self.zcut = zcut
        u = np.linspace(np.log(1e-7), 0.0, n_u)          # ln theta^2
        v = np.linspace(np.log(1e-7), 0.0, n_v)          # ln z
        du = u[1] - u[0]
        dv = v[1] - v[0]
        z = np.exp(v)
        theta = np.exp(0.5 * u)
        kt = np.outer(theta, z) * pt * jet_r             # (n_u, n_v)
        w = _zp(z, flavor)[None, :] * _alpha_cmw(kt, mu_fac, freeze) / (2.0 * np.pi)
        # F[i, j] = integral over v from v[j] to 0 (reversed cumulative sum).
        F = np.concatenate(
            [np.cumsum((w[:, ::-1][:, :-1] + w[:, ::-1][:, 1:]) * 0.5 * dv, axis=1)[:, ::-1],
             np.zeros((n_u, 1))],
            axis=1,
        )
        self._u, self._v, self._du, self._F = u, v, du, F
        self._w = w
        # The z lower limit is max(rho/theta^2, floor): its kink crosses the u
        # grid as rho scans, and a naive grid sum turns that into a sawtooth on
        # R'' (visible as wiggles on the final curve). Split the u integral
        # exactly at the kink u* = ln(rho) - floor instead:
        #   u > u*: vmin = floor        -> precomputed G(u), cumulative C(u)
        #   u < u*: vmin = ln(rho) - u  -> smooth integrand, trapezoid + exact
        #                                  endpoint at (u*, G(u*)).
        self._floor = np.log(zcut) if groomed else v[0]
        self._G = self._F_at(np.full(n_u, self._floor))
        # C(u_i) = integral of G from u_i to 0.
        rev = 0.5 * (self._G[:-1] + self._G[1:]) * du
        self._C = np.concatenate([np.cumsum(rev[::-1])[::-1], [0.0]])

    def _F_at(self, vmin):
        """F(u_i, vmin_i) by linear interpolation along v, vectorized in u."""
        v, F = self._v, self._F
        vmin = np.clip(vmin, v[0], v[-1])
        j = np.clip(np.searchsorted(v, vmin) - 1, 0, len(v) - 2)
        frac = (vmin - v[j]) / (v[1] - v[0])
        rows = np.arange(len(self._u))
        return F[rows, j] * (1.0 - frac) + F[rows, j + 1] * frac

    def __call__(self, x):
        """R at analysis abscissa x = log10(rho_th), vectorized."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        ln_rho = x * _LN10
        u, du = self._u, self._du
        out = np.empty_like(x)
        for k, lr in enumerate(ln_rho):
            ustar = np.clip(lr - self._floor, u[0], 0.0)
            # Angle-limited piece: vmin = lr - u on grid nodes in (lr, u*).
            # Below u = lr the z lower limit exceeds 1 (no phase space, F = 0);
            # insert that boundary as an exact node too.
            ulo = np.clip(lr, u[0], ustar)
            m = (u < ustar) & (u > ulo)
            un = np.concatenate([[ulo], u[m], [ustar]])
            gstar = float(np.interp(ustar, u, self._G))
            fn = np.concatenate([[0.0], self._F_at(lr - u)[m], [gstar]])
            i1 = np.trapezoid(fn, un) if len(un) > 1 else 0.0
            # Floor-limited piece: exact cumulative of G from u* to 0.
            i2 = float(np.interp(ustar, u, self._C))
            out[k] = i1 + i2
        return out

    def _w_at(self, vmin):
        """w(u_i, vmin_i) by linear interpolation along v, vectorized in u."""
        v, w = self._v, self._w
        vmin = np.clip(vmin, v[0], v[-1])
        j = np.clip(np.searchsorted(v, vmin) - 1, 0, len(v) - 2)
        frac = (vmin - v[j]) / (v[1] - v[0])
        rows = np.arange(len(self._u))
        return w[rows, j] * (1.0 - frac) + w[rows, j + 1] * frac

    def rprime(self, x):
        """Exact R' = dR/dln(1/rho): a boundary integral, not a finite
        difference.

        Only the angle-limited piece (vmin = ln rho - u) depends on rho, so
        dR/dL = int_{u < u*} w(u, ln rho - u) du. Computing this analytically
        keeps the multiple-emission factor exp(-gammaE R')/Gamma(1+R') free of
        the grid-crossing sawtooth a numerical derivative of R picks up.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        ln_rho = x * _LN10
        u = self._u
        out = np.empty_like(x)
        for k, lr in enumerate(ln_rho):
            ustar = np.clip(lr - self._floor, u[0], 0.0)
            # w vanishes for vmin = lr - u >= 0 (z limit above 1): integrate
            # only u in (lr, u*), with both boundaries as exact nodes -- the
            # v-clip in _w_at would otherwise fake a w(z=1) contribution from
            # the empty-phase-space angles.
            ulo = np.clip(lr, u[0], ustar)
            m = (u < ustar) & (u > ulo)
            wcol = self._w_at(lr - u)
            un = np.concatenate([[ulo], u[m], [ustar]])
            wn = np.concatenate(
                [[float(np.interp(ulo, u, wcol))], wcol[m],
                 [float(np.interp(ustar, u, wcol))]]
            )
            out[k] = np.trapezoid(wn, un) if len(un) > 1 else 0.0
        return out


def resummed_density(flavor, groomed, pt, mu_fac=1.0, jet_r=JET_R, zcut=ZCUT,
                     freeze=1.0, x_lo=-7.0, n_x=700):
    """dSigma/dx on a wide grid, normalized to unit area over the full range.

    The NLL cumulant is Sigma(x) = exp(-R - gammaE R') / Gamma(1 + R') with
    R' = dR/dln(1/rho); the density is its x derivative. The grid extends to
    ``x_lo`` where Sigma is negligible, so the full-range area is ~1 and
    per-flavor curves can be mixed with event fractions before any
    window renormalization.
    """
    table = _RadiatorTable(flavor, pt, groomed, mu_fac, jet_r, zcut, freeze)
    x = np.linspace(x_lo, 0.0, n_x)
    R = table(x)
    Rp = np.maximum(table.rprime(x), 0.0)                # exact dR/dL
    log_sigma = -R - _EULER_GAMMA * Rp - gammaln(1.0 + Rp)
    sigma = np.exp(log_sigma)
    y = np.gradient(sigma, x)
    y = np.maximum(y, 0.0)
    area = np.trapezoid(y, x)
    return ResummedCurve(
        x=x, y=y / area, sigma_below=float(sigma[0]),
        alphas_hard=float(alpha_s(pt * jet_r, freeze)),
        flavor=flavor, groomed=groomed, mu_fac=mu_fac,
    )


# (mu_fac, freeze/GeV) variation set: renormalization scale x/2 around the
# emission kt, plus the IR freeze scale 0.5..2 GeV. The freeze variation is a
# crude probe of the non-perturbative sensitivity of the unmatched curve; a
# mu_R-only band on a shape-normalized single-log observable is misleadingly
# narrow because normalization cancels the overall rate shift.
DEFAULT_VARIATIONS = ((1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0))


def mixed_prediction(window, groomed, pt, f_quark, variations=DEFAULT_VARIATIONS,
                     **kwargs):
    """Quark/gluon mixture with a variation-envelope band, on the shown window.

    Per (mu_fac, freeze) variation: mix full-range-normalized flavor densities
    with the event fractions (f_quark from the analysis MC), crop to ``window``
    and renormalize to unit area there -- the same normalization as the
    published data. The first entry of ``variations`` is the central setting.
    Returns dict with x, central, lo, hi (envelope), and the per-flavor
    central shapes (window-normalized) for optional display.
    """
    lo_edge, hi_edge = window
    shapes = []
    flavors_central = {}
    for iv, (mu_fac, freeze) in enumerate(variations):
        q = resummed_density("quark", groomed, pt, mu_fac, freeze=freeze, **kwargs)
        g = resummed_density("gluon", groomed, pt, mu_fac, freeze=freeze, **kwargs)
        mix = f_quark * q.y + (1.0 - f_quark) * g.y
        m = (q.x >= lo_edge) & (q.x <= hi_edge)
        xw = q.x[m]
        yw = mix[m] / np.trapezoid(mix[m], xw)
        shapes.append(yw)
        if iv == 0:
            for name, curve in (("quark", q), ("gluon", g)):
                yf = curve.y[m]
                flavors_central[name] = yf / np.trapezoid(yf, xw)
            x_out = xw
            alphas_hard = q.alphas_hard
    band = np.vstack(shapes)
    return {
        "x": x_out,
        "central": shapes[0],
        "lo": band.min(axis=0),
        "hi": band.max(axis=0),
        "flavors": flavors_central,
        "f_quark": f_quark,
        "alphas_hard": alphas_hard,
    }
