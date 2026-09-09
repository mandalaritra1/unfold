"""Analytic LL jet-mass curves expressed in *this analysis'* rho convention.

Origin
------
Adapted from the standalone theory scripts ``sudakov.py`` / ``softdrop.py``
(fixed-coupling LL Sudakov for plain and soft-drop jet mass, quark vs gluon).
Those scripts are written in the textbook soft-drop variable

    rho_th = m^2 / (pT R)^2                     [Larkoski et al., Marzani et al.]

and plot ``log10(rho_th^2)``. This analysis instead uses

    rho = m / (pT R)          and plots   x = log10(rho^2) = log10(rho_th)

so the two x axes -- despite carrying the *same* ``log10(rho^2)`` label --
differ by a factor of two:

    x_theory_script = 2 * x_analysis

Concretely the soft-drop transition ``rho_th = z_cut`` sits at -2 on the
script's axis but at **-1** on ours, and the script's -8..0 range covers only
our -4..0. Everything below is written directly in the analysis variable; the
factor of two is absorbed once, in :func:`_log_rho_inv`.

Three further adjustments make the curves comparable to the unfolded result:

* **Normalization window.** The unfolded spectra are normalized over the shown
  per-pT window only (``normalize_over_shown=True`` in ``RHO_ARC_R2_SPEC``), not
  over all of rho. :func:`ll_density` renormalizes to unit area over the same
  window, otherwise the theory curve sits low by 15-45% depending on grooming.
* **Running coupling.** The scripts use a single fixed ``alpha_s = 0.12`` for
  every pT. Here the default is one-loop running evaluated at the jet scale
  ``mu = pT R``, so each pT bin gets its own curve. Pass ``alpha_s=0.12`` to
  recover the original fixed-coupling behaviour.
* **Flavor.** Z+jet is quark-enriched but mixed, so the quark (C_F) and gluon
  (C_A) curves are meant as a bracket around the data, not as predictions.

Validity: this is LL with fixed coupling inside each pT bin, no non-perturbative
corrections. Below roughly rho ~ Lambda_QCD/(pT R) the curves are meaningless --
that is exactly where the fixed-mass NP bump (m ~ 7-25 GeV) lives in our data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CF = 4.0 / 3.0
CA = 3.0

JET_R = 0.8
ZCUT = 0.1

# One-loop running coupling, nf = 5.
ALPHAS_MZ = 0.118
M_Z = 91.1876
_NF = 5
_BETA0 = (33.0 - 2.0 * _NF) / (12.0 * np.pi)

_LN10 = np.log(10.0)


def alpha_s(mu):
    """One-loop alpha_s(mu), nf = 5, normalized to alpha_s(m_Z) = 0.118."""
    mu = np.asarray(mu, dtype=float)
    return ALPHAS_MZ / (1.0 + ALPHAS_MZ * _BETA0 * np.log(mu**2 / M_Z**2))


def _log_rho_inv(x):
    """L = ln(1/rho_th) from the analysis abscissa x = log10(rho^2).

    rho_th = m^2/(pT R)^2 = rho^2, hence ln(1/rho_th) = -x ln(10). This single
    line is where the factor-of-two axis convention is absorbed.
    """
    return -np.asarray(x, dtype=float) * _LN10


def _dsigma_dx(p_dlnrho):
    """Convert d/dln(rho_th) to d/dx with x = log10(rho_th)."""
    return p_dlnrho * _LN10


def ungroomed_density(x, color_factor, alphas):
    """Plain (ungroomed) jet mass at LL, unnormalized, in d/dx."""
    L = _log_rho_inv(x)
    a = color_factor * alphas / (2.0 * np.pi)
    return _dsigma_dx((color_factor * alphas / np.pi) * L * np.exp(-a * L * L))


def softdrop_density(x, color_factor, alphas, zcut=ZCUT):
    """Soft drop beta = 0 jet mass at LL, unnormalized, in d/dx.

    Above the transition (rho_th > zcut, i.e. x > log10(zcut) = -1) grooming is
    inactive and the plain-mass Sudakov applies; below it the distribution
    flattens to the ``alpha_s C_i ln(1/zcut)`` plateau.
    """
    L = _log_rho_inv(x)
    lcut = np.log(1.0 / zcut)
    b = color_factor * alphas / np.pi
    p = np.empty_like(L)
    plain = L < lcut
    groomed = ~plain
    p[plain] = b * L[plain] * np.exp(-0.5 * b * L[plain] ** 2)
    p[groomed] = b * lcut * np.exp(-b * lcut * (L[groomed] - 0.5 * lcut))
    return _dsigma_dx(p)


def softdrop_transition(zcut=ZCUT):
    """Analysis-convention x at which soft drop turns on (rho_th = zcut)."""
    return np.log10(zcut)


@dataclass(frozen=True)
class LLCurve:
    """An LL curve sampled on the analysis abscissa, unit area over its window."""

    x: np.ndarray
    y: np.ndarray
    window: tuple[float, float]
    alphas: float
    color_factor: float
    groomed: bool


def ll_density(
    window,
    groomed,
    color_factor,
    pt=None,
    alphas=None,
    jet_r=JET_R,
    zcut=ZCUT,
    n_points=2000,
):
    """LL curve on ``window``, renormalized to unit area over that same window.

    Parameters
    ----------
    window : (float, float)
        Analysis-convention ``log10(rho^2)`` range to normalize and sample over.
        Use the *shown* per-pT window, to match ``normalize_over_shown=True``.
    groomed : bool
        Soft drop beta = 0 if True, plain mass otherwise.
    color_factor : float
        :data:`CF` or :data:`CA`.
    pt : float, optional
        Representative jet pT; sets the coupling via ``mu = pT * jet_r``.
        Ignored when ``alphas`` is given explicitly.
    alphas : float, optional
        Fixed coupling. Pass ``0.12`` to reproduce the original scripts.
    """
    if alphas is None:
        if pt is None:
            raise ValueError("give either pt (running coupling) or alphas (fixed)")
        alphas = float(alpha_s(pt * jet_r))

    lo, hi = window
    x = np.linspace(lo, hi, n_points)
    kernel = softdrop_density if groomed else ungroomed_density
    y = kernel(x, color_factor, alphas, zcut) if groomed else kernel(x, color_factor, alphas)

    area = np.trapezoid(y, x)
    if area <= 0:
        raise ValueError(f"non-positive area {area} over window {window}")
    return LLCurve(
        x=x,
        y=y / area,
        window=(lo, hi),
        alphas=alphas,
        color_factor=color_factor,
        groomed=groomed,
    )


def flavor_bracket(window, groomed, pt=None, alphas=None, **kwargs):
    """Convenience: ``(quark_curve, gluon_curve)`` on a common abscissa."""
    quark = ll_density(window, groomed, CF, pt=pt, alphas=alphas, **kwargs)
    gluon = ll_density(window, groomed, CA, pt=pt, alphas=alphas, **kwargs)
    return quark, gluon
