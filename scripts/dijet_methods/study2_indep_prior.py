"""Part 4: PRIOR-DEPENDENCE EVIDENCE for D'Agostini (boss-facing).

Aritra's analysis lead distrusts D'Agostini for prior dependence. This tests it
DIRECTLY: unfold the SAME real data with three different priors and measure how
much the unfolded central value moves.

Priors (per pt slice, in x = log10(rho^2)):
  (i)   nominal   -- the Pythia gen truth spectrum (what the tag uses).
  (ii)  tilt      -- Pythia gen multiplied by a strong linear-in-x tilt, factor
                     0.6 at the slice's low x edge -> 1.4 at its high x edge,
                     then renormalized to the slice's Pythia integral.
  (iii) flat      -- flat-in-x within each pt slice, normalized to the slice's
                     Pythia integral (maximally different shape).

For a converged, prior-independent unfold the three should agree. The spread
(median / p90 / max % over reported bins) at n_iter=4 and n_iter=10 quantifies
exactly how prior-dependent the tag's n=4 result is, and whether more iterations
(the correct fix) reduce it.
"""
from __future__ import annotations

import numpy as np

from scripts.dijet_methods.study2_indep_common import (
    REPORTED_PT, slice_gen, gen_rho_centers,
)
from scripts.dijet_methods import methods as M


def _renorm_slice(prob, base, weights):
    """Return a prior = base * weights, renormalized within each pt slice so the
    slice integral matches the base (Pythia) slice integral. Only touches
    reported slices; other gen bins (sink) keep the base value."""
    out = base.astype(float).copy()
    for pt in REPORTED_PT:
        gidx = slice_gen(prob, pt)
        b = base[gidx]
        w = weights[gidx]
        cand = b * w
        s = cand.sum()
        target = b.sum()
        if s > 0 and target > 0:
            out[gidx] = cand * (target / s)
    return out


def build_priors(prob):
    """Return dict name -> prior vector over gen bins."""
    xc = gen_rho_centers()
    pyth = prob.gen.astype(float)

    # (ii) linear tilt 0.6 -> 1.4 across each slice's populated x range.
    # xc has one entry per gen-rho bin (12); each pt slice contains those 12 rho
    # bins in order, so the per-slice x values are exactly xc.
    tilt_w = np.ones_like(pyth)
    for pt in REPORTED_PT:
        gidx = slice_gen(prob, pt)
        pop = pyth[gidx] > 0
        xs = xc
        xlo, xhi = xs[pop].min(), xs[pop].max()
        frac = (xs - xlo) / max(xhi - xlo, 1e-9)
        tilt_w[gidx] = 0.6 + 0.8 * frac
    prior_tilt = _renorm_slice(prob, pyth, tilt_w)

    # (iii) flat-in-x within slice: weight = 1/pyth so pyth*weight is flat, then
    # renormalized to the slice integral => each populated bin gets equal content.
    flat_w = np.zeros_like(pyth)
    for pt in REPORTED_PT:
        gidx = slice_gen(prob, pt)
        pop = pyth[gidx] > 0
        w = np.zeros(len(gidx))
        w[pop] = 1.0 / pyth[gidx][pop]
        flat_w[gidx] = w
    prior_flat = _renorm_slice(prob, pyth, flat_w)

    return {"nominal_pythia": pyth, "tilt_0p6_1p4": prior_tilt, "flat_in_x": prior_flat}


def _summ(vals):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return {"median_pct": float("nan"), "p90_pct": float("nan"),
                "max_pct": float("nan")}
    return {"median_pct": float(np.median(v)),
            "p90_pct": float(np.percentile(v, 90)),
            "max_pct": float(np.max(v))}


def prior_dependence(prep, n_iters=(4, 10)):
    """Unfold real data with each prior at each n_iter; report the % spread of
    the unfolded result across priors, over reported bins. Returns
    (results_dict, unfolded_by_prior[n_iter])."""
    prob = prep.prob
    priors = build_priors(prob)
    mask = np.zeros(prob.P.shape[1], bool)
    for pt in REPORTED_PT:
        mask[prob.gen_pt_idx == pt] = True

    # Denominator floor per pt slice: relative spreads are only meaningful where
    # the spectrum has appreciable content. Dividing by a near-empty sparse-tail
    # bin gives astronomical, meaningless percentages. We floor the denominator
    # at 1% of the slice's max nominal-prior content, per slice.
    def _slice_floor(nom):
        floor = np.zeros_like(nom)
        for pt in REPORTED_PT:
            gidx = slice_gen(prob, pt)
            mx = nom[gidx].max()
            floor[gidx] = 0.01 * mx if mx > 0 else 1.0
        return floor

    out = {}
    unf_store = {}
    for n in n_iters:
        unf = {name: M.dagostini(prob, prep.data_matched, n_iter=n, prior=pr.copy())
               for name, pr in priors.items()}
        unf_store[n] = unf
        nom = unf["nominal_pythia"]
        den = np.maximum(nom, _slice_floor(nom))
        stack = np.vstack([unf[name] for name in priors])
        spread = (stack.max(axis=0) - stack.min(axis=0)) / den * 100
        per_prior = {}
        for name in priors:
            if name == "nominal_pythia":
                continue
            rel = np.abs(unf[name] - nom) / den * 100
            per_prior[f"{name}_vs_nominal"] = _summ(rel[mask])
        out[f"n_iter_{n}"] = {
            "max_minus_min_spread": _summ(spread[mask]),
            **per_prior,
        }
    return out, unf_store
