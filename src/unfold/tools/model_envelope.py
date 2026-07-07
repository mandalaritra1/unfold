"""Shower/hadronization model uncertainty via offline column-scaled responses.

Method (validated 2026-07-07, see ai-wiki bug
``zjet_casa_reskim_reweight_contamination`` and research-notes
``zjet_rho_reunfold_model_envelope``):

The gen reweights (Vincia, CR1/CR2, frag hard/soft) are piecewise-constant on
the 24-bin gen rho axis, so reweighting the generator per event is *exactly*
a gen-column scaling of the nominal response::

    resp_var[reco, gen24] = resp_nom[reco, gen24] * w(gen24)

We build each varied response from the high-statistics fine 2018 nominal
(``inputs/zjet/rho/finebins/minimal_rho_fine_pythia_2018.pkl``, 48-gen x
96-reco, coarsened by pairs to 24 x 48), re-unfold the same data through it at
the spec's settings (production: tau=0), and take the per-bin shift of the
normalized result versus the w=1 (offline-nominal) baseline. The 2018-only
response is fine here: the era mismatch cancels in the var/nom ratio, and the
offline-nominal baseline removes the residual (checked at the 1-2% level).

The weighters live in ``inputs/zjet/rho/model_weighters/`` (see its README for
provenance). FSR is *not* computed here -- it comes from the stored PSWeight
response variations already unfolded by the Unfolder; the caller maxes it into
the envelope (see ``Unfolder._compute_total_systematic``).
"""

import pickle
from pathlib import Path

import numpy as np

from unfold.utils.merge_helpers import (
    merge_mass_flat,
    mosaic_no_padding,
    reorder_to_expected,
    unflatten_gen_by_pt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FINE_PKL = REPO_ROOT / "inputs/zjet/rho/finebins/minimal_rho_fine_pythia_2018.pkl"
WEIGHTER_DIR = REPO_ROOT / "inputs/zjet/rho/model_weighters"

MODEL_SOURCES = ("vincia", "cr1", "cr2", "fraghard", "fragsoft")
# Representative pt per ptgen bin (edges [0, 200, 290, 400, 13000]); the
# weighter only uses pt to pick its pt bin, which shares those edges.
_PTMID = np.array([100.0, 245.0, 345.0, 700.0])


def _safe_ratio(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def _weighter_w(source, groomed, centers):
    """Per-ptgen-bin weight arrays w(gen rho) on the 24-bin centers."""
    tag = "groomed" if groomed else "ungroomed"
    z = np.load(WEIGHTER_DIR / f"{source}_rho_reweight_{tag}.npz", allow_pickle=True)
    pt_edges = np.asarray(z["pt_edges"], float)
    rho_grids = z["rho_grids"]
    w_grids = z["w_grids"]
    out = np.ones((len(_PTMID), len(centers)))
    for i, pt in enumerate(_PTMID):
        j = int(np.clip(np.searchsorted(pt_edges, pt, side="right") - 1, 0,
                        len(rho_grids) - 1))
        out[i] = np.interp(centers, np.asarray(rho_grids[j], float),
                           np.asarray(w_grids[j], float))
    return out


def _coarsened_nominal(groomed):
    """Fine 2018 nominal coarsened to (ptreco, mreco48, ptgen, mgen24) + gen."""
    tag = "g" if groomed else "u"
    with open(FINE_PKL, "rb") as handle:
        d = pickle.load(handle)
    h = d[f"response_matrix_rho_{tag}"]
    v = h.view(flow=False)["value"].sum(axis=0)[..., 0]
    v = v[:, ::2] + v[:, 1::2]                       # reco 96 -> 48
    v = v[:, :, :, ::2] + v[:, :, :, 1::2]           # gen  48 -> 24
    e24 = np.asarray(h.axes["mpt_gen"].edges)[::2]
    gh = d[f"ptjet_rhojet_{tag}_gen"]
    gv = gh.view(flow=False)["value"].sum(axis=0)[..., 0]
    gv = gv[:, ::2] + gv[:, 1::2]
    return v, gv, e24


def _mosaic_inputs(uf, resp, gen, e24):
    """Turn a (ptreco, mreco48, ptgen, mgen24) response + gen hist into the
    (mosaic, misses_flat, fake_fraction) triple ``_perform_unfold`` needs."""
    v = resp[:, ::2] + resp[:, 1::2]                 # reco 48 -> 24 analysis bins
    Mv = np.transpose(v, (2, 3, 0, 1))               # (ptgen, mgen24, ptreco, mreco24)
    M2d, _ = reorder_to_expected(Mv, uf.edges, uf.pt_edges, e24)
    mosaic, _ = mosaic_no_padding(M2d, uf.edges, e24, uf.reco_edges_by_pt,
                                  uf.gen_edges_by_pt)
    misses = np.clip(gen - Mv.sum(axis=(2, 3)), 0.0, None).T
    misses_flat = merge_mass_flat(misses, e24, uf.gen_edges_by_pt)
    fake_fraction = np.clip(
        _safe_ratio(uf.fakes_2d, mosaic.sum(axis=1) + uf.fakes_2d), 0.0, 1.0)
    return mosaic, misses_flat, fake_fraction


def _unfold_with(uf, mosaic, misses_flat, fake_fraction, key):
    """Re-unfold the same data with a swapped response (no nominal clobber)."""
    uf.misses_2d_dict = getattr(uf, "misses_2d_dict", {})
    uf.misses_2d_dict[key] = misses_flat
    saved_ff = uf.fake_fraction_2d
    uf.fake_fraction_2d = fake_fraction
    try:
        uf._perform_unfold(systematic=key, resp_np=mosaic)
    finally:
        uf.fake_fraction_2d = saved_ff
    return np.asarray(uf.y_unf_dict[key], float)


def compute_model_shifts(uf):
    """Per-source, per-pt-bin fractional shifts of the normalized result.

    Returns ``{source: {i: |frac shift| array}}`` for ``i`` indexing
    ``uf.gen_edges_by_pt`` (same convention as ``normalized_results``).
    Shifts are on the per-pT normalized shape, so they apply directly to
    ``normalized_results[i]['unfolded']``.
    """
    resp_nom, gen_nom, e24 = _coarsened_nominal(uf.groomed)
    eb = uf.gen_edges_by_pt
    n_pt = len(eb)

    def _norm_by_pt(y_flat):
        y = unflatten_gen_by_pt(np.asarray(y_flat, float), eb)
        return {i: _safe_ratio(y[i], y[i].sum()) for i in range(n_pt)}

    y_base = _norm_by_pt(_unfold_with(
        uf, *_mosaic_inputs(uf, resp_nom, gen_nom, e24), key="model_nom"))

    centers = 0.5 * (e24[:-1] + e24[1:])
    shifts = {}
    for source in MODEL_SOURCES:
        w = _weighter_w(source, uf.groomed, centers)      # (ptgen, 24)
        resp_var = resp_nom * w[None, None, :, :]
        gen_var = gen_nom * w
        y_var = _norm_by_pt(_unfold_with(
            uf, *_mosaic_inputs(uf, resp_var, gen_var, e24),
            key=f"model_{source}"))
        shifts[source] = {
            i: np.abs(_safe_ratio(y_var[i], y_base[i]) - 1.0)
            for i in range(n_pt)
        }
    return shifts


def group_model_shifts(shifts, n_pt):
    """Group raw source shifts into the reported components.

    CR = max(cr1, cr2); frag = max(fraghard, fragsoft); Vincia as-is.
    Returns ``{component: {i: frac array}}``.
    """
    grouped = {"Vincia": shifts["vincia"]}
    grouped["CR"] = {i: np.maximum(shifts["cr1"][i], shifts["cr2"][i])
                     for i in range(n_pt)}
    grouped["frag"] = {i: np.maximum(shifts["fraghard"][i], shifts["fragsoft"][i])
                       for i in range(n_pt)}
    return grouped
