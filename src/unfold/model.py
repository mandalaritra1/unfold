"""Modelling uncertainty and prediction statistics.

Three pieces, all pure numpy apart from the pickle read of the fine response:

1. offline column-scaled responses (Vincia, CR1/CR2, frag hard/soft) for the
   Z+jet shower/hadronization envelope;
2. the two-group model covariance from origin-centered enclosing ellipsoids
   (pair-split);
3. the statistical covariance of a prediction normalized in a fixed window.

Part 1 -- column-scaled responses

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

Part 2 -- enclosing template covariance
Two-group model covariance from origin-centered enclosing ellipsoids.

Alternatives within a group are bounded templates, not independent Gaussian
nuisances. Each group's unit ellipsoid contains every supplied signed shift.
Adding the two groups is the adopted independent-group quadrature convention;
it is not a simultaneous worst-case bound or a calibrated confidence region.
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from unfold.histmath import (
    merge_mass_flat,
    mosaic_no_padding,
    reorder_to_expected,
    unflatten_gen_by_pt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    e48 = np.asarray(h.axes["mpt_reco"].edges)[::2]
    gh = d[f"ptjet_rhojet_{tag}_gen"]
    gv = gh.view(flow=False)["value"].sum(axis=0)[..., 0]
    gv = gv[:, ::2] + gv[:, 1::2]
    return v, gv, e24, e48


def _mosaic_inputs(uf, resp, gen, e24, e48):
    """Turn a (ptreco, mreco48, ptgen, mgen24) response + gen hist into the
    (mosaic, misses_flat, fake_fraction) triple ``_perform_unfold`` needs.

    The reco side stays at the 48-bin half-analysis binning and is merged into
    ``uf.reco_edges_by_pt`` by the mosaic builder, so any spec whose reported
    reco edges are a subset of the 48-edge set works unchanged (the legacy
    24-bin binning and the arc_r2 28-bin buffer binning both are)."""
    Mv = np.transpose(resp, (2, 3, 0, 1))            # (ptgen, mgen24, ptreco, mreco48)
    M2d, _ = reorder_to_expected(Mv, e48, uf.pt_edges, e24)
    mosaic, _ = mosaic_no_padding(M2d, e48, e24, uf.reco_edges_by_pt,
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

    Returns ``{source: {i: signed frac shift array}}`` for ``i`` indexing
    ``uf.gen_edges_by_pt`` (same convention as ``normalized_results``).
    Shifts are on the per-pT normalized shape, so they apply directly to
    ``normalized_results[i]['unfolded']``.
    """
    resp_nom, gen_nom, e24, e48 = _coarsened_nominal(uf.groomed)
    eb = uf.gen_edges_by_pt
    n_pt = len(eb)

    def _norm_by_pt(y_flat):
        y = unflatten_gen_by_pt(np.asarray(y_flat, float), eb)
        return {i: _safe_ratio(y[i], y[i].sum()) for i in range(n_pt)}

    y_base = _norm_by_pt(_unfold_with(
        uf, *_mosaic_inputs(uf, resp_nom, gen_nom, e24, e48), key="model_nom"))

    centers = 0.5 * (e24[:-1] + e24[1:])
    shifts = {}
    for source in MODEL_SOURCES:
        w = _weighter_w(source, uf.groomed, centers)      # (ptgen, 24)
        resp_var = resp_nom * w[None, None, :, :]
        gen_var = gen_nom * w
        y_var = _norm_by_pt(_unfold_with(
            uf, *_mosaic_inputs(uf, resp_var, gen_var, e24, e48),
            key=f"model_{source}"))
        shifts[source] = {
            i: _safe_ratio(y_var[i], y_base[i]) - 1.0
            for i in range(n_pt)
        }
    return shifts


def compute_prepared_model_shifts(uf):
    """Read model shifts already re-unfolded through prepared responses.

    Pair-split inputs build ``model_<source>`` response categories on their
    stored fine generator coordinate before the analysis GEN-bin merge.  The
    normal Unfolder systematic loop re-unfolds those responses, so this helper
    only converts their normalized results into the signed fractional shifts
    consumed by the shared Z+jet two-leg envelope.
    """

    shifts = {source: {} for source in MODEL_SOURCES}
    for i, result in enumerate(uf.normalized_results):
        nominal = np.asarray(result["unfolded"], dtype=float)
        varied_results = uf.normalized_systematics[i]["unfolded"]
        for source in MODEL_SOURCES:
            key = f"model_{source}"
            if key not in varied_results:
                raise KeyError(
                    f"prepared model envelope is missing unfolded variation {key!r}"
                )
            varied = np.asarray(varied_results[key], dtype=float)
            shifts[source][i] = np.divide(
                varied - nominal,
                nominal,
                out=np.zeros_like(nominal),
                where=nominal != 0.0,
            )
    return shifts


def compute_model_reco_shifts(uf):
    """Per-source fractional shifts of the normalized reco-level MC projection.

    Same column-scaled varied responses as :func:`compute_model_shifts`, but no
    re-unfold: the shift is on the reco projection ``mosaic.sum(axis=1)`` of
    the varied response versus the offline w=1 baseline, each normalized per
    pT slice. The offline baseline plays the same role as in the unfolded-level
    shifts (the 2018-only era mismatch cancels in the var/nom ratio).

    Returns ``{source: {i: signed frac shift array}}`` for ``i`` indexing
    ``uf.reco_edges_by_pt``.
    """
    resp_nom, gen_nom, e24, e48 = _coarsened_nominal(uf.groomed)
    eb = uf.reco_edges_by_pt
    n_pt = len(eb)

    def _norm_by_pt(reco_flat):
        y = unflatten_gen_by_pt(np.asarray(reco_flat, float), eb)
        return {i: _safe_ratio(y[i], y[i].sum()) for i in range(n_pt)}

    mosaic_nom, _, _ = _mosaic_inputs(uf, resp_nom, gen_nom, e24, e48)
    y_base = _norm_by_pt(mosaic_nom.sum(axis=1))

    centers = 0.5 * (e24[:-1] + e24[1:])
    shifts = {}
    for source in MODEL_SOURCES:
        w = _weighter_w(source, uf.groomed, centers)      # (ptgen, 24)
        resp_var = resp_nom * w[None, None, :, :]
        gen_var = gen_nom * w
        mosaic_var, _, _ = _mosaic_inputs(uf, resp_var, gen_var, e24, e48)
        y_var = _norm_by_pt(mosaic_var.sum(axis=1))
        shifts[source] = {
            # Zero (not -1) where the offline baseline projection is empty
            # (e.g. the [-10,-8] catch-all bin): no shift information there.
            i: np.where(y_base[i] > 0,
                        _safe_ratio(y_var[i], y_base[i]) - 1.0, 0.0)
            for i in range(n_pt)
        }
    return shifts


def vincia_truth_by_pt(uf):
    """True standalone-Vincia gen prediction, normalized like the plotted MC.

    Histograms the path-B Vincia gen cache (60k selected events, CMS_ZJET_JETMASS
    gen selection) into ``uf.gen_edges_by_pt`` per pT slice. Returns
    ``{i: (density, stat_err)}`` in the same per-bin-width normalized units as
    ``normalized_results[i]['true']``, or None when the cache is absent.
    """
    cache = WEIGHTER_DIR / "vincia_gen_cache.npz"
    if not cache.exists():
        return None
    z = np.load(cache)
    rho = z["v_rho_g"] if uf.groomed else z["v_rho_u"]
    jet_pt = z["v_jet_pt"]
    w = z["v_weight"]
    pt_edges = np.asarray(uf.pt_edges, float)
    out = {}
    for i in range(len(uf.gen_edges_by_pt)):
        edges = np.asarray(uf.gen_edges_by_pt[i], float)
        m = (jet_pt >= pt_edges[i]) & (jet_pt < pt_edges[i + 1]) & np.isfinite(rho)
        h, _ = np.histogram(rho[m], bins=edges, weights=w[m])
        h2, _ = np.histogram(rho[m], bins=edges, weights=w[m] ** 2)
        widths = np.diff(edges)
        total = uf._shown_norm_total(h, i)
        if total <= 0:
            out[i] = (np.zeros(len(widths)), np.zeros(len(widths)))
            continue
        out[i] = (h / widths / total, np.sqrt(h2) / widths / total)
    return out


def group_model_shifts(shifts, n_pt):
    """Group raw source shifts into the reported components.

    CR = max(cr1, cr2); frag = max(fraghard, fragsoft); Vincia as-is.
    Returns ``{component: {i: frac array}}``.
    """
    grouped = {"Vincia": {i: np.abs(shifts["vincia"][i]) for i in range(n_pt)}}
    grouped["CR"] = {i: np.maximum(np.abs(shifts["cr1"][i]), np.abs(shifts["cr2"][i]))
                     for i in range(n_pt)}
    grouped["frag"] = {i: np.maximum(np.abs(shifts["fraghard"][i]), np.abs(shifts["fragsoft"][i]))
                       for i in range(n_pt)}
    return grouped


# ---------------------------------------------------------------------------
# Part 2: two-group model covariance
# ---------------------------------------------------------------------------



MODEL_GROUPS = {
    "parton_shower": ("model_vincia", "fsrUp", "fsrDown"),
    "hadronization": ("model_cr1", "model_cr2", "model_fraghard", "model_fragsoft"),
}


def enclosing_template_covariance(shifts, *, rank_rtol=1e-12):
    """Return covariance and containment diagnostics for bin-by-template shifts.

    Fit only in the numerical template span. In whitened coordinates Z, the
    dual maximizes logdet(Z diag(weights) Z.T) over the weight simplex.
    Multiplication by the span rank gives the centered enclosing ellipsoid.
    No physical-space diagonal regularization is added.
    """
    shifts = np.asarray(shifts, dtype=float)
    if shifts.ndim != 2 or shifts.shape[1] == 0 or not np.isfinite(shifts).all():
        raise ValueError("Expected finite bin-by-template shifts with at least one template")
    basis, singular_values, coordinates = np.linalg.svd(shifts, full_matrices=False)
    rank = int(np.sum(singular_values > rank_rtol * singular_values[0])) if singular_values.size else 0
    if rank == 0:
        return np.zeros((shifts.shape[0], shifts.shape[0])), {
            "rank": 0, "weights": [0.0] * shifts.shape[1],
            "mahalanobis_squared": [0.0] * shifts.shape[1],
            "span_relative_residual": 0.0, "rank_rtol": rank_rtol,
            "optimality_gap": 0.0, "containment_inflation": 1.0,
        }
    transform = basis[:, :rank] * singular_values[:rank]
    coordinates = coordinates[:rank]
    residual = np.linalg.norm(shifts - transform @ coordinates) / np.linalg.norm(shifts)
    if residual > 10 * rank_rtol:
        raise ValueError("Template span truncation discarded a non-negligible direction")

    def objective(weights):
        moment = (coordinates * weights) @ coordinates.T
        sign, logdet = np.linalg.slogdet(moment)
        if sign <= 0:
            return np.inf, np.zeros_like(weights)
        leverage = np.sum(coordinates * np.linalg.solve(moment, coordinates), axis=0)
        return -logdet, -leverage

    weights = np.full(shifts.shape[1], 1.0 / shifts.shape[1])
    if rank < shifts.shape[1]:
        fit = minimize(
            objective, weights, jac=True, method="SLSQP",
            bounds=[(0.0, 1.0)] * len(weights),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                         "jac": lambda w: np.ones_like(w)},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not fit.success:
            raise RuntimeError(f"Model ellipsoid optimization failed: {fit.message}")
        weights = fit.x
    moment = rank * (coordinates * weights) @ coordinates.T
    distances = np.sum(coordinates * np.linalg.solve(moment, coordinates), axis=0)
    gap = max(0.0, float(distances.max()) - 1.0)
    if gap > 1e-6:
        raise RuntimeError(f"Model ellipsoid did not converge: containment gap {gap:g}")
    inflation = max(1.0, float(distances.max())) * (1.0 + 8 * np.finfo(float).eps)
    covariance = transform @ (inflation * moment) @ transform.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {
        "rank": rank, "weights": weights.tolist(),
        "mahalanobis_squared": (distances / inflation).tolist(),
        "span_relative_residual": float(residual), "rank_rtol": rank_rtol,
        "optimality_gap": gap, "containment_inflation": inflation,
    }


def two_group_model_covariance(nominal, varied, normalization_weights):
    """Build group matrices from normalized densities, preserving all templates.

    normalization_weights has one row per pT slice: bin widths inside its
    normalization window and zero elsewhere. Already normalized template
    differences must satisfy these constraints; a material mismatch fails.
    Tiny floating-point leakage is removed before fitting the ellipsoids.
    """
    nominal = np.asarray(nominal, dtype=float)
    normalization_weights = np.asarray(normalization_weights, dtype=float)
    if nominal.ndim != 1 or not np.isfinite(nominal).all():
        raise ValueError("Expected a finite nominal density vector")
    if (normalization_weights.ndim != 2
            or normalization_weights.shape[1] != nominal.size
            or not np.isfinite(normalization_weights).all()):
        raise ValueError("Normalization weights do not match the nominal bins")
    matrices, diagnostics = {}, {}
    for group, sources in MODEL_GROUPS.items():
        shifts = np.column_stack([np.asarray(varied[name], dtype=float) - nominal for name in sources])
        if shifts.shape[0] != nominal.size or not np.isfinite(shifts).all():
            raise ValueError(f"Invalid normalized templates for {group}")
        leakage = normalization_weights @ shifts
        if np.max(np.abs(leakage), initial=0.0) > 1e-10:
            raise ValueError(f"Model templates for {group} violate per-pT normalization")
        # Rows have disjoint normalization windows. Correct within each window
        # only at roundoff level, without masking sparse or negative bins.
        for weights, leak in zip(normalization_weights, leakage):
            inside = weights != 0
            integral = weights @ nominal
            if integral == 0:
                raise ValueError("Empty model normalization window")
            shifts[inside] -= np.outer(nominal[inside] / integral, leak)
        matrices[group], diagnostics[group] = enclosing_template_covariance(shifts)
        diagnostics[group]["sources"] = list(sources)
        diagnostics[group]["max_input_normalization_leakage"] = float(np.max(np.abs(leakage), initial=0.0))
    return matrices, diagnostics


# ---------------------------------------------------------------------------
# Part 3: statistical covariance of a normalized prediction
# ---------------------------------------------------------------------------



def normalized_prediction_covariance(counts, covariance, widths, normalization_mask):
    """Propagate raw-count covariance through p_i = n_i / (width_i * N).

    N includes only the normalization-mask bins. The returned covariance covers
    all bins, including bins outside that window. A diagonal sumw2 input cannot
    recover any missing event-level correlations between entries.
    """
    counts = np.asarray(counts, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    widths = np.asarray(widths, dtype=float)
    mask = np.asarray(normalization_mask, dtype=bool)
    if (counts.ndim != 1 or widths.shape != counts.shape or mask.shape != counts.shape
            or covariance.shape != (counts.size, counts.size)):
        raise ValueError("Prediction counts, covariance, widths and mask have incompatible shapes")
    if (not np.all(np.isfinite(counts)) or not np.all(np.isfinite(covariance))
            or not np.all(np.isfinite(widths)) or np.any(widths <= 0)):
        raise ValueError("Prediction inputs must be finite and bin widths positive")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Prediction covariance must be symmetric")
    total = counts[mask].sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Prediction normalization must be positive")
    jacobian = (np.eye(counts.size) - np.outer(counts / total, mask)) / (widths[:, None] * total)
    result = jacobian @ covariance @ jacobian.T
    return 0.5 * (result + result.T)
