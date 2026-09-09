"""The input contract of the unfolding engine, and the builder for prepared inputs.

``UnfoldInputs`` is what every loader produces and what ``engine.Unfolder``
consumes.  All spectra are "flat": pT slice after pT slice, each slice on
its own analysis edges (see ``histmath``).  The response ``mosaic`` is
(n_reco, n_gen) in that layout.

Two loaders exist:

* ``zjet_inputs.load_zjet_inputs`` reads the merged-era Z+jet pickles;
* ``prepared_inputs`` (this module) takes already adapted ``hist`` objects
  with a ``systematic`` axis, which is how the dijet/trijet and pair-split
  channels arrive (``channel_inputs``, ``pairsplit.inputs``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from unfold.binning import Binning
from unfold.histmath import merge_mass_flat, mosaic_no_padding, reorder_to_expected, reorder_to_expected_2d


@dataclass
class UnfoldInputs:
    binning: Binning
    systematics: list                      # run order; "nominal" included

    # response matrices per systematic, flat (n_reco, n_gen); sumw2 alongside
    mosaic_dict: dict
    mosaic_var_dict: dict = field(default_factory=dict)
    M_np_2d_dict: dict = field(default_factory=dict)     # 4-d (reco_obs, reco_pt, gen_obs, gen_pt) per systematic

    # efficiency and fake inputs (flat)
    misses_2d: np.ndarray = None           # nominal misses (gen events failing reco)
    misses_2d_dict: dict = field(default_factory=dict)   # per systematic
    misses_var_dict: dict = field(default_factory=dict)
    fakes_2d: np.ndarray = None            # nominal fakes (reco events with no matched gen)
    fakes_2d_var: np.ndarray = None
    fake_fraction_2d: np.ndarray = None    # fakes / (matched + fakes), nominal
    fakes_2d_dict: dict | None = None      # prepared inputs carry these per systematic
    fakes_2d_var_dict: dict | None = None
    fake_fraction_2d_dict: dict | None = None
    fake_survival_dict: dict | None = None

    # measured data (flat reco)
    mosaic_2d: np.ndarray = None
    measured_variances: np.ndarray | None = None
    measured_covariance: np.ndarray | None = None
    corrected_measured_covariance_dict: dict | None = None

    # non-DY background to subtract before the fake correction (Z+jet)
    bkg_2d: np.ndarray | None = None
    bkg_var_2d: np.ndarray | None = None

    # delete-one-tenth jackknife replicas (Z+jet); None when absent
    mosaic_2d_jk_list: list | None = None
    mosaic_jk_list: list | None = None
    has_jackknife: bool = False

    # HERWIG (overlay, closure/bias test, and the "herwig" systematic)
    has_herwig: bool = False
    mosaic_herwig_2d: np.ndarray | None = None      # matched HERWIG reco spectrum
    fakes_2d_herwig: np.ndarray | None = None
    misses_2d_herwig: np.ndarray | None = None
    fake_fraction_2d_herwig: np.ndarray | None = None
    y_true_herwig: np.ndarray | None = None         # full HERWIG gen spectrum
    herwig_gen_val_flat: np.ndarray | None = None
    herwig_gen_var_flat: np.ndarray | None = None
    mosaic_gen_herwig: np.ndarray | None = None     # HERWIG response on gen x gen binning
    h2d_herwig: np.ndarray | None = None

    # PYTHIA gen prediction (values and sumw2, flat gen)
    pythia_gen_val_flat: np.ndarray | None = None
    pythia_gen_var_flat: np.ndarray | None = None
    mosaic_gen: np.ndarray | None = None            # nominal response on gen x gen binning
    M_np_2d_gen: np.ndarray | None = None
    h2d: np.ndarray | None = None

    # prepared-input extras (MC marginals per systematic)
    reco_mc_flat_dict: dict | None = None
    reco_mc_var_dict: dict | None = None
    gen_mc_flat_dict: dict | None = None
    gen_mc_var_dict: dict | None = None
    matched_reco_dict: dict | None = None
    matched_gen_dict: dict | None = None

    # raw hist objects kept for the input data/MC plots (Z+jet)
    data_2d: object = None
    pythia_2d: object = None
    herwig_2d: object = None
    pythia_4d: object = None
    herwig_4d: object = None
    herwig_4d_gen: object = None
    fakes_herwig: object = None
    misses_herwig: object = None

    # flags
    has_validation_inputs: bool = False
    uses_prepared_inputs: bool = False
    response_matrix_stat_available: bool = True
    first_reported_pt_bin: int = 0
    stat_uncertainty_method: str = "jackknife"


# ---------------------------------------------------------------------------
# small helpers shared with the engine
# ---------------------------------------------------------------------------
def compute_fake_fraction(fakes_flat, matched_flat):
    total_reco = matched_flat + fakes_flat
    with np.errstate(divide="ignore", invalid="ignore"):
        fake_fraction = np.divide(
            fakes_flat, total_reco, out=np.zeros_like(fakes_flat, dtype=float), where=total_reco > 0,
        )
    return np.clip(fake_fraction, 0.0, 1.0)


def scale_covariance_by_fake_survival(covariance, fake_survival):
    """Apply a diagonal fake-survival transform C -> D C D."""
    covariance = np.asarray(covariance, dtype=float)
    fake_survival = np.asarray(fake_survival, dtype=float)
    if covariance.shape != (fake_survival.size, fake_survival.size):
        raise ValueError("Measured covariance shape does not match the flattened reco spectrum")
    return covariance * np.outer(fake_survival, fake_survival)


def nonnegative_variance_difference(inclusive_variance, matched_variance, *, systematic, role):
    """Variance of a disjoint inclusive-minus-matched subset.

    A tiny negative remainder can arise from floating point summation after
    rebinning; a material one signals an incompatible input and must not be
    hidden by clipping it to zero.
    """
    inclusive_variance = np.asarray(inclusive_variance, dtype=float)
    matched_variance = np.asarray(matched_variance, dtype=float)
    difference = inclusive_variance - matched_variance
    scale = np.maximum(1.0, np.maximum(np.abs(inclusive_variance), np.abs(matched_variance)))
    material_negative = difference < -1e-10 * scale
    if np.any(material_negative):
        worst = float(np.min(difference[material_negative]))
        raise ValueError(
            f"Prepared {role} variance is materially negative for systematic "
            f"{systematic!r} after subtracting the matched response (minimum {worst:.6g})."
        )
    return np.where(difference < 0.0, 0.0, difference)


# ---------------------------------------------------------------------------
# prepared inputs: hist objects with a "systematic" axis -> UnfoldInputs
# ---------------------------------------------------------------------------
def _select_systematic(h_obj, systematic, role):
    """Select one required systematic category; no fallback to nominal."""
    if "systematic" not in h_obj.axes.name:
        raise ValueError(
            f"Prepared {role} histogram has no systematic axis; required category {systematic!r} is unavailable"
        )
    if systematic not in list(h_obj.axes["systematic"]):
        raise ValueError(f"Prepared {role} histogram is missing required systematic category {systematic!r}")
    return h_obj[{"systematic": systematic}]


def _select_nominal(h_obj):
    if "systematic" in h_obj.axes.name:
        return h_obj[{"systematic": "nominal"}]
    return h_obj


def _flatten_2d(h_obj, fine_edges, pt_edges, edges_by_pt, axes):
    """(values, variances) of a 2-d hist flattened onto the per-pT layout."""
    projected = h_obj.project(*axes)
    values, _ = reorder_to_expected_2d(projected.values(flow=False), fine_edges, pt_edges)
    flat_values = merge_mass_flat(values, fine_edges, edges_by_pt)
    variances = projected.variances(flow=False)
    flat_variances = None
    if variances is not None:
        reordered, _ = reorder_to_expected_2d(variances, fine_edges, pt_edges)
        flat_variances = merge_mass_flat(reordered, fine_edges, edges_by_pt)
    return flat_values, flat_variances


def response_mosaic(response_hist, systematic, binning, gen_axis, reco_axis):
    """Flatten one response category: (reordered 4-d, mosaic, var 4-d, var mosaic)."""
    selected = _select_systematic(response_hist, systematic, "response")
    projected = selected.project("ptgen", gen_axis, "ptreco", reco_axis)
    reordered, _ = reorder_to_expected(projected.values(flow=False))
    mosaic, _ = mosaic_no_padding(reordered, binning.reco_edges, binning.gen_edges,
                                  binning.reco_edges_by_pt, binning.gen_edges_by_pt)
    variances = projected.variances(flow=False)
    if variances is None:
        return reordered, mosaic, None, None
    var_reordered, _ = reorder_to_expected(np.clip(variances, 0.0, None))
    var_mosaic, _ = mosaic_no_padding(var_reordered, binning.reco_edges, binning.gen_edges,
                                      binning.reco_edges_by_pt, binning.gen_edges_by_pt)
    return reordered, mosaic, var_reordered, var_mosaic


def _resolve_covariance(covariance, groomed, n_reco):
    if isinstance(covariance, Mapping):
        mode = "groomed" if groomed else "ungroomed"
        covariance = covariance.get(mode, covariance.get(groomed))
    if covariance is None:
        return None
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (n_reco, n_reco):
        raise ValueError(f"Prepared measured covariance has shape {covariance.shape}; expected {(n_reco, n_reco)}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Prepared measured covariance contains non-finite entries")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Prepared measured covariance must be symmetric")
    return covariance


def prepared_inputs(spec, groomed, binning, *, mc_inputs, data_inputs, systematics,
                    herwig_inputs=None, measured_covariance=None, first_reported_pt_bin=1):
    """Build ``UnfoldInputs`` from adapted histograms (dijet/trijet, pair-split).

    Every requested systematic must exist in the response and in both MC
    marginals: a varied migration matrix is never combined with nominal fakes
    or misses.  The producer's weighted sumw2 is carried through so TUnfold
    gets MC-stat bin errors, and the analytic stat treatment is TUnfold's input
    covariance plus its response-matrix covariance (never a fabricated jackknife).
    """
    keys = spec.hist_keys(groomed)
    gen_axis, reco_axis = spec.gen_axis, spec.reco_axis
    response_hist = mc_inputs[keys["response"]]
    mc_reco_hist = mc_inputs[keys["reco"]]
    mc_gen_hist = mc_inputs[keys["gen"]]
    data_reco_hist = data_inputs[keys["reco"]]

    systematics = list(systematics)
    if not systematics or "nominal" not in systematics:
        raise ValueError("Prepared systematics must include nominal")
    if len(set(systematics)) != len(systematics):
        raise ValueError("Prepared systematic list contains duplicate categories")
    missing_by_role = {}
    for role, h_obj in {"response": response_hist, "reco marginal": mc_reco_hist, "gen marginal": mc_gen_hist}.items():
        available = list(h_obj.axes["systematic"]) if "systematic" in h_obj.axes.name else []
        missing = [name for name in systematics if name not in available]
        if missing:
            missing_by_role[role] = missing
    if missing_by_role:
        raise ValueError("Prepared inputs are missing required systematic categories: "
                         + "; ".join(f"{role}={missing}" for role, missing in missing_by_role.items()))

    n_pt = binning.n_pt
    if not 0 <= int(first_reported_pt_bin) < n_pt:
        raise ValueError(f"first_reported_pt_bin={first_reported_pt_bin} is outside the {n_pt} prepared pT slices")

    inp = UnfoldInputs(binning=binning, systematics=systematics, mosaic_dict={},
                       uses_prepared_inputs=True, first_reported_pt_bin=int(first_reported_pt_bin),
                       stat_uncertainty_method="TUnfold GetEmatrixInput + GetEmatrixSysUncorr")
    inp.reco_mc_flat_dict, inp.reco_mc_var_dict = {}, {}
    inp.gen_mc_flat_dict, inp.gen_mc_var_dict = {}, {}
    inp.matched_reco_dict, inp.matched_gen_dict = {}, {}
    inp.fakes_2d_dict, inp.fakes_2d_var_dict = {}, {}
    inp.fake_fraction_2d_dict, inp.fake_survival_dict = {}, {}

    reco_axes = ("ptreco", reco_axis)
    gen_axes = ("ptgen", gen_axis)
    for systematic in systematics:
        reordered, mosaic, _, mosaic_variance = response_mosaic(response_hist, systematic, binning, gen_axis, reco_axis)
        if mosaic_variance is None:
            raise ValueError(f"Prepared response {systematic!r} has no sumw2 variances")
        reco_flat, reco_variance = _flatten_2d(
            _select_systematic(mc_reco_hist, systematic, "reco marginal"),
            binning.reco_edges, binning.pt_edges, binning.reco_edges_by_pt, reco_axes)
        gen_flat, gen_variance = _flatten_2d(
            _select_systematic(mc_gen_hist, systematic, "gen marginal"),
            binning.gen_edges, binning.pt_edges, binning.gen_edges_by_pt, gen_axes)
        if reco_variance is None or gen_variance is None:
            raise ValueError(f"Prepared {systematic!r} MC marginals must retain sumw2 variances")
        inp.M_np_2d_dict[systematic] = reordered
        inp.mosaic_dict[systematic] = mosaic
        inp.mosaic_var_dict[systematic] = np.clip(mosaic_variance, 0.0, None)
        inp.reco_mc_flat_dict[systematic] = reco_flat
        inp.reco_mc_var_dict[systematic] = np.clip(reco_variance, 0.0, None)
        inp.gen_mc_flat_dict[systematic] = gen_flat
        inp.gen_mc_var_dict[systematic] = np.clip(gen_variance, 0.0, None)

        matched_reco = mosaic.sum(axis=1)
        matched_gen = mosaic.sum(axis=0)
        inp.matched_reco_dict[systematic] = matched_reco
        inp.matched_gen_dict[systematic] = matched_gen
        fakes = reco_flat - matched_reco
        inp.fakes_2d_dict[systematic] = fakes
        inp.fakes_2d_var_dict[systematic] = nonnegative_variance_difference(
            reco_variance, inp.mosaic_var_dict[systematic].sum(axis=1), systematic=systematic, role="reco fake")
        inp.misses_2d_dict[systematic] = gen_flat - matched_gen
        inp.misses_var_dict[systematic] = nonnegative_variance_difference(
            gen_variance, inp.mosaic_var_dict[systematic].sum(axis=0), systematic=systematic, role="gen miss")
        fake_fraction = compute_fake_fraction(fakes, matched_reco)
        inp.fake_fraction_2d_dict[systematic] = fake_fraction
        inp.fake_survival_dict[systematic] = 1.0 - fake_fraction

    inp.mosaic_2d, inp.measured_variances = _flatten_2d(
        _select_systematic(data_reco_hist, "nominal", "data reco"),
        binning.reco_edges, binning.pt_edges, binning.reco_edges_by_pt, reco_axes)
    inp.measured_covariance = _resolve_covariance(measured_covariance, groomed, len(inp.mosaic_2d))
    inp.corrected_measured_covariance_dict = {}
    if inp.measured_covariance is not None:
        inp.measured_variances = np.diag(inp.measured_covariance).copy()
        for systematic in systematics:
            inp.corrected_measured_covariance_dict[systematic] = scale_covariance_by_fake_survival(
                inp.measured_covariance, inp.fake_survival_dict[systematic])

    inp.fakes_2d = inp.fakes_2d_dict["nominal"]
    inp.fakes_2d_var = inp.fakes_2d_var_dict["nominal"]
    inp.misses_2d = inp.misses_2d_dict["nominal"]
    inp.fake_fraction_2d = inp.fake_fraction_2d_dict["nominal"]
    inp.data_2d = data_reco_hist
    inp.pythia_2d = mc_reco_hist
    inp.pythia_4d = response_hist

    if herwig_inputs is not None:
        _add_prepared_herwig(inp, spec, groomed, binning, herwig_inputs)
    return inp


def _add_prepared_herwig(inp, spec, groomed, binning, herwig_inputs):
    """HERWIG for the prepared path: overlay, bias test, and the herwig systematic.

    The HERWIG response is registered as the herwigUp/herwigDown systematics so
    the alternate-generator difference enters the band through the ordinary
    systematics loop (symmetric: both directions share the HERWIG mosaic).
    """
    keys = spec.hist_keys(groomed)
    reordered, mosaic, _, mosaic_variance = response_mosaic(
        herwig_inputs[keys["response"]], "nominal", binning, spec.gen_axis, spec.reco_axis)
    reco_flat, _ = _flatten_2d(_select_nominal(herwig_inputs[keys["reco"]]),
                               binning.reco_edges, binning.pt_edges, binning.reco_edges_by_pt,
                               ("ptreco", spec.reco_axis))
    gen_flat, gen_var = _flatten_2d(_select_nominal(herwig_inputs[keys["gen"]]),
                                    binning.gen_edges, binning.pt_edges, binning.gen_edges_by_pt,
                                    ("ptgen", spec.gen_axis))
    matched_reco = mosaic.sum(axis=1)
    matched_gen = mosaic.sum(axis=0)
    inp.fakes_2d_herwig = reco_flat - matched_reco
    inp.misses_2d_herwig = gen_flat - matched_gen
    inp.fake_fraction_2d_herwig = compute_fake_fraction(inp.fakes_2d_herwig, matched_reco)
    inp.mosaic_herwig_2d = matched_reco
    inp.y_true_herwig = gen_flat
    inp.herwig_gen_val_flat = gen_flat
    inp.herwig_gen_var_flat = gen_var
    inp.has_herwig = True
    for name in ("herwigUp", "herwigDown"):
        inp.mosaic_dict[name] = mosaic
        inp.M_np_2d_dict[name] = reordered
        if mosaic_variance is not None:
            inp.mosaic_var_dict[name] = np.clip(mosaic_variance, 0.0, None)
        if gen_var is not None and mosaic_variance is not None:
            inp.misses_var_dict[name] = nonnegative_variance_difference(
                gen_var, mosaic_variance.sum(axis=0), systematic="herwig", role="gen miss")
        if name not in inp.systematics:
            inp.systematics.append(name)
