"""Z+jet inputs: merged-era coffea pickles -> ``UnfoldInputs``.

The Z+jet skims come as one pickle per era (``pythia_<era>.pkl``) holding
``hist`` objects with a ``systematic`` axis, plus all-era data, HERWIG and
optional jackknife pickles.  This module

* projects and sums the eras (in memory; nothing is written back);
* builds one response matrix per systematic, splitting each JES source into
  a year-correlated leg and one uncorrelated leg per year group;
* derives fakes and misses from the difference between the inclusive reco/gen
  spectra and the matched response, restricted to the in-range pT bins;
* flattens everything onto the analysis binning.

The per-systematic misses anchor on the nominal misses and add only the
changes in the gen total and in the matched projection, so the efficiency
stays consistent with each varied response.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from unfold import systematics as syst
from unfold.histmath import (
    merge_mass_flat, mosaic_no_padding, rebin_hist, reorder_to_expected, reorder_to_expected_2d,
)
from unfold.inputs import UnfoldInputs, compute_fake_fraction

N_JACKKNIFE = 10


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _first_existing(*candidates):
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Could not find any of: " + ", ".join(str(c) for c in candidates if c))


# ---------------------------------------------------------------------------
# response matrices per systematic
# ---------------------------------------------------------------------------
def _era_of(filename, prefix):
    return Path(filename).stem.replace(prefix, "", 1)


def build_systematic_responses(spec, groomed, *, era_split="sqrt"):
    """Return (sys_matrix, nominal_variance, available_systematics).

    ``sys_matrix`` maps every systematic name to a 4-d array
    (ptgen, gen_obs, ptreco, reco_obs) summed over eras; JES sources are split
    into ``<name>Up_corr`` and ``<name>Up_uncorr_<year>`` legs (and Down).
    The key order is the run order of the systematics and is kept identical to
    the historical production (JES ups, other ups, nominal, herwigUp, JES
    downs, other downs, herwigDown).
    """
    key = spec.hist_keys(groomed)["response"]
    axes = ("ptgen", spec.gen_axis, "ptreco", spec.reco_axis)
    by_dataset = {}
    for filename in spec.era_mc_files:
        era = _era_of(filename, spec.era_prefix)
        data = _load_pickle(Path(spec.input_dir) / filename)
        by_dataset[syst.ZJET_ERA_DATASETS[era]] = data[key].project(
            "ptreco", "dataset", "ptgen", spec.gen_axis, spec.reco_axis, "systematic")

    available = set()
    for h in by_dataset.values():
        available.update(str(c) for c in h.axes["systematic"])
    jes_list = [s for s in syst.JES_SYSTEMATICS if s in available]
    non_jes_list = [s for s in syst.ZJET_NON_JES_SYSTEMATICS if s in available]

    def project(dataset, systematic):
        return by_dataset[dataset][..., systematic].project(*axes).values()

    def year_sums(systematic):
        """Per year-group projections; the two 2016 halves are one group."""
        y16 = project("pythia_UL16NanoAODv9", systematic) + project("pythia_UL16NanoAODAPVv9", systematic)
        return y16, project("pythia_UL17NanoAODv9", systematic), project("pythia_UL18NanoAODv9", systematic)

    nominal_by_year = year_sums("nominal")
    nominal_sum = nominal_by_year[0] + nominal_by_year[1] + nominal_by_year[2]

    def era_split_legs(name, source):
        rho = syst.JES_RUN2_CORRELATIONS[source] if source != "JER" else syst.JER_RUN2_CORRELATION
        corr, uncorr = syst.era_split_coefficients(rho, era_split)
        shifts = [s - n for s, n in zip(year_sums(name), nominal_by_year)]
        # summation order kept from the production code (floating point)
        legs = {f"{name}_corr": nominal_sum + (corr * shifts[0] + corr * shifts[1] + corr * shifts[2])}
        for year, shift in zip(("2016", "2017", "2018"), shifts):
            legs[f"{name}_uncorr_{year}"] = nominal_sum + uncorr * shift
        return legs

    def all_eras(systematic):
        return (project("pythia_UL16NanoAODv9", systematic) + project("pythia_UL16NanoAODAPVv9", systematic)
                + project("pythia_UL17NanoAODv9", systematic) + project("pythia_UL18NanoAODv9", systematic))

    def source_of(name):
        base, direction = syst.split_updown(name)
        return base[len("JES_"):] if base.startswith("JES_") else base

    sys_matrix = {}
    for direction in ("Up", "Down"):
        for name in (s for s in jes_list if s.endswith(direction)):
            sys_matrix.update(era_split_legs(name, source_of(name)))
        for name in (s for s in non_jes_list if s.endswith(direction)):
            sys_matrix[name] = all_eras(name)
        if direction == "Up":
            # summation order kept from the production code
            sys_matrix["nominal"] = (project("pythia_UL17NanoAODv9", "nominal")
                                     + project("pythia_UL18NanoAODv9", "nominal")
                                     + project("pythia_UL16NanoAODv9", "nominal")
                                     + project("pythia_UL16NanoAODAPVv9", "nominal"))
            sys_matrix["herwigUp"] = _reweighted_response(spec, key, axes)
    sys_matrix["herwigDown"] = sys_matrix["herwigUp"]

    variances = [by_dataset[d][..., "nominal"].project(*axes).variances()
                 for d in ("pythia_UL16NanoAODv9", "pythia_UL16NanoAODAPVv9",
                           "pythia_UL17NanoAODv9", "pythia_UL18NanoAODv9")]
    nominal_variance = np.sum(variances, axis=0) if all(v is not None for v in variances) else None
    return sys_matrix, nominal_variance


def _reweighted_response(spec, key, axes):
    """Response used for the "herwig" systematic (first existing fallback file).

    Projected without selecting a systematic category, exactly as the
    production did: for a file with several categories this sums over them.
    The arc_r2-style tags point this at the nominal PYTHIA file and exclude
    the herwig systematic from the band, so it only matters for the legacy tags.
    """
    path = _first_existing(*(Path(spec.input_dir) / f for f in spec.reweighted_fallback_files))
    return _load_pickle(path)[key].project(*axes).values()


# ---------------------------------------------------------------------------
# era merging of the hist objects
# ---------------------------------------------------------------------------
def merge_eras(spec, groomed, filenames=None):
    """Sum the per-era reco/gen/response hists (projected to physics axes)."""
    keys = spec.hist_keys(groomed)
    projections = {
        keys["reco"]: ("ptreco", spec.reco_axis, "systematic"),
        keys["response"]: ("ptgen", spec.gen_axis, "ptreco", spec.reco_axis, "systematic"),
        keys["gen"]: ("ptgen", spec.gen_axis, "systematic"),
    }
    merged = {}
    for filename in filenames or spec.era_mc_files:
        payload = _load_pickle(Path(spec.input_dir) / filename)
        for key, axes in projections.items():
            projected = payload[key].project(*axes)
            merged[key] = projected if key not in merged else merged[key] + projected
    return merged


def merge_eras_jk(spec, groomed):
    key = spec.hist_keys(groomed)["response"]
    axes = ("jk", "ptgen", spec.gen_axis, "ptreco", spec.reco_axis, "systematic")
    merged = None
    for filename in spec.era_jk_files:
        projected = _load_pickle(Path(spec.input_dir) / filename)[key].project(*axes)
        merged = projected if merged is None else merged + projected
    return merged


# ---------------------------------------------------------------------------
# flattening helpers
# ---------------------------------------------------------------------------
def _sum_inrange_pt(h, pt_axis):
    """Sum over the in-range pT bins only (exclude pT flow).

    A flow-inclusive projection would fold a cross-boundary event (gen < 185 &
    reco >= 185, or the reverse) back into "matched", cancelling it out of
    fakes/misses, while the matrix itself is built with flow=False and drops
    it.  Summing in-range bins puts those events where they belong.
    """
    return h[{pt_axis: slice(0, len(h.axes[pt_axis]), sum)}]


def _flat_2d(values, fine_edges, pt_edges, edges_by_pt):
    reordered, _ = reorder_to_expected_2d(values, fine_edges, pt_edges)
    return merge_mass_flat(reordered, fine_edges, edges_by_pt)


def _mosaic(values_4d, b):
    reordered, _ = reorder_to_expected(values_4d)
    mosaic, _ = mosaic_no_padding(reordered, b.reco_edges, b.gen_edges, b.reco_edges_by_pt, b.gen_edges_by_pt)
    return reordered, mosaic


def flat_gen_total(gen_hist, systematic, gen_axis, b):
    proj = gen_hist.project("ptgen", gen_axis, "systematic")[:, :, systematic].project("ptgen", gen_axis)
    return _flat_2d(proj.values(), b.gen_edges, b.pt_edges, b.gen_edges_by_pt)


def flat_gen_var(gen_hist, systematic, gen_axis, b):
    proj = gen_hist.project("ptgen", gen_axis, "systematic")[:, :, systematic].project("ptgen", gen_axis)
    variances = proj.variances()
    if variances is None:
        return None
    return _flat_2d(variances, b.gen_edges, b.pt_edges, b.gen_edges_by_pt)


def truth_spectrum(gen2d, gen_axis, b):
    if "systematic" in gen2d.axes.name:
        proj = gen2d.project("ptgen", gen_axis, "systematic")[:, :, "nominal"]
    else:
        proj = gen2d.project("ptgen", gen_axis)
    return _flat_2d(proj.values(), b.gen_edges, b.pt_edges, b.gen_edges_by_pt)


def herwig_pieces(herwig4d, herwig4d_gen, fakes_herwig, misses_herwig, spec, b):
    """Flattened HERWIG reco spectrum, fakes, misses and gen x gen response."""
    nominal = herwig4d[{"systematic": "nominal"}]
    h2d = reorder_to_expected_2d(nominal.project("ptreco", spec.reco_axis).values(flow=False),
                                 b.reco_edges, b.pt_edges)[0]
    fakes = _flat_2d(fakes_herwig.project("ptreco", spec.reco_axis).values(), b.reco_edges, b.pt_edges, b.reco_edges_by_pt)
    misses = _flat_2d(misses_herwig.project("ptgen", spec.gen_axis).values(), b.gen_edges, b.pt_edges, b.gen_edges_by_pt)
    gen_nominal = herwig4d_gen[{"systematic": "nominal"}]
    reordered, _ = reorder_to_expected(gen_nominal.project("ptreco", spec.reco_axis, "ptgen", spec.gen_axis).values(flow=False))
    mosaic_gen, _ = mosaic_no_padding(reordered, b.gen_edges, b.gen_edges, b.gen_edges_by_pt, b.gen_edges_by_pt)
    return h2d, fakes, misses, mosaic_gen


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------
def load_zjet_inputs(spec, groomed, binning, *, do_syst=True, jackknife=True, era_split="sqrt"):
    """Read the Z+jet pickles named by ``spec`` and build ``UnfoldInputs``."""
    b = binning
    keys = spec.hist_keys(groomed)
    reco_axis, gen_axis = spec.reco_axis, spec.gen_axis
    in_dir = Path(spec.input_dir)

    print("------------- Building response matrices from the per-era pickles -----------------")
    sys_matrix, nominal_variance = build_systematic_responses(spec, groomed, era_split=era_split)
    systematics = list(sys_matrix) if do_syst else ["nominal"]

    print("------------- Adding inputs to unfolder -----------------")
    pythia = merge_eras(spec, groomed)
    jk_paths = [in_dir / f for f in spec.era_jk_files] + [in_dir / spec.jk_data_file]
    has_jackknife = bool(jackknife)
    if has_jackknife and not all(p.exists() for p in jk_paths):
        print("WARNING: jackknife pkls missing -> statistical uncertainty falls back to the "
              "TUnfold-propagated input covariance (no jackknife, no matrix-stat term). Missing: "
              + ", ".join(str(p) for p in jk_paths if not p.exists()))
        has_jackknife = False

    herwig = _load_pickle(_first_existing(in_dir / spec.herwig_file, in_dir / "herwig_all.pkl"))
    data = _load_pickle(_first_existing(in_dir / spec.data_file, in_dir / "data_all.pkl"))

    pythia4d, pythia2d, pythia_gen2d = pythia[keys["response"]], pythia[keys["reco"]], pythia[keys["gen"]]
    herwig4d, herwig2d, herwig_gen2d = herwig[keys["response"]], herwig[keys["reco"]], herwig[keys["gen"]]
    data2d = data[keys["reco"]]

    pythia4d_gen = rebin_hist(pythia4d.copy(), reco_axis, list(b.gen_edges))
    herwig4d_gen = rebin_hist(herwig4d.copy(), reco_axis, list(b.gen_edges))

    # fakes/misses = inclusive reco/gen minus the matched matrix (in-range pT only)
    fakes = (pythia2d.project("ptreco", reco_axis, "systematic")[:, :, "nominal"]
             + (-1) * _sum_inrange_pt(pythia4d, "ptgen").project("ptreco", reco_axis, "systematic")[:, :, "nominal"])
    fakes_herwig = (herwig2d.project("ptreco", reco_axis, "systematic")
                    + (-1) * _sum_inrange_pt(herwig4d, "ptgen").project("ptreco", reco_axis, "systematic"))
    misses = (pythia_gen2d.project("ptgen", gen_axis, "systematic")[:, :, "nominal"]
              + (-1) * _sum_inrange_pt(pythia4d, "ptreco").project("ptgen", gen_axis, "systematic")[:, :, "nominal"])
    misses_herwig = (herwig_gen2d.project("ptgen", gen_axis, "systematic")
                     + (-1) * _sum_inrange_pt(herwig4d, "ptreco").project("ptgen", gen_axis, "systematic"))

    inp = UnfoldInputs(binning=b, systematics=systematics, mosaic_dict={}, has_jackknife=has_jackknife,
                       has_herwig=True, has_validation_inputs=True,
                       data_2d=data2d, pythia_2d=pythia2d, pythia_4d=pythia4d, herwig_2d=herwig2d,
                       herwig_4d=herwig4d, herwig_4d_gen=herwig4d_gen, fakes_herwig=fakes_herwig,
                       misses_herwig=misses_herwig, first_reported_pt_bin=0)
    inp.y_true_herwig = truth_spectrum(herwig_gen2d, gen_axis, b)

    # nominal data and MC pieces
    reco_proj = data2d.project("ptreco", reco_axis)
    inp.h2d, _ = reorder_to_expected_2d(reco_proj.values(), b.reco_edges, b.pt_edges)
    reco_variances = reco_proj.variances()
    inp.measured_variances = (None if reco_variances is None else
                              _flat_2d(np.clip(reco_variances, 0.0, None), b.reco_edges, b.pt_edges, b.reco_edges_by_pt))
    inp.fakes_2d = _flat_2d(fakes.project("ptreco", reco_axis).values(), b.reco_edges, b.pt_edges, b.reco_edges_by_pt)
    inp.misses_2d = _flat_2d(misses.project("ptgen", gen_axis).values(), b.gen_edges, b.pt_edges, b.gen_edges_by_pt)
    gen_nominal = pythia4d_gen[{"systematic": "nominal"}]
    inp.M_np_2d_gen, _ = reorder_to_expected(gen_nominal.project("ptreco", reco_axis, "ptgen", gen_axis).values(flow=False))
    inp.mosaic_gen, _ = mosaic_no_padding(inp.M_np_2d_gen, b.gen_edges, b.gen_edges, b.gen_edges_by_pt, b.gen_edges_by_pt)

    _load_background(inp, spec, groomed, b)

    if has_jackknife:
        print("Processing jk inputs...")
        data2d_jk = _load_pickle(_first_existing(in_dir / spec.jk_data_file, in_dir / "jk_data_all.pkl"))[keys["reco"]]
        pythia4d_jk = merge_eras_jk(spec, groomed)
        inp.mosaic_2d_jk_list = [
            _flat_2d(data2d_jk.project("jk", "ptreco", reco_axis)[i, ...].values(), b.reco_edges, b.pt_edges, b.reco_edges_by_pt)
            for i in range(N_JACKKNIFE)
        ]
        nominal_jk = pythia4d_jk[{"systematic": "nominal"}]
        inp.mosaic_jk_list = [
            _mosaic(nominal_jk[{"jk": i}].project("ptgen", gen_axis, "ptreco", reco_axis).values(flow=False), b)[1]
            for i in range(N_JACKKNIFE)
        ]

    # response mosaics per systematic
    for name in systematics:
        inp.M_np_2d_dict[name], inp.mosaic_dict[name] = _mosaic(sys_matrix[name], b)
    if "herwigUp" in systematics or "herwigDown" in systematics:
        inp.h2d_herwig, inp.fakes_2d_herwig, inp.misses_2d_herwig, inp.mosaic_gen_herwig = herwig_pieces(
            herwig4d, herwig4d_gen, fakes_herwig, misses_herwig, spec, b)

    # per-systematic misses: nominal + (gen total change) - (matched change)
    gen_syst_labels = set(pythia_gen2d.axes["systematic"])
    matched_nom = inp.mosaic_dict["nominal"].sum(axis=0)
    gen_total_nom = flat_gen_total(pythia_gen2d, "nominal", gen_axis, b)
    for name in systematics:
        if name in {"nominal", "herwigUp", "herwigDown"}:
            continue
        gen_delta = 0.0
        if name in gen_syst_labels:
            gen_delta = flat_gen_total(pythia_gen2d, name, gen_axis, b) - gen_total_nom
        matched_delta = inp.mosaic_dict[name].sum(axis=0) - matched_nom
        inp.misses_2d_dict[name] = np.clip(inp.misses_2d + gen_delta - matched_delta, 0.0, None)

    # generator predictions with MC-stat variances (same flattening for both)
    inp.pythia_gen_val_flat = gen_total_nom
    inp.pythia_gen_var_flat = flat_gen_var(pythia_gen2d, "nominal", gen_axis, b)
    inp.herwig_gen_val_flat = flat_gen_total(herwig_gen2d, "nominal", gen_axis, b)
    inp.herwig_gen_var_flat = flat_gen_var(herwig_gen2d, "nominal", gen_axis, b)

    # MC-stat variances of the nominal response, fakes and misses (fakes and
    # misses are disjoint subsets of the totals, so the variance is the difference)
    if nominal_variance is not None:
        var_2d, _ = reorder_to_expected(nominal_variance)
        mosaic_var, _ = mosaic_no_padding(var_2d, b.reco_edges, b.gen_edges, b.reco_edges_by_pt, b.gen_edges_by_pt)
        inp.mosaic_var_dict["nominal"] = np.clip(mosaic_var, 0.0, None)

    def nominal_var(hist_obj, pt_axis, obs_axis):
        return hist_obj.project(pt_axis, obs_axis, "systematic")[:, :, "nominal"].variances()

    reco_total_var = nominal_var(pythia2d, "ptreco", reco_axis)
    reco_matched_var = nominal_var(pythia4d, "ptreco", reco_axis)
    gen_total_var = nominal_var(pythia_gen2d, "ptgen", gen_axis)
    gen_matched_var = nominal_var(pythia4d, "ptgen", gen_axis)
    if reco_total_var is not None and reco_matched_var is not None:
        inp.fakes_2d_var = _flat_2d(np.clip(reco_total_var - reco_matched_var, 0.0, None),
                                    b.reco_edges, b.pt_edges, b.reco_edges_by_pt)
    if gen_total_var is not None and gen_matched_var is not None:
        inp.misses_var_dict["nominal"] = _flat_2d(np.clip(gen_total_var - gen_matched_var, 0.0, None),
                                                  b.gen_edges, b.pt_edges, b.gen_edges_by_pt)

    inp.mosaic_2d = merge_mass_flat(inp.h2d, b.reco_edges, b.reco_edges_by_pt)
    inp.fake_fraction_2d = compute_fake_fraction(inp.fakes_2d, inp.mosaic_dict["nominal"].sum(axis=1))
    if inp.h2d_herwig is not None:
        inp.mosaic_herwig_2d = merge_mass_flat(inp.h2d_herwig, b.reco_edges, b.reco_edges_by_pt)
        inp.fake_fraction_2d_herwig = compute_fake_fraction(inp.fakes_2d_herwig, inp.mosaic_herwig_2d)
    print("Loaded data and prepared response matrices.")
    return inp


def _load_background(inp, spec, groomed, b):
    """Non-DY background (ttbar, single top, diboson) for bin-by-bin subtraction."""
    if not spec.bkg_file:
        return
    bkg_path = Path(spec.input_dir) / spec.bkg_file
    if not bkg_path.exists():
        print("WARNING: background pkl not found -> the non-DY background is NOT subtracted "
              f"from the unfolding input (legacy behavior). Missing: {bkg_path}")
        return
    reco_key = spec.hist_keys(groomed)["reco"]
    payload = _load_pickle(bkg_path)
    if reco_key not in payload:
        raise RuntimeError(f"Background pkl {bkg_path} has no '{reco_key}' histogram; it must be "
                           "produced with the same processor/histogram set as the data.")
    bkg_hist = payload[reco_key]
    for axis_name, expected in (("ptreco", b.pt_edges), (spec.reco_axis, b.reco_edges)):
        found = np.asarray(bkg_hist.axes[axis_name].edges, dtype=float)
        if not np.allclose(found, np.asarray(expected, dtype=float)):
            raise RuntimeError(f"Background pkl {bkg_path} has '{axis_name}' edges {list(found)} but the "
                               f"unfolding input uses {list(np.asarray(expected, dtype=float))}.")
    if "systematic" in bkg_hist.axes.name and len(bkg_hist.axes["systematic"]) > 1:
        bkg_hist = bkg_hist[{"systematic": "nominal"}]
    proj = bkg_hist.project("ptreco", spec.reco_axis)
    inp.bkg_2d = _flat_2d(proj.values(), b.reco_edges, b.pt_edges, b.reco_edges_by_pt)
    variances = proj.variances()
    if variances is not None:
        inp.bkg_var_2d = _flat_2d(np.clip(variances, 0.0, None), b.reco_edges, b.pt_edges, b.reco_edges_by_pt)
    print(f"Loaded non-DY background from {bkg_path.name}: {inp.bkg_2d.sum():.1f} events will be subtracted from the input.")
