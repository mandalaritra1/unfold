#!/usr/bin/env python
"""Propagate the 27 JES uncertainty sources through the hadronic groomed-rho
Run 2 unfolding with the Z+jet era-correlation prescription.

A JES source is neither fully correlated nor fully uncorrelated across the
data-taking years: the JEC group assigns each source a correlation fraction
``rho`` (0, 0.5 or 1) between eras.  The Z+jet machinery
(``unfold/src/unfold/tools/unfolder_core.py``, ``correlation_dic``) turns that
single number into FOUR independent nuisances per source and direction, each of
which is a complete Run 2 object that gets unfolded on its own:

    corr      = nominal_run2 + f_corr   * (sigma_2016 + sigma_2017 + sigma_2018)
    uncorr_e  = nominal_run2 + f_uncorr * sigma_e                (e = the 3 groups)

with ``sigma_e = varied_e - nominal_e`` evaluated at histogram level on the
era-summed arrays.  The four legs then enter the quadrature together, which
reproduces the intended era-to-era correlation structure of the source.

THREE era groups, not four.  2016APV and 2016 are summed into a single "2016"
group before the split, i.e. those two eras are taken fully correlated for
every source -- the JEC uncertainties are not split at the HIPM boundary.

Split convention
----------------
The zjet code uses ``f_corr = rho`` and ``f_uncorr = 1 - rho``.  That is exact
at rho = 0 and rho = 1 but UNDER-COVERS in between: the total variance it
builds for one source is

    f_corr^2 * (sum_e sigma_e)^2 + f_uncorr^2 * sum_e sigma_e^2

and at rho = 0.5 with three equal-size eras that is 0.25*(3 sigma)^2 +
0.25*3 sigma^2 = 3.0 sigma^2, which is exactly what rho = 0 gives
(f_corr = 0, f_uncorr = 1 -> 3 sigma^2).  A half-correlated source therefore
costs nothing over a fully uncorrelated one.  The textbook split
``f_corr = sqrt(rho)``, ``f_uncorr = sqrt(1 - rho)`` reproduces the intended
covariance ``cov(e, e') = rho * sigma_e * sigma_e'`` for e != e' and
``sigma_e^2`` on the diagonal, and is the DEFAULT here; ``--split-mode
linear`` reproduces the zjet behaviour for comparison.  The two coincide at
rho = 0 and rho = 1, so only the eleven rho = 0.5 sources differ.

Null legs
---------
``f = 0`` makes a leg identical to nominal by construction.  Those legs are
detected, verified to be bit-identical to nominal, skipped (no unfold) and
counted as an exactly zero shift.  At rho = 1 the three uncorr legs are null;
at rho = 0 the corr leg is null.

Efficiency
----------
Every era pickle is opened ONCE per channel.  The dataset-summed,
flow-included arrays for the nominal and all 56 JES/JER categories are
regrouped onto the adopted binning at that moment and cached in a small NPZ
(a few MB); all legs are then built in numpy from the cache.  No pickle is
ever reopened per source.

    source ~/Projects/unfold/scripts/setup_root.sh
    ~/Projects/unfold/.venv/bin/python \
        scripts/studies/unfold/rho_jes_era_correlated.py --build-cache
    ~/Projects/unfold/.venv/bin/python \
        scripts/studies/unfold/rho_jes_era_correlated.py
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.studies.unfold.rho_unfold_inputs_from_hists import (  # noqa: E402
    ADOPTED,
    HIST_KEYS,
    edge_groups,
    regroup,
)
from scripts.studies.unfold.rho_unfold_systematics import (  # noqa: E402
    FLOOR,
    SPECS,
    normalized_shape,
    relative,
    summarize,
    unfold_variation,
    use_binning,
)

REPO = Path(__file__).resolve().parents[3]
PKL_BASE = Path("/Users/aritra/cernbox (2)/hadronic_minimal_rho_pairsplit")
OUT_DIR = REPO / "outputs" / "pairsplit_unfold"
CACHE_DIR = OUT_DIR / "npz_jes"

#### The four data-taking eras collapse into THREE correlation groups: the
#### JEC sources are not split at the HIPM boundary, so 2016APV and 2016 are
#### summed before anything else happens.
ERA_GROUPS = {
    "2016": ("2016APV", "2016"),
    "2017": ("2017",),
    "2018": ("2018",),
}
GROUPS = ("2016", "2017", "2018")

#### JEC era-correlation fractions, verbatim from the zjet ``correlation_dic``
#### (unfold/src/unfold/tools/unfolder_core.py).
CORRELATION = {
    "AbsoluteMPFBias": 1.0,
    "AbsoluteScale": 1.0,
    "AbsoluteStat": 0.0,
    "FlavorQCD": 1.0,
    "Fragmentation": 1.0,
    "PileUpDataMC": 0.5,
    "PileUpPtBB": 0.5,
    "PileUpPtEC1": 0.5,
    "PileUpPtEC2": 0.5,
    "PileUpPtHF": 0.5,
    "PileUpPtRef": 0.5,
    "RelativeFSR": 0.5,
    "RelativeJEREC1": 0.0,
    "RelativeJEREC2": 0.0,
    "RelativeJERHF": 0.5,
    "RelativePtBB": 0.5,
    "RelativePtEC1": 0.0,
    "RelativePtEC2": 0.0,
    "RelativePtHF": 0.5,
    "RelativeBal": 0.5,
    "RelativeSample": 0.0,
    "RelativeStatEC": 0.0,
    "RelativeStatFSR": 0.0,
    "RelativeStatHF": 0.0,
    "SinglePionECAL": 1.0,
    "SinglePionHCAL": 1.0,
    "TimePtEta": 0.0,
}
JES_SOURCES = tuple(CORRELATION)
JER_RHO = 0.0

#### PRODUCTION DEFECT, found while running this study.  dijet_processor.py
#### (l.601) and trijet_processor.py (l.501) dispatch the jet variation with
#### ``elif 'JER' in jetsyst`` BEFORE ``elif "JES" in jetsyst``, so the three
#### JES sources whose NAME contains the substring "JER" never reach the JES
#### branch: they are filled with the JER smearing variation instead of their
#### own source.  zjet_processor.py is not affected -- it dispatches on
#### ``jet_syst == "JERUp"`` (exact equality).  The three sources are kept in
#### the 27-source total here because that is the requested prescription, but
#### the total WITHOUT them is reported alongside, and ``checks`` quantifies
#### how close each one is to the JER variation.
DEFECTIVE_SOURCES = ("RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF")
LEGS = ("corr", "uncorr_2016", "uncorr_2017", "uncorr_2018")
ARRAYS = ("A", "A_w2", "gen", "gen_w2", "reco", "reco_w2")


def split_factors(rho: float, mode: str) -> tuple[float, float]:
    """(f_corr, f_uncorr) for one source under one split convention."""
    if mode == "sqrt":
        return float(np.sqrt(rho)), float(np.sqrt(1.0 - rho))
    if mode == "linear":
        return float(rho), float(1.0 - rho)
    raise ValueError(f"unknown split mode {mode!r}")


# --------------------------------------------------------------------------
# stage 1: one pass over the pickles -> a small per-channel cache
# --------------------------------------------------------------------------
def category_names() -> list[str]:
    names = ["nominal"]
    for source in JES_SOURCES:
        names += [f"JES_{source}Up", f"JES_{source}Down"]
    names += ["JERUp", "JERDown"]
    return names


def _era_pickle(era: str, channel: str) -> Path:
    return (PKL_BASE / era / f"{channel}_mc"
            / f"minimal_rho_{channel}_mg_pythia8_{era}.pkl")


def build_cache(channel: str, groom: str = "g") -> Path:
    """Load each era pickle ONCE and store the regrouped arrays per category.

    Regrouping onto the adopted binning is linear, so summing the regrouped
    per-era arrays is identical to regrouping the summed histograms -- which
    is what makes the whole study fit in a few MB.
    """
    spec = ADOPTED[channel]
    redges = np.asarray(spec["rho"], float)
    pt_groups = spec["pt_groups"]
    keys = HIST_KEYS[groom]
    cats = category_names()
    slc = slice(1, -1)

    per_group: dict[str, dict[str, np.ndarray]] = {}
    pt_edges = None
    for group, eras in ERA_GROUPS.items():
        for era in eras:
            path = _era_pickle(era, channel)
            start = time.time()
            payload = pickle.load(open(path, "rb"))
            hists = {name: payload[keys[name]][{"dataset": sum}]
                     for name in ("reco", "gen", "matrix")}
            available = {str(c) for c in hists["reco"].axes["systematic"]}
            missing = [c for c in cats if c not in available]
            if missing:
                raise SystemExit(f"{era} {channel}: missing categories {missing}")

            pt_fine = np.asarray(hists["reco"].axes["ptreco"].edges)
            reco_groups = edge_groups(
                np.asarray(hists["reco"].axes["mpt_reco"].edges), redges)
            gen_groups = edge_groups(
                np.asarray(hists["gen"].axes["mpt_gen"].edges), redges)
            edges = np.array([pt_fine[g[0]] for g in pt_groups]
                             + [pt_fine[pt_groups[-1][-1] + 1]])
            if pt_edges is None:
                pt_edges = edges
            elif not np.allclose(pt_edges, edges):
                raise SystemExit(f"{era}: pt axis differs")

            def flat(values):
                x = regroup(regroup(values[slc, slc], 0, pt_groups),
                            1, reco_groups)
                return x.reshape(-1)

            def flat_gen(values):
                x = regroup(regroup(values[slc, slc], 0, pt_groups),
                            1, gen_groups)
                return x.reshape(-1)

            def flat_matrix(values):
                #### (ptreco, ptgen, rho_reco, rho_gen) -> (reco, gen)
                x = np.transpose(values[slc, slc, slc, slc], (0, 2, 1, 3))
                x = regroup(regroup(regroup(regroup(
                    x, 0, pt_groups), 1, reco_groups), 2, pt_groups),
                    3, gen_groups)
                return x.reshape(x.shape[0] * x.shape[1],
                                 x.shape[2] * x.shape[3])

            store = per_group.setdefault(group, {})
            for index, category in enumerate(cats):
                sel = {"systematic": category}
                blocks = {
                    "reco": (hists["reco"][sel], flat),
                    "gen": (hists["gen"][sel], flat_gen),
                    "A": (hists["matrix"][sel], flat_matrix),
                }
                for name, (h, fn) in blocks.items():
                    value = fn(h.values(flow=True))
                    var = fn(h.variances(flow=True))
                    w2 = f"{name}_w2"
                    for key, arr in ((name, value), (w2, var)):
                        if key not in store:
                            store[key] = np.zeros((len(cats),) + arr.shape)
                        store[key][index] += arr
            del payload, hists
            gc.collect()
            print(f"  {channel} {era} -> group {group}: "
                  f"{time.time() - start:.1f}s")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{channel}_era_cache.npz"
    payload = {"categories": np.array(cats),
               "groups": np.array(GROUPS),
               "pt_edges": pt_edges,
               "rho_edges": redges}
    for name in ARRAYS:
        payload[name] = np.stack([per_group[g][name] for g in GROUPS])
    np.savez_compressed(out, **payload)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


# --------------------------------------------------------------------------
# stage 2: legs in numpy, one tau=0 unfold each
# --------------------------------------------------------------------------
class Cache:
    """The era-resolved regrouped arrays for one channel."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.categories = [str(c) for c in z["categories"]]
        self.index = {c: i for i, c in enumerate(self.categories)}
        self.groups = [str(g) for g in z["groups"]]
        self.pt_edges = z["pt_edges"]
        self.rho_edges = z["rho_edges"]
        self.data = {name: z[name] for name in ARRAYS}

    def era(self, category: str, group: str) -> dict[str, np.ndarray]:
        g, c = self.groups.index(group), self.index[category]
        return {name: self.data[name][g, c] for name in ARRAYS}

    def run2(self, category: str) -> dict[str, np.ndarray]:
        c = self.index[category]
        return {name: self.data[name][:, c].sum(axis=0) for name in ARRAYS}

    def sigma(self, category: str, group: str) -> dict[str, np.ndarray]:
        varied = self.era(category, group)
        nominal = self.era("nominal", group)
        return {name: varied[name] - nominal[name] for name in ARRAYS}


def build_leg(nominal, sigmas, factor, groups):
    """nominal + factor * sum(sigmas over `groups`), variances clipped at 0."""
    leg = {}
    for name in ARRAYS:
        shift = sum(sigmas[g][name] for g in groups)
        value = nominal[name] + factor * shift
        leg[name] = np.maximum(value, 0.0) if name.endswith("_w2") else value
    return leg


def as_inputs(leg, cache: Cache, data: dict[str, np.ndarray]):
    inputs = {name: leg[name] for name in ARRAYS}
    inputs["pt_edges"] = cache.pt_edges
    inputs["rho_edges"] = cache.rho_edges
    for key in ("data_reco", "data_w2", "data_V"):
        inputs[key] = data[key]
    return inputs


def analyse_channel(channel: str, split_mode: str, verbose: bool = True):
    spec = SPECS[channel]
    groups = spec["rho_groups"]
    cache = Cache(CACHE_DIR / f"{channel}_era_cache.npz")
    reference = dict(np.load(OUT_DIR / "npz" / f"{channel}_run2_nominal.npz"))
    data = {key: reference[key] for key in ("data_reco", "data_w2", "data_V")}

    checks: dict[str, object] = {}

    #### validation 0 -- the cache, summed over era groups, must reproduce the
    #### Phase A Run 2 nominal NPZ that every other driver consumes.
    nominal_run2 = cache.run2("nominal")
    checks["cache_vs_phaseA_nominal_max_rel"] = {
        name: float(np.nanmax(np.abs(
            nominal_run2[name] - reference[name])
            / np.maximum(np.abs(reference[name]), 1e-12)))
        for name in ("A", "gen", "reco")
    }

    n_pt = len(cache.pt_edges) - 1
    n_rho = len(groups)
    rho_edges = cache.rho_edges
    gen_edges = np.r_[[rho_edges[g[0]] for g in groups], rho_edges[-1]]
    in_window = ((gen_edges[:-1] >= spec["norm"][0] - 1e-9)
                 & (gen_edges[1:] <= spec["norm"][1] + 1e-9))

    nominal_result, gen, nominal_response = unfold_variation(
        as_inputs(nominal_run2, cache, data), groups, f"_{channel}_jesnom")
    nominal_shape = normalized_shape(
        nominal_result["x"], n_pt, n_rho, in_window)
    shown = (np.tile(gen_edges[:-1] >= FLOOR, (n_pt, 1)).reshape(-1)
             & (gen > 0.0) & (nominal_response.sum(axis=0) > 0.0))
    core = shown & np.tile(in_window, n_pt)

    n_unfolds = 1
    null_legs = 0
    null_max_dev = 0.0
    shape_cache: dict[tuple, np.ndarray] = {}

    def leg_shape(source_key, direction, leg, factor, tag):
        """Normalized unfolded shape of one leg; null legs return nominal."""
        nonlocal n_unfolds, null_legs, null_max_dev
        key = (source_key, direction, leg, round(factor, 12))
        if key in shape_cache:
            return shape_cache[key], True
        if factor == 0.0:
            #### null leg: verify it is bit-identical to nominal, then skip
            members = GROUPS if leg == "corr" else (leg.split("_", 1)[1],)
            sigmas = {g: cache.sigma(f"{source_key}{direction}", g)
                      for g in members}
            built = build_leg(nominal_run2, sigmas, 0.0, members)
            dev = max(float(np.abs(built[name] - nominal_run2[name]).max())
                      for name in ARRAYS)
            null_max_dev = max(null_max_dev, dev)
            null_legs += 1
            shape_cache[key] = nominal_shape
            return nominal_shape, False
        members = GROUPS if leg == "corr" else (leg.split("_", 1)[1],)
        sigmas = {g: cache.sigma(f"{source_key}{direction}", g)
                  for g in members}
        built = build_leg(nominal_run2, sigmas, factor, members)
        result, _, _ = unfold_variation(
            as_inputs(built, cache, data), groups, tag)
        n_unfolds += 1
        shape = normalized_shape(result["x"], n_pt, n_rho, in_window)
        shape_cache[key] = shape
        return shape, False

    def propagate(source_key, rho, mode, label):
        """Four legs x Up/Down for one source -> per-leg and total bands."""
        f_corr, f_uncorr = split_factors(rho, mode)
        legs, per_leg_half = {}, []
        for leg in LEGS:
            factor = f_corr if leg == "corr" else f_uncorr
            shapes = {}
            for direction in ("Up", "Down"):
                shapes[direction], _ = leg_shape(
                    source_key, direction, leg, factor,
                    f"_{channel}_{label}_{leg}_{direction}_{factor:.4f}")
            half = 0.5 * np.abs(shapes["Up"] - shapes["Down"])
            legs[leg] = {
                "factor": factor,
                "null": factor == 0.0,
                "signed_up": relative(
                    shapes["Up"] - nominal_shape, nominal_shape).tolist(),
                "signed_down": relative(
                    shapes["Down"] - nominal_shape, nominal_shape).tolist(),
                "half_difference": relative(half, nominal_shape).tolist(),
            }
            per_leg_half.append(relative(half, nominal_shape))
        band = np.sqrt(sum(h ** 2 for h in per_leg_half))
        return legs, band

    #### ---- the 27 JES sources, both split conventions ------------------
    sources: dict[str, dict] = {}
    bands = {"sqrt": {}, "linear": {}}
    for source in JES_SOURCES:
        rho = CORRELATION[source]
        entry = {"rho": rho, "legs": {}, "band": {}, "core_summary": {}}
        for mode in ("sqrt", "linear"):
            legs, band = propagate(
                f"JES_{source}", rho, mode, f"{source}_{mode}")
            entry["legs"][mode] = legs
            entry["band"][mode] = band.tolist()
            entry["core_summary"][mode] = summarize(band, core)
            bands[mode][source] = band
        sources[source] = entry
        if verbose:
            print(f"  {source:<18} rho={rho:<4} "
                  f"sqrt {entry['core_summary']['sqrt']['median']:.3%} / "
                  f"{entry['core_summary']['sqrt']['max']:.3%}   "
                  f"linear {entry['core_summary']['linear']['median']:.3%} / "
                  f"{entry['core_summary']['linear']['max']:.3%}")

    totals = {mode: np.sqrt(sum(b ** 2 for b in bands[mode].values()))
              for mode in ("sqrt", "linear")}
    clean = {mode: np.sqrt(sum(b ** 2 for s, b in bands[mode].items()
                               if s not in DEFECTIVE_SOURCES))
             for mode in ("sqrt", "linear")}

    #### quantify the processor defect on THESE inputs: how close is each
    #### "RelativeJER*" category to the JER smearing variation?
    defect = {}
    for source in DEFECTIVE_SOURCES:
        for direction in ("Up", "Down"):
            jer = cache.run2(f"JER{direction}")["reco"]
            nom = cache.run2("nominal")["reco"]
            got = cache.run2(f"JES_{source}{direction}")["reco"]
            scale = float(np.abs(jer - nom).max())
            defect[f"{source}{direction}"] = float(
                np.abs(got - jer).max() / scale) if scale > 0 else float("nan")
    checks["relativeJER_sources_vs_JER_max_abs_over_JER_shift"] = defect

    #### ---- JER: the coherent band and the zjet-style rho=0 split -------
    jer_coherent_shapes = {}
    for direction in ("Up", "Down"):
        varied = cache.run2(f"JER{direction}")
        result, _, _ = unfold_variation(
            as_inputs(varied, cache, data), groups,
            f"_{channel}_jercoh_{direction}")
        n_unfolds += 1
        jer_coherent_shapes[direction] = normalized_shape(
            result["x"], n_pt, n_rho, in_window)
    jer_coherent = relative(
        0.5 * np.abs(jer_coherent_shapes["Up"] - jer_coherent_shapes["Down"]),
        nominal_shape)
    jer_legs, jer_decorrelated = propagate("JER", JER_RHO, "sqrt", "JER")

    with np.errstate(divide="ignore", invalid="ignore"):
        jer_ratio = np.where(jer_coherent > 0.0,
                             jer_decorrelated / jer_coherent, np.nan)

    #### ---- validation (a): the rho=1 sum rule --------------------------
    sum_rule = {}
    for source in ("FlavorQCD", "AbsoluteScale"):
        for direction in ("Up", "Down"):
            category = f"JES_{source}{direction}"
            sigmas = {g: cache.sigma(category, g) for g in GROUPS}
            leg = build_leg(nominal_run2, sigmas, 1.0, GROUPS)
            coherent = cache.run2(category)
            sum_rule[f"{source}{direction}"] = {
                name: float(np.abs(leg[name] - coherent[name]).max())
                for name in ("A", "gen", "reco")
            }
    checks["sum_rule_corr_leg_vs_coherent_max_abs"] = sum_rule
    checks["null_legs_skipped"] = int(null_legs)
    checks["null_leg_max_abs_deviation_from_nominal"] = float(null_max_dev)
    checks["n_unfolds"] = int(n_unfolds)

    #### the coherent JER built here must reproduce the Phase A JER NPZ
    jer_npz = OUT_DIR / "npz" / f"{channel}_run2_JERUp.npz"
    if jer_npz.exists():
        ref = dict(np.load(jer_npz))
        built = cache.run2("JERUp")
        checks["cache_vs_phaseA_JERUp_max_rel"] = {
            name: float(np.nanmax(np.abs(built[name] - ref[name])
                                  / np.maximum(np.abs(ref[name]), 1e-12)))
            for name in ("A", "gen", "reco")
        }

    ranked = sorted(
        ((source, float(np.nanmedian(bands["sqrt"][source][core])))
         for source in JES_SOURCES), key=lambda kv: -kv[1])

    output = {
        "channel": channel,
        "split_mode_default": split_mode,
        "gen_rho_edges": gen_edges.tolist(),
        "pt_edges": cache.pt_edges.tolist(),
        "n_pt": n_pt,
        "n_rho": n_rho,
        "shown": shown.tolist(),
        "core": core.tolist(),
        "in_window": in_window.tolist(),
        "norm_window": [float(v) for v in spec["norm"]],
        "nominal_shape": nominal_shape.tolist(),
        "sources": sources,
        "jes_total": {mode: totals[mode].tolist()
                      for mode in ("sqrt", "linear")},
        "jes_total_core_summary": {
            mode: summarize(totals[mode], core) for mode in ("sqrt", "linear")},
        "jes_total_shown_summary": {
            mode: summarize(totals[mode], shown) for mode in ("sqrt", "linear")},
        "defective_sources": list(DEFECTIVE_SOURCES),
        "jes_total_excl_defective": {
            mode: clean[mode].tolist() for mode in ("sqrt", "linear")},
        "jes_total_excl_defective_core_summary": {
            mode: summarize(clean[mode], core) for mode in ("sqrt", "linear")},
        "sqrt_over_linear_total": relative(
            totals["sqrt"] - totals["linear"], totals["linear"]).tolist(),
        "ranking_by_core_median": [
            {"source": s, "core_median": v} for s, v in ranked],
        "jer": {
            "coherent": jer_coherent.tolist(),
            "decorrelated": jer_decorrelated.tolist(),
            "decorrelated_legs": jer_legs,
            "ratio_decorrelated_over_coherent": jer_ratio.tolist(),
            "core_summary": {
                "coherent": summarize(jer_coherent, core),
                "decorrelated": summarize(jer_decorrelated, core),
                "ratio": summarize(jer_ratio, core),
            },
        },
        "checks": checks,
    }
    if verbose:
        summary = output["jes_total_core_summary"]
        print(f"\n{channel}: {int(shown.sum())} reported bins, "
              f"{int(core.sum())} core, {n_unfolds} unfolds, "
              f"{null_legs} null legs skipped")
        print(f"  JES total (sqrt)   {summary['sqrt']['median']:.3%} median / "
              f"{summary['sqrt']['max']:.3%} max")
        print(f"  JES total (linear) {summary['linear']['median']:.3%} median / "
              f"{summary['linear']['max']:.3%} max")
        clean_summary = output["jes_total_excl_defective_core_summary"]["sqrt"]
        print(f"  JES excl. RelativeJER* (sqrt) "
              f"{clean_summary['median']:.3%} median / "
              f"{clean_summary['max']:.3%} max")
        print(f"  JER coherent       "
              f"{output['jer']['core_summary']['coherent']['median']:.3%} "
              f"median / "
              f"{output['jer']['core_summary']['coherent']['max']:.3%} max")
        print(f"  JER decorrelated   "
              f"{output['jer']['core_summary']['decorrelated']['median']:.3%} "
              f"median / "
              f"{output['jer']['core_summary']['decorrelated']['max']:.3%} max")
        print("  top 3: " + ", ".join(
            f"{s} {v:.3%}" for s, v in ranked[:3]))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", action="append",
                        choices=("dijet", "trijet"),
                        help="channel to run; repeat for both (default: both)")
    parser.add_argument("--split-mode", choices=("sqrt", "linear"),
                        default="sqrt",
                        help="which convention is the reported default; both "
                             "are always computed")
    parser.add_argument("--binning", default="coarse_tail",
                        help="gen merge; coarse_tail is production")
    parser.add_argument("--build-cache", action="store_true",
                        help="(re)build the per-era regrouped cache NPZ from "
                             "the pickles and exit")
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "hadronic_rho_jes_run2.json")
    args = parser.parse_args(argv)

    channels = args.channel or ["dijet", "trijet"]
    if args.build_cache:
        for channel in channels:
            print(f"== cache {channel}")
            build_cache(channel)
        return 0

    use_binning(args.binning)
    output = {"prescription": {
        "era_groups": {g: list(e) for g, e in ERA_GROUPS.items()},
        "correlation": CORRELATION,
        "legs": list(LEGS),
        "split_modes": {
            "sqrt": "f_corr = sqrt(rho), f_uncorr = sqrt(1 - rho) [default]",
            "linear": "f_corr = rho, f_uncorr = 1 - rho [zjet legacy]",
        },
        "binning": args.binning,
        "default_split_mode": args.split_mode,
    }}
    for channel in channels:
        print(f"== {channel}")
        output[channel] = analyse_channel(channel, args.split_mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
