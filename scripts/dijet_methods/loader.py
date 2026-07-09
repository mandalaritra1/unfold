"""Standalone numpy loader for dijet rho unfolding inputs.

Loads the producer pickles, sums over the dataset axis, selects a systematic,
and exposes the 4D response / 2D reco / 2D gen arrays at *native* binning. A
flexible rebinner then merges pt and rho edges (any target edge set that is a
subset of the native edges) so we can study arbitrary binnings.

This is intentionally independent of ROOT/TUnfold and the repo Unfolder so a
method bake-off can iterate fast on pure numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np


MODE_KEYS = {
    "groomed": {
        "response": "response_matrix_rho_g",
        "reco": "ptjet_rhojet_g_reco",
        "gen": "ptjet_rhojet_g_gen",
    },
    "ungroomed": {
        "response": "response_matrix_rho_u",
        "reco": "ptjet_rhojet_u_reco",
        "gen": "ptjet_rhojet_u_gen",
    },
}


_PKL_CACHE = {}


def _load(path):
    path = str(path)
    if path not in _PKL_CACHE:
        with open(path, "rb") as fh:
            _PKL_CACHE[path] = pickle.load(fh)
    return _PKL_CACHE[path]


def list_systematics(pkl_path, mode):
    """Systematics available on the response matrix for a mode."""
    d = _load(pkl_path)
    h = d[MODE_KEYS[mode]["response"]]
    return list(h.axes["systematic"]) if "systematic" in h.axes.name else ["nominal"]


@dataclass
class NativeInputs:
    """Native-binning arrays for one mode.

    response: (nptreco, nptgen, nrhoreco, nrhogen)
    reco:     (nptreco, nrhoreco)
    gen:      (nptgen, nrhogen)
    """

    pt_edges_reco: np.ndarray
    pt_edges_gen: np.ndarray
    rho_edges_reco: np.ndarray
    rho_edges_gen: np.ndarray
    response: np.ndarray
    response_var: np.ndarray | None
    reco: np.ndarray
    reco_var: np.ndarray | None
    gen: np.ndarray
    gen_var: np.ndarray | None


def load_native(pkl_path, mode, systematic="nominal", need_gen=True):
    """Load native arrays. Data pickles lack a response/gen -> those stay None
    and pt/rho edges are taken from the reco histogram instead."""
    d = _load(pkl_path)
    keys = MODE_KEYS[mode]
    has_response = keys["response"] in d
    resp = resp_var = gen = gen_var = None
    reco, reco_var = _project_2d(d[keys["reco"]], systematic, ("ptreco", "mpt_reco"))

    if has_response:
        hr = d[keys["response"]]
        resp, resp_var = _project_4d(hr, systematic)
        ax = hr.axes
        pt_edges_reco = np.array(ax["ptreco"].edges)
        pt_edges_gen = np.array(ax["ptgen"].edges)
        rho_edges_reco = np.array(ax["mpt_reco"].edges)
        rho_edges_gen = np.array(ax["mpt_gen"].edges)
        if need_gen and keys["gen"] in d:
            gsys = systematic if _has_syst(d[keys["gen"]], systematic) else "nominal"
            gen, gen_var = _project_2d(d[keys["gen"]], gsys, ("ptgen", "mpt_gen"))
    else:
        ax = d[keys["reco"]].axes
        pt_edges_reco = pt_edges_gen = np.array(ax["ptreco"].edges)
        rho_edges_reco = rho_edges_gen = np.array(ax["mpt_reco"].edges)

    return NativeInputs(
        pt_edges_reco=pt_edges_reco,
        pt_edges_gen=pt_edges_gen,
        rho_edges_reco=rho_edges_reco,
        rho_edges_gen=rho_edges_gen,
        response=resp,
        response_var=resp_var,
        reco=reco,
        reco_var=reco_var,
        gen=gen,
        gen_var=gen_var,
    )


def _has_syst(h, systematic):
    return "systematic" in h.axes.name and systematic in list(h.axes["systematic"])


def _project_4d(h, systematic):
    sel = h[{"systematic": systematic}] if _has_syst(h, systematic) else h
    if "dataset" in sel.axes.name:
        sel = sel[{"dataset": sum}]
    p = sel.project("ptreco", "ptgen", "mpt_reco", "mpt_gen")
    var = p.variances(flow=False)
    return (
        np.asarray(p.values(flow=False), dtype=float),
        None if var is None else np.asarray(var, dtype=float),
    )


def _project_2d(h, systematic, axes):
    sel = h[{"systematic": systematic}] if _has_syst(h, systematic) else h
    if "dataset" in sel.axes.name:
        sel = sel[{"dataset": sum}]
    p = sel.project(*axes)
    var = p.variances(flow=False)
    return (
        np.asarray(p.values(flow=False), dtype=float),
        None if var is None else np.asarray(var, dtype=float),
    )


# --------------------------------------------------------------------------
# Rebinning: group native bins into target edges (subset of native edges).
# --------------------------------------------------------------------------
def _group_indices(native_edges, target_edges, tol=1e-6):
    """For each target bin, the list of native bin indices it contains."""
    native_edges = np.asarray(native_edges, float)
    target_edges = np.asarray(target_edges, float)
    # locate each target edge in native edges
    edge_idx = []
    for e in target_edges:
        j = np.where(np.abs(native_edges - e) < tol)[0]
        if len(j) == 0:
            raise ValueError(f"target edge {e} not present in native edges {native_edges}")
        edge_idx.append(int(j[0]))
    groups = []
    for a, b in zip(edge_idx[:-1], edge_idx[1:]):
        groups.append(list(range(a, b)))
    return groups


@dataclass
class Binning:
    pt_edges: np.ndarray      # shared reco/gen pt edges
    rho_edges_reco: np.ndarray
    rho_edges_gen: np.ndarray


@dataclass
class Rebinned:
    binning: Binning
    response: np.ndarray      # (nptreco, nptgen, nrhoreco, nrhogen)
    response_var: np.ndarray | None
    reco: np.ndarray          # (nptreco, nrhoreco)
    reco_var: np.ndarray | None
    gen: np.ndarray | None    # (nptgen, nrhogen)
    gen_var: np.ndarray | None


def _sum_groups_axis(arr, groups, axis):
    """Sum arr along `axis` collapsing native bins into target groups."""
    out = np.add.reduceat(
        np.take(arr, [i for g in groups for i in g], axis=axis),
        np.cumsum([0] + [len(g) for g in groups])[:-1],
        axis=axis,
    )
    return out


def rebin(native: NativeInputs, binning: Binning) -> Rebinned:
    pt_g = _group_indices(native.pt_edges_reco, binning.pt_edges)
    rhor_g = _group_indices(native.rho_edges_reco, binning.rho_edges_reco)
    rhog_g = _group_indices(native.rho_edges_gen, binning.rho_edges_gen)

    def rb4(a):
        if a is None:
            return None
        a = _sum_groups_axis(a, pt_g, 0)     # ptreco
        a = _sum_groups_axis(a, pt_g, 1)     # ptgen
        a = _sum_groups_axis(a, rhor_g, 2)   # rhoreco
        a = _sum_groups_axis(a, rhog_g, 3)   # rhogen
        return a

    def rb_reco(a):
        if a is None:
            return None
        a = _sum_groups_axis(a, pt_g, 0)
        a = _sum_groups_axis(a, rhor_g, 1)
        return a

    def rb_gen(a):
        if a is None:
            return None
        a = _sum_groups_axis(a, pt_g, 0)
        a = _sum_groups_axis(a, rhog_g, 1)
        return a

    return Rebinned(
        binning=binning,
        response=rb4(native.response),
        response_var=rb4(native.response_var),
        reco=rb_reco(native.reco),
        reco_var=rb_reco(native.reco_var),
        gen=rb_gen(native.gen),
        gen_var=rb_gen(native.gen_var),
    )


# --------------------------------------------------------------------------
# Unrolled problem builder: (pt, rho) -> flat vectors + response matrix.
# --------------------------------------------------------------------------
@dataclass
class Problem:
    """Unrolled unfolding problem over a chosen pt range.

    R:      (Nreco, Ngen) matched response counts
    reco:   (Nreco,) measured total = matched + fakes
    gen:    (Ngen,) truth total = matched + misses (None if no gen, e.g. data)
    fakes:  (Nreco,)
    misses: (Ngen,)
    eff:    (Ngen,) = matched_gen / gen_total
    purity: (Nreco,) = matched_reco / reco_total
    P:      (Nreco, Ngen) folding probabilities R[i,j]/gen_total[j]
    index maps to (pt_bin, rho_bin)
    """

    R: np.ndarray
    reco: np.ndarray
    reco_var: np.ndarray | None
    gen: np.ndarray | None
    gen_var: np.ndarray | None
    fakes: np.ndarray
    misses: np.ndarray | None
    eff: np.ndarray | None
    P: np.ndarray | None
    reco_pt_idx: np.ndarray
    reco_rho_idx: np.ndarray
    gen_pt_idx: np.ndarray
    gen_rho_idx: np.ndarray
    binning: Binning


def build_problem(mc: Rebinned, data_reco: np.ndarray | None = None,
                  data_reco_var: np.ndarray | None = None) -> Problem:
    """Unroll the rebinned MC (and optional data) into vectors + matrix.

    All pt bins present in the binning are kept (the lowest, 0-200, acts as a
    migration sink/underflow); reporting code can drop it later.
    """
    nptr, nrr = mc.reco.shape
    nptg, nrg = mc.gen.shape

    # unroll order: pt-major, rho-minor
    Nreco = nptr * nrr
    Ngen = nptg * nrg
    R = mc.response.transpose(0, 2, 1, 3).reshape(Nreco, Ngen)
    R_var = (
        mc.response_var.transpose(0, 2, 1, 3).reshape(Nreco, Ngen)
        if mc.response_var is not None else None
    )
    reco_mc = mc.reco.reshape(Nreco)
    gen_mc = mc.gen.reshape(Ngen) if mc.gen is not None else None

    matched_reco = R.sum(axis=1)
    matched_gen = R.sum(axis=0)
    fakes = reco_mc - matched_reco
    misses = (gen_mc - matched_gen) if gen_mc is not None else None
    eff = np.divide(matched_gen, gen_mc, out=np.zeros_like(matched_gen),
                    where=gen_mc > 0) if gen_mc is not None else None
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(gen_mc > 0, R / gen_mc[None, :], 0.0) if gen_mc is not None else None

    reco_vec = data_reco.reshape(Nreco) if data_reco is not None else reco_mc
    reco_var_vec = (
        data_reco_var.reshape(Nreco) if data_reco_var is not None
        else (mc.reco_var.reshape(Nreco) if mc.reco_var is not None else None)
    )

    reco_pt_idx = np.repeat(np.arange(nptr), nrr)
    reco_rho_idx = np.tile(np.arange(nrr), nptr)
    gen_pt_idx = np.repeat(np.arange(nptg), nrg)
    gen_rho_idx = np.tile(np.arange(nrg), nptg)

    return Problem(
        R=R, reco=reco_vec, reco_var=reco_var_vec, gen=gen_mc,
        gen_var=(mc.gen_var.reshape(Ngen) if mc.gen_var is not None else None),
        fakes=fakes, misses=misses, eff=eff, P=P,
        reco_pt_idx=reco_pt_idx, reco_rho_idx=reco_rho_idx,
        gen_pt_idx=gen_pt_idx, gen_rho_idx=gen_rho_idx, binning=mc.binning,
    )
