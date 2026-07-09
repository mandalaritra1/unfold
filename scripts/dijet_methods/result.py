"""Unfolding + statistical-uncertainty propagation via toys.

Propagates two stat sources through any method:
  - data statistics (Gaussian toys using the measured variance)
  - MC response-matrix statistics (Gaussian toys on R using response_var)
and combines them into a covariance for the unfolded truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scripts.dijet_methods.loader import (
    Binning, load_native, rebin, build_problem,
)
from scripts.dijet_methods import methods as M

import os

MC = "inputs/dijet/rho/minimal_rho_dijet_mg_pythia8_2018.pkl"
DATA = "inputs/dijet/rho/minimal_rho_dijet_data_2018.pkl"
HER = "inputs/dijet/rho/minimal_rho_dijet_herwig_2018.pkl"


def channel_paths(channel, year=2018, zjet_tag="original"):
    """Return (mc, data, herwig|None) pickle paths for a channel.

    zjet stores merged-era rho as pythia_all / data_all / herwig_all under
    inputs/zjet/rho/<tag>/ (default tag 'original'); dijet/trijet use a single
    minimal_rho_<channel>_<sample>_<year>.pkl per sample.
    """
    if channel == "zjet":
        base = f"inputs/zjet/rho/{zjet_tag}"
        her = f"{base}/herwig_all.pkl"
        return (f"{base}/pythia_all.pkl", f"{base}/data_all.pkl",
                her if os.path.exists(her) else None)
    base = f"inputs/{channel}/rho/minimal_rho_{channel}"
    mc = f"{base}_mg_pythia8_{year}.pkl"
    data = f"{base}_data_{year}.pkl"
    her = f"{base}_herwig_{year}.pkl"
    return mc, data, (her if os.path.exists(her) else None)


@dataclass
class Prepared:
    mode: str
    binning: Binning
    prob: object
    nat_mc: object
    rb_mc: object
    data_meas: np.ndarray
    data_var: np.ndarray
    data_matched: np.ndarray
    her_truth: np.ndarray | None
    her_matched: np.ndarray | None
    channel: str = "dijet"


def prepare(mode, binning, channel="dijet") -> Prepared:
    mc_path, data_path, her_path = channel_paths(channel)
    nat = load_native(mc_path, mode)
    rb = rebin(nat, binning)
    prob = build_problem(rb)

    datn = load_native(data_path, mode, need_gen=False)
    datrb = rebin(datn, binning)
    data_meas = datrb.reco.reshape(-1)
    data_var = (datrb.reco_var.reshape(-1) if datrb.reco_var is not None
                else data_meas.copy())
    data_matched = M.subtract_fakes(data_meas, prob)

    her_truth = her_matched = None
    if her_path:
        her = load_native(her_path, mode)
        herrb = rebin(her, binning)
        her_truth = herrb.gen.reshape(-1)
        her_matched = M.subtract_fakes(herrb.reco.reshape(-1), prob)

    return Prepared(mode, binning, prob, nat, rb, data_meas, data_var,
                    data_matched, her_truth, her_matched, channel)


def _rebuild_prob_from_R(prob, R_toy):
    """Rebuild a lightweight problem with a toyed response matrix R_toy."""
    import copy
    p = copy.copy(prob)
    p.R = R_toy
    gen_tot = prob.gen
    matched_gen = R_toy.sum(axis=0)
    p.eff = np.divide(matched_gen, gen_tot, out=np.zeros_like(matched_gen),
                      where=gen_tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p.P = np.where(gen_tot > 0, R_toy / gen_tot[None, :], 0.0)
    return p


def unfold_with_unc(prep: Prepared, method, *, measured=None, measured_var=None,
                    n_toys=200, response_toys=True, seed=0, **kw):
    """Return (central, std, cov, toy_matrix).

    method(prob, measured_matched, **kw) -> truth estimate.
    """
    prob = prep.prob
    if measured is None:
        measured = prep.data_meas
        measured_var = prep.data_var
    rng = np.random.default_rng(seed)

    central = method(prob, M.subtract_fakes(measured, prob), **kw)

    # response-stat: precompute std of R
    R_std = None
    if response_toys and prep.rb_mc.response_var is not None:
        Rv = prep.rb_mc.response_var.transpose(0, 2, 1, 3).reshape(prob.R.shape)
        R_std = np.sqrt(np.clip(Rv, 0, None))

    toys = np.empty((n_toys, prob.R.shape[1]))
    for t in range(n_toys):
        m_toy = measured + rng.normal(size=measured.shape) * np.sqrt(np.clip(measured_var, 0, None))
        m_toy = np.clip(m_toy, 0, None)
        p_use = prob
        if R_std is not None:
            R_toy = np.clip(prob.R + rng.normal(size=prob.R.shape) * R_std, 0, None)
            p_use = _rebuild_prob_from_R(prob, R_toy)
        toys[t] = method(p_use, M.subtract_fakes(m_toy, p_use), **kw)

    cov = np.cov(toys, rowvar=False)
    std = toys.std(axis=0)
    return central, std, cov, toys


def _pair_systematics(systs):
    """Group '<name>Up'/'<name>Down' into pairs; ignore lone/nominal entries."""
    pairs = {}
    for s in systs:
        if s.endswith("Up"):
            pairs.setdefault(s[:-2], {})["up"] = s
        elif s.endswith("Down"):
            pairs.setdefault(s[:-4], {})["down"] = s
    return {k: v for k, v in pairs.items() if "up" in v and "down" in v}


def systematic_band(prep: Prepared, method, *, n_iter=3, **kw):
    """Response-modeling systematics: vary the response to each systematic,
    re-unfold the SAME (nominal) data, quadrature-sum the shifts.

    Returns (central, syst_std, per_source dict).
    """
    from scripts.dijet_methods.loader import list_systematics, load_native, rebin, build_problem

    mode, binning = prep.mode, prep.binning
    mc_path, _, _ = channel_paths(prep.channel)
    central = method(prep.prob, prep.data_matched, n_iter=n_iter, **kw)
    systs = list_systematics(mc_path, mode)
    pairs = _pair_systematics(systs)
    per_source = {}
    var = np.zeros_like(central)
    for name, ud in pairs.items():
        shifts = []
        for key in ("up", "down"):
            nat_s = load_native(mc_path, mode, systematic=ud[key])
            rb_s = rebin(nat_s, binning)
            prob_s = build_problem(rb_s)
            dm_s = M.subtract_fakes(prep.data_meas, prob_s)
            u_s = method(prob_s, dm_s, n_iter=n_iter, **kw)
            shifts.append(u_s - central)
        sym = 0.5 * (np.abs(shifts[0]) + np.abs(shifts[1]))
        per_source[name] = sym
        var += sym**2
    return central, np.sqrt(var), per_source
