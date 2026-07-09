"""Quick closure + data-unfold sanity test for the methods on PT6+native rho."""

from __future__ import annotations

import numpy as np

from scripts.dijet_methods.loader import Binning, load_native, rebin, build_problem
from scripts.dijet_methods import methods as M

MC = "inputs/dijet/rho/minimal_rho_dijet_mg_pythia8_2018.pkl"
DATA = "inputs/dijet/rho/minimal_rho_dijet_data_2018.pkl"
HER = "inputs/dijet/rho/minimal_rho_dijet_herwig_2018.pkl"
PT6 = np.array([0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0])


def per_pt_ratio(prob, est, truth, label):
    print(f"  --- {label}: unfolded/truth per pt (reported 200+ only) ---")
    for ptg in np.unique(prob.gen_pt_idx):
        if ptg == 0:
            continue
        m = prob.gen_pt_idx == ptg
        t = truth[m]
        e = est[m]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(t > 0, e / t, np.nan)
        good = np.isfinite(r) & (t > t.max() * 1e-3)
        if good.any():
            print(f"    ptbin{ptg}: ratio median={np.nanmedian(r[good]):.3f} "
                  f"min={np.nanmin(r[good]):.3f} max={np.nanmax(r[good]):.3f} "
                  f"(n={good.sum()})")


def main():
    mode = "groomed"
    nat = load_native(MC, mode)
    binning = Binning(PT6, nat.rho_edges_reco.copy(), nat.rho_edges_gen.copy())
    rb = rebin(nat, binning)
    prob = build_problem(rb)

    truth = prob.gen

    # 1) self-closure from MC matched reco, FLAT prior (hard test of convergence)
    mc_matched = M.fold(prob, truth)
    flat = np.where(truth > 0, truth.mean(), 0.0)
    print("=== D'Agostini self-closure, FLAT prior ===")
    for nit in (1, 2, 4, 8, 16):
        est = M.dagostini(prob, mc_matched, n_iter=nit, prior=flat)
        per_pt_ratio(prob, est, truth, f"n_iter={nit}")

    # 2) data unfold
    datn = load_native(DATA, mode, need_gen=False)
    datrb = rebin(datn, binning)
    data_meas = datrb.reco.reshape(-1)
    data_matched = M.subtract_fakes(data_meas, prob)
    print("\n=== Data unfold (D'Agostini, MC prior) ===")
    for nit in (1, 2, 4, 8):
        est = M.dagostini(prob, data_matched, n_iter=nit)
        per_pt_ratio(prob, est, truth, f"data n_iter={nit}")

    # 3) HERWIG-as-data model closure (unfold herwig reco w/ pythia response)
    her = load_native(HER, mode)
    herrb = rebin(her, binning)
    her_meas = herrb.reco.reshape(-1)
    her_matched = M.subtract_fakes(her_meas, prob)
    her_truth = herrb.gen.reshape(-1)
    print("\n=== HERWIG-as-data (unfold w/ PYTHIA response), ratio to HERWIG truth ===")
    for nit in (2, 4, 8):
        est = M.dagostini(prob, her_matched, n_iter=nit)
        per_pt_ratio(prob, est, her_truth, f"herwig n_iter={nit}")


if __name__ == "__main__":
    main()
