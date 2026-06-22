"""Diagnostics for the dijet rho response: conditioning, purity, stability.

Run:  python -m scripts.dijet_methods.diagnose   (from repo root, venv active)
"""

from __future__ import annotations

import numpy as np

from scripts.dijet_methods.loader import (
    Binning, NativeInputs, load_native, rebin, build_problem,
)

MC = "inputs/dijet/rho/minimal_rho_dijet_mg_pythia8_2018.pkl"
DATA = "inputs/dijet/rho/minimal_rho_dijet_data_2018.pkl"

# Analysis pt binning used by the committed result (drops the 480/680/820 splits)
PT6 = np.array([0.0, 200.0, 290.0, 400.0, 570.0, 760.0, 13000.0])
PT_REPORT_FROM = 1  # report pt bins >= this index (i.e. 200 GeV+)


def native_binning(nat: NativeInputs) -> Binning:
    return Binning(nat.pt_edges_reco.copy(), nat.rho_edges_reco.copy(),
                   nat.rho_edges_gen.copy())


def slice_block(prob, ptr, ptg):
    """rho-reco x rho-gen response block for a (reco pt, gen pt) pair."""
    rm = prob.reco_pt_idx == ptr
    gm = prob.gen_pt_idx == ptg
    return prob.R[np.ix_(rm, gm)]


def report(mode, binning, label):
    nat = load_native(MC, mode)
    rb = rebin(nat, binning)
    prob = build_problem(rb)
    nptg = len(binning.pt_edges) - 1
    print(f"\n################  {mode.upper()}  |  {label}  ################")
    print(f"pt edges: {binning.pt_edges}")
    print(f"rho_gen edges ({len(binning.rho_edges_gen)-1} bins): {binning.rho_edges_gen}")
    print(f"rho_reco edges ({len(binning.rho_edges_reco)-1} bins): {binning.rho_edges_reco}")
    print(f"total gen={prob.gen.sum():.3e}  reco={prob.reco.sum():.3e}  "
          f"matched={prob.R.sum():.3e}  fakes={prob.fakes.sum():.3e}  "
          f"misses={prob.misses.sum():.3e}")
    for ptg in range(nptg):
        if ptg < PT_REPORT_FROM:
            tag = "(sink)"
        else:
            tag = ""
        gm = prob.gen_pt_idx == ptg
        eff = prob.eff[gm]
        # same-pt rho block, column-normalized to matched
        blk = slice_block(prob, ptg, ptg)
        colsum = blk.sum(axis=0)
        Pblk = np.divide(blk, colsum, out=np.zeros_like(blk), where=colsum > 0)
        # purity per reco rho bin within this pt: diag/rowtotal (compressed)
        # stability per gen rho bin: diag-ish / colsum
        # pt migration: fraction of matched-from-this-gen-pt that lands in same reco pt
        all_reco_from_ptg = prob.R[:, gm].sum()
        same_pt_reco = blk.sum()
        pt_diag_frac = same_pt_reco / all_reco_from_ptg if all_reco_from_ptg else np.nan
        # condition number of the same-pt probability block (square-ish)
        try:
            cond = np.linalg.cond(Pblk) if Pblk.shape[0] == Pblk.shape[1] else np.linalg.cond(Pblk.T @ Pblk)
        except Exception:
            cond = np.nan
        ptlo, pthi = binning.pt_edges[ptg], binning.pt_edges[ptg + 1]
        print(f"  pt[{ptlo:.0f},{pthi:.0f}] {tag:6s} "
              f"eff=[{eff.min():.2f},{eff.max():.2f}] med={np.median(eff):.2f}  "
              f"pt-diag={pt_diag_frac:.3f}  cond(Pblk)={cond:.1f}  "
              f"rho_block={blk.shape}")


def main():
    for mode in ("groomed", "ungroomed"):
        nat = load_native(MC, mode)
        # 1) native fine binning, native pt
        report(mode, native_binning(nat), "NATIVE fine")
        # 2) 6-pt binning, native rho
        report(mode, Binning(PT6, nat.rho_edges_reco.copy(),
                             nat.rho_edges_gen.copy()), "PT6 + native rho")


if __name__ == "__main__":
    main()
