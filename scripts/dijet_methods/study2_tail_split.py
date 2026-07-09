"""Should the groomed coarse binning split its [-10,-3] tail bin?

Candidates (groomed), all with the production convention of merging the reco
bins below the first interior gen edge:
  push_B : gen [-10,-3, 0.5-wide...]              (current production coarse)
  addm4  : gen [-10,-4,-3, 0.5-wide...]           (one extra 1.0-wide bin)
  addm35 : gen [-10,-3.5,-3, 0.5-wide...]         (one extra 0.5-wide bin)
  addm4m35: gen [-10,-4,-3.5,-3, 0.5-wide...]     (two extra bins)

For each: per-bin purity/stability/efficiency in the tail region, cond(P),
unregularized weighted lstsq on data (negative bins, oscillation, 100-toy
data-stat spread) and agreement with D'Agostini n=4.

Run: .venv/bin/python -m scripts.dijet_methods.study2_tail_split
"""
from __future__ import annotations

import json
import os

import numpy as np

from scripts.dijet_methods.loader import Binning
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.result import prepare

OUT = "outputs/dijet/2018/rho/method_study2/tail_split"
os.makedirs(OUT, exist_ok=True)

PT_EDGES = np.array([0, 200, 290, 400, 570, 760, 13000.0])
NATIVE_RECO = np.array([-10, -8, -7, -6, -5.5, -5, -4.75, -4.5, -4.25, -4,
                        -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75,
                        -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0])
HIGH = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
HIGH_U = [-1.5, -1.0, -0.5, 0.0]

CANDS_BY_MODE = {
    "groomed": {
        "push_B":   [-10.0, -3.0] + HIGH,
        "addm4":    [-10.0, -4.0, -3.0] + HIGH,
        "addm35":   [-10.0, -3.5, -3.0] + HIGH,
        "addm4m35": [-10.0, -4.0, -3.5, -3.0] + HIGH,
        "buf5_wide": [-10.0, -5.0, -3.0] + HIGH,
        "buf45_wide": [-10.0, -4.5, -3.0] + HIGH,
        "buf6_wide": [-10.0, -6.0, -3.0] + HIGH,
        "buf5_m4":   [-10.0, -5.0, -4.0, -3.0] + HIGH,
        "buf6_m4":   [-10.0, -6.0, -4.0, -3.0] + HIGH,
    },
    "ungroomed": {
        "push_B":    [-10.0, -2.0] + HIGH_U,
        "buf3":      [-10.0, -3.0, -2.0] + HIGH_U,
        "buf4_wide": [-10.0, -4.0, -2.0] + HIGH_U,
        "buf4_m3":   [-10.0, -4.0, -3.0, -2.0] + HIGH_U,
        "buf3_m25":  [-10.0, -3.0, -2.5, -2.0] + HIGH_U,
    },
}

N_TOYS = 100
RNG = np.random.default_rng(7)


def merged_reco(first_interior_gen_edge):
    """Native reco edges with everything below the edge merged into one bin."""
    keep = NATIVE_RECO[NATIVE_RECO >= first_interior_gen_edge - 1e-9]
    return np.concatenate([[-10.0], keep]) if keep[0] > -10 else keep


def weighted_lstsq(prob, matched, var):
    v = np.maximum(var, np.maximum(matched, 1.0))
    w = 1.0 / np.sqrt(v)
    A = prob.P * w[:, None]
    x, *_ = np.linalg.lstsq(A, matched * w, rcond=None)
    return x


def purity_stability(prob):
    """Per gen bin: purity/stability via mapping reco bins into gen windows."""
    gedges = np.asarray(prob.binning.rho_edges_gen)
    redges = np.asarray(prob.binning.rho_edges_reco)
    rcent = 0.5 * (redges[:-1] + redges[1:])
    # reco (pt, rho-bin) -> gen rho window index
    rho_to_gen = np.digitize(rcent, gedges) - 1
    rho_to_gen = np.clip(rho_to_gen, 0, len(gedges) - 2)
    pur, stab = {}, {}
    for j in range(prob.P.shape[1]):
        ptg, rg = prob.gen_pt_idx[j], prob.gen_rho_idx[j]
        same = (prob.reco_pt_idx == ptg) & (rho_to_gen[prob.reco_rho_idx] == rg)
        col = prob.R[:, j]
        row_tot = prob.R[same].sum()
        stab[j] = col[same].sum() / col.sum() if col.sum() > 0 else 0.0
        pur[j] = col[same].sum() / row_tot if row_tot > 0 else 0.0
    return pur, stab


def run(mode="groomed"):
    out = {}
    for name, gen_edges in CANDS_BY_MODE[mode].items():
        gen_edges = np.array(gen_edges)
        b = Binning(pt_edges=PT_EDGES,
                    rho_edges_reco=merged_reco(gen_edges[1]),
                    rho_edges_gen=gen_edges)
        prep = prepare(mode, b)
        prob = prep.prob
        rep = (prob.gen_pt_idx >= 1) & (prob.gen_rho_idx >= 1)  # published: buffer bin hidden
        var = prep.data_var

        P = prob.P
        s = np.linalg.svd(P, compute_uv=False)
        cond = float(s[0] / s[s > 1e-12][-1])

        u = weighted_lstsq(prob, prep.data_matched, var)
        dago = M.dagostini(prob, prep.data_matched, n_iter=4)
        toys = np.empty((N_TOYS, len(u)))
        for t in range(N_TOYS):
            m_toy = np.clip(prep.data_meas + RNG.normal(size=var.shape) * np.sqrt(np.maximum(var, 1)), 0, None)
            toys[t] = weighted_lstsq(prob, M.subtract_fakes(m_toy, prob), var)
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = np.where(u != 0, toys.std(axis=0) / np.abs(u), np.nan)
            rr = np.where(prob.gen > 0, u / prob.gen, np.nan)
            agree = np.where(dago != 0, np.abs(u / dago - 1), np.nan)
        # oscillation within reported pt slices
        d2 = []
        for pt in range(1, 6):
            idx = np.where((prob.gen_pt_idx == pt) & (prob.gen_rho_idx >= 1))[0]
            r = rr[idx]
            d2.extend((r[2:] - 2 * r[1:-1] + r[:-2]).tolist())
        pur, stabd = purity_stability(prob)

        # tail-region details (gen rho windows below -2.5), reported slices
        tail = {}
        for j in np.where((prob.gen_pt_idx >= 1))[0]:
            gl = prob.binning.rho_edges_gen[prob.gen_rho_idx[j]]
            gh = prob.binning.rho_edges_gen[prob.gen_rho_idx[j] + 1]
            if gh <= (-2.5 if mode == "groomed" else -1.5) + 1e-9:
                tag = "BUF " if prob.gen_rho_idx[j] == 0 else ""
                key = f"{tag}pt{prob.gen_pt_idx[j]}_[{gl:g},{gh:g}]"
                tail[key] = dict(purity=round(pur[j], 3), stability=round(stabd[j], 3),
                                 eff=round(float(prob.eff[j]), 3),
                                 u_over_gen=round(float(rr[j]), 3),
                                 toy_spread=round(float(spread[j]), 4),
                                 vs_dago4=round(float(agree[j]), 4))
        res = dict(
            n_gen=int(P.shape[1]), cond=round(cond, 1),
            neg_reported=int(np.sum(u[rep] < 0)),
            osc_median=round(float(np.nanmedian(np.abs(d2))), 4),
            toy_spread_median=round(float(np.nanmedian(spread[rep])), 4),
            toy_spread_p90=round(float(np.nanpercentile(spread[rep], 90)), 4),
            vs_dago4_median=round(float(np.nanmedian(agree[rep])), 4),
            tail_bins=tail,
        )
        out[name] = res
        print(f"[{name}] cond={res['cond']} neg={res['neg_reported']} "
              f"osc={res['osc_median']} spread={res['toy_spread_median']*100:.2f}% "
              f"(p90 {res['toy_spread_p90']*100:.1f}%) vs_dago4={res['vs_dago4_median']*100:.1f}%")
        for k, v in tail.items():
            print(f"    {k}: pur={v['purity']} stab={v['stability']} eff={v['eff']} "
                  f"u/gen={v['u_over_gen']} spread={v['toy_spread']*100:.1f}% dago-diff={v['vs_dago4']*100:.1f}%")

    with open(f"{OUT}/metrics_{mode}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", f"{OUT}/metrics_{mode}.json")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "groomed")
