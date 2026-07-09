"""What happens if reco is rebinned to the SAME edges as gen before unfolding?

Square (NxN per pT slice) response vs the production convention (finer reco,
merged sink). Adopted coarse gen binning. Metrics on published bins
(pt slices 1-5, lowest rho bin = hidden buffer).

Run: .venv/bin/python -m scripts.dijet_methods.study2_square_reco
"""
from __future__ import annotations

import json
import os

import numpy as np

from scripts.dijet_methods.loader import Binning
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.result import prepare

OUT = "outputs/dijet/2018/rho/method_study2/square_reco"
os.makedirs(OUT, exist_ok=True)

PT = np.array([0, 200, 290, 400, 570, 760, 13000.0])
NATIVE_RECO = np.array([-10, -8, -7, -6, -5.5, -5, -4.75, -4.5, -4.25, -4,
                        -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75,
                        -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0])
GEN = {
    "groomed": np.array([-10., -5., -3., -2.5, -2., -1.5, -1., -0.5, 0.]),
    "ungroomed": np.array([-10., -2., -1.5, -1., -0.5, 0.]),
}
N_TOYS = 100
RNG = np.random.default_rng(11)


def fine_reco(gen_edges):
    keep = NATIVE_RECO[NATIVE_RECO >= gen_edges[1] - 1e-9]
    return np.concatenate([[-10.0], keep]) if keep[0] > -10 else keep


def weighted_lstsq(prob, matched, var):
    v = np.maximum(np.asarray(var, float), np.maximum(matched, 1.0))
    w = 1.0 / np.sqrt(v)
    x, *_ = np.linalg.lstsq(prob.P * w[:, None], matched * w, rcond=None)
    return x


def run(mode):
    gen = GEN[mode]
    out = {}
    for name, redges in [("fine_reco", fine_reco(gen)), ("square_reco", gen.copy())]:
        prep = prepare(mode, Binning(pt_edges=PT, rho_edges_reco=redges,
                                     rho_edges_gen=gen))
        prob = prep.prob
        pub = (prob.gen_pt_idx >= 1) & (prob.gen_rho_idx >= 1)
        var = prep.data_var
        s = np.linalg.svd(prob.P, compute_uv=False)
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
        d2 = []
        for pt in range(1, 6):
            idx = np.where((prob.gen_pt_idx == pt) & (prob.gen_rho_idx >= 1))[0]
            r = rr[idx]
            d2.extend((r[2:] - 2 * r[1:-1] + r[:-2]).tolist())
        # normalized-shape ratio per published bin (like the result plots)
        shape = {}
        for pt in range(1, 6):
            gsl = (prob.gen_pt_idx == pt)
            with np.errstate(divide="ignore", invalid="ignore"):
                sr = (u[gsl] / u[gsl].sum()) / (prob.gen[gsl] / prob.gen[gsl].sum())
            shape[f"pt{pt}"] = [round(float(x), 3) for x in sr[1:]]
        out[name] = dict(
            n_reco=int(prob.P.shape[0]), n_gen=int(prob.P.shape[1]),
            cond=round(cond, 1),
            neg_published=int(np.sum(u[pub] < 0)),
            osc_median=round(float(np.nanmedian(np.abs(d2))), 4),
            toy_spread_median=round(float(np.nanmedian(spread[pub])), 4),
            toy_spread_p90=round(float(np.nanpercentile(spread[pub], 90)), 4),
            toy_spread_max=round(float(np.nanmax(spread[pub])), 4),
            vs_dago4_median=round(float(np.nanmedian(agree[pub])), 4),
            vs_dago4_p90=round(float(np.nanpercentile(agree[pub], 90)), 4),
            shape_ratio_u_over_prior=shape,
        )
        r = out[name]
        print(f"[{mode}/{name}] reco x gen = {r['n_reco']}x{r['n_gen']} cond={r['cond']} "
              f"neg={r['neg_published']} osc={r['osc_median']} "
              f"spread={r['toy_spread_median']*100:.2f}% (p90 {r['toy_spread_p90']*100:.1f}%, "
              f"max {r['toy_spread_max']*100:.1f}%) vs_dago4={r['vs_dago4_median']*100:.1f}% "
              f"(p90 {r['vs_dago4_p90']*100:.1f}%)")
    with open(f"{OUT}/metrics_{mode}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    run("groomed")
    run("ungroomed")
