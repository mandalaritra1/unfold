#!/usr/bin/env python
"""TUnfold inputs for the hadronic groomed-rho proposal, from the PRODUCTION
histograms instead of the rho_skim ntuples.

Since 2026-07-25 the adopted binning IS the production rho axis
(`hist_utils._had_rho_gen_g`: 14 gen / 28 reco, containing both the dijet
12-bin and the trijet 7-bin proposals), so the response no longer has to be
rebuilt jet by jet -- every candidate is an exact `reduceat` away and the run
carries FULL statistics instead of the skim's 1-in-10 prescale.

Output schema is byte-compatible with scripts/rho_unfold_inputs.py (the skim
path), so scripts/rho_unfold_stability.py consumes either one unchanged:
A, A_w2, gen, gen_w2, reco, reco_w2, data_reco, data_w2, data_V, pt_edges,
rho_edges, n_buffer.  Flat index is pt-major (i * n_rho + j).

Conventions kept from the skim path / the native runner:
  * in-range only -- the pt underflow ([185,200] sink and everything below) and
    the rho flow bins are EXCLUDED, so jets crossing those boundaries appear as
    fakes/misses by subtraction rather than being folded back into `matched`.
  * a rho buffer bin [-10, -4] on both axes (the production axis' two sink bins
    merged), never reported.
  * data covariance from the production `reco_cov_rho_g/u` (the event-clustered
    two-jet covariance) when the channel has one; trijet measures a single jet
    per event, so its covariance is exactly diagonal and is built from sumw2.

One difference from the skim study, by construction: the skim applied a
groomed-mass floor (m_g > 2 GeV) per jet, which is a pt-dependent diagonal in
(pt, rho) and cannot be reproduced by selecting bins. These inputs have no mass
floor.

    python scripts/rho_unfold_inputs_from_hists.py --channel dijet \
        --mc  ~/Projects/unfold/inputs/dijet/rho/minimal_rho_dijet_mg_pythia8_2018.pkl \
        --data ~/Projects/unfold/inputs/dijet/rho/minimal_rho_dijet_data_2018.pkl \
        --out dijet_prod_inputs.npz
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

HIST_KEYS = {
    "u": dict(reco="ptjet_rhojet_u_reco", gen="ptjet_rhojet_u_gen",
              matrix="response_matrix_rho_u", cov="reco_cov_rho_u"),
    "g": dict(reco="ptjet_rhojet_g_reco", gen="ptjet_rhojet_g_gen",
              matrix="response_matrix_rho_g", cov="reco_cov_rho_g"),
}

#### Adopted binnings (research-notes topics/{dijet,trijet}_rho_binning_*).
#### rho: the buffer bin first; pt: index groups into the production pt axis
#### [185,200,290,400,480,570,680,760,820,13000] -- index 0 (the 185-200 sink)
#### is deliberately dropped, matching the skim study, so migration across
#### 200 GeV is handled as fakes/misses.
ADOPTED = {
    "dijet": dict(
        rho=[-10.0, -4.0, -3.4, -2.85, -2.25, -1.8, -1.5, -1.3, -1.1, -0.9,
             -0.75, -0.65, -0.55, 0.0],
        pt_groups=[[1], [2], [3], [4], [5, 6, 7, 8]],
    ),
    "trijet": dict(
        rho=[-10.0, -4.0, -2.85, -2.25, -1.8, -1.5, -1.1, -0.75, 0.0],
        pt_groups=[[1], [2], [3, 4, 5, 6, 7, 8]],
    ),
}


def _sel(out, key, syst="nominal"):
    """one systematic, summed over datasets -> (values, variances) WITH flow."""
    h = out[key][{"systematic": syst}][{"dataset": sum}]
    return h.values(flow=True), h.variances(flow=True), h


def edge_groups(fine, target):
    """Index groups of `fine` bins making up each `target` bin (exact edges)."""
    fine = np.asarray(fine, dtype=float)
    idx = []
    for e in target:
        m = np.nonzero(np.isclose(fine, e, atol=1e-9))[0]
        if not len(m):
            raise SystemExit(f"edge {e} is not on the production axis {list(fine)}")
        idx.append(int(m[0]))
    return [list(range(idx[k], idx[k + 1])) for k in range(len(idx) - 1)]


def regroup(a, axis, groups):
    return np.stack([np.take(a, g, axis=axis).sum(axis=axis) for g in groups],
                    axis=axis)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel", default="dijet", choices=["dijet", "trijet"])
    ap.add_argument("--mc", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--groom", default="g", choices=["g", "u"])
    ap.add_argument("--mass-floor-hists", action="store_true",
                    help="use the dedicated full-stat groomed m_g > 2 GeV "
                         "histograms (groomed only)")
    ap.add_argument("--rho-edges", nargs="+", type=float, default=None,
                    help="override the adopted rho edges (buffer bin first)")
    ap.add_argument("--syst", default="nominal",
                    help="MC systematic slice (nominal, JER/JMS/JMRUp/Down). "
                         "Data and its covariance are always nominal -- a jet "
                         "variation is a property of the response, not of the "
                         "measurement.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    spec = ADOPTED[args.channel]
    redges = np.array(args.rho_edges if args.rho_edges else spec["rho"], float)
    ptg = spec["pt_groups"]
    k = HIST_KEYS[args.groom].copy()
    if args.mass_floor_hists:
        if args.groom != "g":
            ap.error("--mass-floor-hists is available only for --groom g")
        k = {name: hist_name + "_mfloor2"
             for name, hist_name in k.items()}

    mc = pickle.load(open(args.mc, "rb"))
    reco_v, reco_w2, h_reco = _sel(mc, k["reco"], args.syst)
    gen_v, gen_w2, h_gen = _sel(mc, k["gen"], args.syst)
    mat_v, mat_w2, _ = _sel(mc, k["matrix"], args.syst)

    pt_fine = np.asarray(h_reco.axes["ptreco"].edges)
    rr_fine = np.asarray(h_reco.axes["mpt_reco"].edges)
    rg_fine = np.asarray(h_gen.axes["mpt_gen"].edges)
    gr, gg = edge_groups(rr_fine, redges), edge_groups(rg_fine, redges)
    pt_edges = np.array([pt_fine[g[0]] for g in ptg] + [pt_fine[ptg[-1][-1] + 1]])
    print(f"pt edges  {list(pt_edges)}   (groups {ptg})")
    print(f"rho edges {list(redges)}  (1 buffer bin)")

    s = slice(1, -1)                       # drop flow on every numeric axis

    def rb_1d(v):                          # (pt, rho) -> (n_pt, n_rho) flat
        x = regroup(regroup(v[s, s], 0, ptg), 1, gr)
        return x.reshape(-1)

    reco = rb_1d(reco_v)
    reco_wsq = rb_1d(reco_w2)
    gen = regroup(regroup(gen_v[s, s], 0, ptg), 1, gg).reshape(-1)
    gen_wsq = regroup(regroup(gen_w2[s, s], 0, ptg), 1, gg).reshape(-1)

    #### (ptreco, ptgen, rho_reco, rho_gen) -> (ptreco, rho_reco, ptgen, rho_gen)
    def rb_mat(m):
        x = np.transpose(m[s, s, s, s], (0, 2, 1, 3))
        x = regroup(regroup(regroup(regroup(x, 0, ptg), 1, gr), 2, ptg), 3, gg)
        n_r = x.shape[0] * x.shape[1]
        n_g = x.shape[2] * x.shape[3]
        return x.reshape(n_r, n_g)

    A, A_w2 = rb_mat(mat_v), rb_mat(mat_w2)

    if args.data is not None:
        dat = pickle.load(open(args.data, "rb"))
        d_v, d_w2, _ = _sel(dat, k["reco"])
        d_reco, d_wsq = rb_1d(d_v), rb_1d(d_w2)
        if k["cov"] in dat:
            cov_v, _, _ = _sel(dat, k["cov"])
            c = cov_v[s, s, s, s]
            c = regroup(regroup(regroup(regroup(c, 0, ptg), 1, gr), 2, ptg), 3, gr)
            n = c.shape[0] * c.shape[1]
            d_V = c.reshape(n, n)
            src = "reco_cov (event-clustered)"
        else:
            d_V = np.diag(d_wsq)
            src = "diagonal sumw2 (one measured jet per event)"
    else:
        d_reco = d_wsq = np.zeros_like(reco)
        d_V = np.diag(d_wsq)
        src = "no data file"

    np.savez_compressed(args.out, A=A, A_w2=A_w2, gen=gen, gen_w2=gen_wsq,
                        reco=reco, reco_w2=reco_wsq, data_reco=d_reco,
                        data_w2=d_wsq, data_V=d_V, pt_edges=pt_edges,
                        rho_edges=redges, n_buffer=1)

    fakes = reco - A.sum(axis=1)
    misses = gen - A.sum(axis=0)
    print(f"\nMC reco {reco.sum():.5g}, gen {gen.sum():.5g}, matched {A.sum():.5g}")
    print(f"fakes {fakes.sum() / reco.sum():.2%} of reco, "
          f"misses {misses.sum() / gen.sum():.2%} of in-range gen")
    print(f"data {d_reco.sum():.5g} ({int((d_reco > 0).sum())} filled bins), V from {src}")
    if args.data is not None and (d_wsq > 0).any():
        m = d_wsq > 0
        print(f"     cov diag / sumw2 median "
              f"{np.median(np.diag(d_V)[m] / d_wsq[m]):.3f}")
    print(f"wrote {args.out}  (reco {A.shape[0]}, gen {A.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
