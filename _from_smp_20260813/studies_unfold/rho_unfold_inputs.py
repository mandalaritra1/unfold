#!/usr/bin/env python
"""TUnfold inputs for the 12-bin groomed-rho proposal, straight from the skims.

The proposed edges are NOT a subset of the production rho axis (-3.4, -2.85,
-1.8, -1.3, -1.1, -0.9, -0.75, -0.65, -0.55 are all absent from it), so the
production histograms cannot be merged onto this binning -- the response has to
be rebuilt from the skim, which is exactly what the skim exists for.

Written to be read by scripts/rho_unfold_stability.py (ROOT/TUnfold lives in a different
venv, hence the npz hand-off). Everything follows the native-binning runner's
conventions:

  * fakes/misses by SUBTRACTION using in-range sums only
        fakes_i  = reco_i - sum_{j in-range} A_ij
        misses_j = gen_j  - sum_{i in-range} A_ij
  * a rho BUFFER bin [-10, -4] on both gen and reco (zjet convention): jets
    that migrate in from below the reported floor are then represented in the
    response instead of being subtracted as fakes. Never reported.
  * data covariance from the two-jet event clustering, V_ij = sum_e w_e^2 n_ei
    n_ej -- the skim stores both jets of every event with jetidx 0/1, so the
    pairing is recoverable exactly.

    python scripts/rho_unfold_inputs.py --out unfold_inputs.npz
"""
from __future__ import annotations

import argparse
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import smp_jetmass_run2.corrections as corr
from scripts.studies.unfold.rho_width_scan import PROPOSED_RHO, PT_EDGES, idx, rho, xs_weight


def flat(i, j, nr):
    """(pt, rho) -> flat index, pt-major (the native runner's reshape order)."""
    return i * nr + j


def mc_pass(paths, ptedges, redges, year, mfloor):
    """Response + gen/reco table occupancies, weighted and sumw2."""
    npt, nr = len(ptedges) - 1, len(redges) - 1
    N = npt * nr
    A = np.zeros((N, N))
    A_w2 = np.zeros((N, N))
    gen = np.zeros(N)
    gen_w2 = np.zeros(N)
    reco = np.zeros(N)
    reco_w2 = np.zeros(N)

    for p in sorted(paths):
        sk = pickle.load(open(p, "rb"))["skim"]
        wmap = {}
        for ds in sk.get("datasets", []):
            w = xs_weight(ds, year)
            if w is None:
                print(f"  !! no xs/sumw for {ds} -- SKIPPED")
            wmap[zlib.crc32(ds.encode()) & 0xFFFFFFFF] = (w or 0.0)

        def cols(tab, names):
            t = sk[tab]
            if not len(t[names[0]].value):
                return None
            out = {c: np.asarray(t[c].value) for c in names + ["weight", "dataset_id"]}
            lut = {d: wmap.get(d, 0.0) for d in np.unique(out["dataset_id"])}
            out["w"] = (out["weight"] * np.array([lut[d] for d in out["dataset_id"]])
                        * corr.lumi[year] * 1000.0)
            return out

        m = cols("matched", ["ptreco", "mreco_g", "ptgen", "mgen_g"])
        if m is not None:
            ipg, ipr = idx(m["ptgen"], ptedges), idx(m["ptreco"], ptedges)
            irg = idx(rho(m["mgen_g"], m["ptgen"]), redges)
            irr = idx(rho(m["mreco_g"], m["ptreco"]), redges)
            okg = (ipg >= 0) & (irg >= 0) & (m["mgen_g"] > mfloor)
            okr = (ipr >= 0) & (irr >= 0) & (m["mreco_g"] > mfloor)
            b = okg & okr
            fi, fj = flat(ipr[b], irr[b], nr), flat(ipg[b], irg[b], nr)
            np.add.at(A, (fi, fj), m["w"][b])
            np.add.at(A_w2, (fi, fj), m["w"][b] ** 2)

        g = cols("gen", ["ptgen", "mgen_g"])
        if g is not None:
            i, j = idx(g["ptgen"], ptedges), idx(rho(g["mgen_g"], g["ptgen"]), redges)
            ok = (i >= 0) & (j >= 0) & (g["mgen_g"] > mfloor)
            f = flat(i[ok], j[ok], nr)
            np.add.at(gen, f, g["w"][ok])
            np.add.at(gen_w2, f, g["w"][ok] ** 2)

        r = cols("reco", ["ptreco", "mreco_g"])
        if r is not None:
            i, j = idx(r["ptreco"], ptedges), idx(rho(r["mreco_g"], r["ptreco"]), redges)
            ok = (i >= 0) & (j >= 0) & (r["mreco_g"] > mfloor)
            f = flat(i[ok], j[ok], nr)
            np.add.at(reco, f, r["w"][ok])
            np.add.at(reco_w2, f, r["w"][ok] ** 2)
        print(f"  {Path(p).name:<52s} done")
    return A, A_w2, gen, gen_w2, reco, reco_w2


def data_pass(paths, ptedges, redges, mfloor):
    """Data reco spectrum, sumw2, and the event-clustered two-jet covariance."""
    npt, nr = len(ptedges) - 1, len(redges) - 1
    N = npt * nr
    reco = np.zeros(N)
    reco_w2 = np.zeros(N)
    V = np.zeros((N, N))

    for p in sorted(paths):
        sk = pickle.load(open(p, "rb"))["skim"]["reco"]
        pt = np.asarray(sk["ptreco"].value)
        if not len(pt):
            continue
        mg = np.asarray(sk["mreco_g"].value)
        w = np.asarray(sk["weight"].value)
        ji = np.asarray(sk["jetidx"].value).astype(int)
        #### An event's jets are stored back to back with strictly increasing
        #### jetidx, so a new event starts wherever jetidx fails to increase.
        #### Dijet writes 0,1,0,1,... -> pairs. Trijet measures ONE jet per
        #### event (the third-leading, jetidx == 2 always) -> every row is its
        #### own event and V comes out exactly diagonal, which is correct.
        #### The old `ji == 0` test silently put every trijet jet in ONE event
        #### and turned V into a single giant outer product.
        ev = np.cumsum(np.r_[True, ji[1:] <= ji[:-1]]) - 1
        i, j = idx(pt, ptedges), idx(rho(mg, pt), redges)
        ok = (i >= 0) & (j >= 0) & (mg > mfloor)
        f = flat(i[ok], j[ok], nr)
        np.add.at(reco, f, w[ok])
        np.add.at(reco_w2, f, w[ok] ** 2)

        #### V_ij = sum_e w_e^2 n_ei n_ej. Only the surviving jets count, and
        #### the weight is per event (both jets carry the same prescale), so
        #### build the per-event occupancy sparsely: at most 2 filled bins.
        e, b, ww = ev[ok], f, w[ok]
        order = np.argsort(e, kind="stable")
        e, b, ww = e[order], b[order], ww[order]
        starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
        ends = np.r_[starts[1:], len(e)]
        for s, t in zip(starts, ends):
            bb, w2 = b[s:t], ww[s] ** 2
            if len(bb) == 1:
                V[bb[0], bb[0]] += w2
            else:
                for x in bb:
                    for y in bb:
                        V[x, y] += w2
        print(f"  {Path(p).name:<52s} {ok.sum():>9d} jets, {len(starts)} events")
    return reco, reco_w2, V


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    ap.add_argument("--data", default="/Users/aritra/Downloads/dijet_data_skims")
    ap.add_argument("--year", default="2018")
    ap.add_argument("--mass-floor", type=float, default=2.0)
    ap.add_argument("--rho-edges", nargs="+", type=float, default=None)
    ap.add_argument("--buffer", type=float, default=-10.0,
                    help="prepend a rho buffer bin [buffer, first edge]; "
                         "pass nan to disable")
    ap.add_argument("--pt-edges", nargs="+", type=float, default=None,
                        help="override PT_EDGES. The defaults are the DIJET edges; the "
                             "trijet third jet is far softer and must not reuse them")
    ap.add_argument("--out", default="unfold_inputs.npz")
    args = ap.parse_args(argv)

    if args.pt_edges:
        globals()["PT_EDGES"] = np.array(args.pt_edges, dtype=float)

    prop = np.array(args.rho_edges if args.rho_edges else PROPOSED_RHO)
    redges = prop if np.isnan(args.buffer) else np.r_[args.buffer, prop]
    n_buf = 0 if np.isnan(args.buffer) else 1
    print(f"pt edges {list(PT_EDGES)}")
    print(f"rho edges {list(redges)}  ({n_buf} buffer bin)")

    print("MC:")
    A, A_w2, gen, gen_w2, reco, reco_w2 = mc_pass(
        Path(args.mc).glob("*.pkl"), PT_EDGES, redges, args.year, args.mass_floor)
    print("data:")
    d_reco, d_w2, d_V = data_pass(Path(args.data).glob("*.pkl"), PT_EDGES,
                                  redges, args.mass_floor)

    np.savez_compressed(args.out, A=A, A_w2=A_w2, gen=gen, gen_w2=gen_w2,
                        reco=reco, reco_w2=reco_w2, data_reco=d_reco,
                        data_w2=d_w2, data_V=d_V, pt_edges=PT_EDGES,
                        rho_edges=redges, n_buffer=n_buf)
    fakes = reco - A.sum(axis=1)
    misses = gen - A.sum(axis=0)
    print(f"\nMC reco {reco.sum():.4g}, gen {gen.sum():.4g}, matched "
          f"{A.sum():.4g}")
    print(f"fakes {fakes.sum() / reco.sum():.2%} of reco, "
          f"misses {misses.sum() / gen.sum():.2%} of in-range gen")
    print(f"data {d_reco.sum():.4g} ({(d_reco > 0).sum()} filled bins); "
          f"cov diag/sumw2 median "
          f"{np.median(np.diag(d_V)[d_w2 > 0] / d_w2[d_w2 > 0]):.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
