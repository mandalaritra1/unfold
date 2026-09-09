#!/usr/bin/env python
"""Minimum resolvable groomed-rho bin width, per pT bin, from the rho skim.

The question this answers is "how fine can the rho binning be before purity and
stability fall below 0.5", which cannot be asked of the production histograms --
they can only ever be MERGED, and the answer needs edges the production binning
does not have.

Strategy: accumulate ONE fine 4D migration matrix

    (ptgen, rhogen, ptreco, rhoreco)   rho on a FINE grid of width `--fine`

plus the gen-table and reco-table occupancies on the same fine grid. Every
candidate binning whose edges are a union of fine edges is then an exact
`reduceat` away, so a width scan costs one pass over the files instead of one
pass per width.

    python scripts/rho_width_scan.py --mass-floor 2.0

Widths are scanned in units of the fine grid; anything not a multiple of it is
rejected rather than silently rounded.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import smp_jetmass_run2.corrections as corr

JET_R = 0.8

#### The sink bin [185,200] is dropped (it is a sink, nothing is quoted from it)
#### and everything above 570 is merged into the last bin, which is where the
#### purity collapse lives -- see the width scan for why 570-820 cannot be kept
#### as separate bins at any rho granularity.
PT_EDGES = np.array([200., 290., 400., 480., 570., 13000.])

#### The rho binning this scan converged on. Ten of the twelve bins clear 0.5
#### rho-only purity in all five pT bins at once; the two tail bins sit at
#### 0.39/0.43 under the relaxed rule (a couple of bins may go lower if they buy
#### shape information) and are the only two narrower than 2x the rho resolution.
PROPOSED_RHO = [-4.0, -3.4, -2.85, -2.25, -1.8, -1.5, -1.3, -1.1, -0.9,
                -0.75, -0.65, -0.55, 0.0]


def rho(m, pt):
    """rho = 2*log10(m/(pt*R)) -- the repo definition, never 2*log10(m/pt)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * JET_R))


def idx(v, edges):
    """Bin index, or -1 when out of range (under/overflow both -> -1)."""
    i = np.digitize(v, edges) - 1
    return np.where(np.isfinite(v) & (i >= 0) & (i < len(edges) - 1), i, -1)


def xs_weight(dataset, year):
    """xs/sumw for a DAS path; raw counts across HT bins are not physical."""
    key = dataset.split("/")[1] if dataset.startswith("/") else dataset
    sumw = corr.sumw_qcd_mg[year]
    if key not in sumw or key not in corr.xsdb:
        return None
    return corr.xsdb[key] / sumw[key]


def accumulate(paths, ptedges, fedges, year, mfloor):
    """One pass -> fine 4D response + fine gen/reco table occupancies."""
    npt, nf = len(ptedges) - 1, len(fedges) - 1
    R4 = np.zeros((npt, nf, npt, nf))          # ptgen, rhogen, ptreco, rhoreco
    Gt = np.zeros((npt, nf))                   # gen table  (misses included)
    Rt = np.zeros((npt, nf))                   # reco table (fakes included)
    Mg = np.zeros((npt, nf))                   # matched, binned at gen
    Mr = np.zeros((npt, nf))                   # matched, binned at reco
    N4 = np.zeros((npt, nf, npt, nf))          # raw counts, for stat sanity

    for p in paths:
        sk = pickle.load(open(p, "rb"))["skim"]
        wmap = {}
        for ds in sk.get("datasets", []):
            w = xs_weight(ds, year)
            if w is None:
                print(f"  !! no xs/sumw for {ds} -- SKIPPED")
            wmap[zlib.crc32(ds.encode()) & 0xFFFFFFFF] = w or 0.0

        def cols(tab, names):
            t = sk[tab]
            if not len(t[names[0]].value):
                return None
            out = {c: np.asarray(t[c].value) for c in names + ["weight", "dataset_id"]}
            lut = {d: wmap.get(d, 0.0) for d in np.unique(out["dataset_id"])}
            out["w"] = out["weight"] * np.array([lut[d] for d in out["dataset_id"]])
            return out

        m = cols("matched", ["ptreco", "mreco_g", "ptgen", "mgen_g"])
        if m is not None:
            ipg = idx(m["ptgen"], ptedges)
            ipr = idx(m["ptreco"], ptedges)
            irg = idx(rho(m["mgen_g"], m["ptgen"]), fedges)
            irr = idx(rho(m["mreco_g"], m["ptreco"]), fedges)
            okg = (ipg >= 0) & (irg >= 0) & (m["mgen_g"] > mfloor)
            okr = (ipr >= 0) & (irr >= 0) & (m["mreco_g"] > mfloor)
            both = okg & okr
            np.add.at(R4, (ipg[both], irg[both], ipr[both], irr[both]), m["w"][both])
            np.add.at(N4, (ipg[both], irg[both], ipr[both], irr[both]), 1.0)
            np.add.at(Mg, (ipg[okg], irg[okg]), m["w"][okg])
            np.add.at(Mr, (ipr[okr], irr[okr]), m["w"][okr])

        g = cols("gen", ["ptgen", "mgen_g"])
        if g is not None:
            i, j = idx(g["ptgen"], ptedges), idx(rho(g["mgen_g"], g["ptgen"]), fedges)
            ok = (i >= 0) & (j >= 0) & (g["mgen_g"] > mfloor)
            np.add.at(Gt, (i[ok], j[ok]), g["w"][ok])

        r = cols("reco", ["ptreco", "mreco_g"])
        if r is not None:
            i, j = idx(r["ptreco"], ptedges), idx(rho(r["mreco_g"], r["ptreco"]), fedges)
            ok = (i >= 0) & (j >= 0) & (r["mreco_g"] > mfloor)
            np.add.at(Rt, (i[ok], j[ok]), r["w"][ok])

        print(f"  {Path(p).name:<50s} done")
    return R4, Gt, Rt, Mg, Mr, N4


def merge(a, k, axes):
    """Sum groups of k adjacent fine bins along each of `axes`."""
    for ax in axes:
        n = a.shape[ax] // k
        sl = [slice(None)] * a.ndim
        sl[ax] = slice(0, n * k)
        a = a[tuple(sl)]
        shape = list(a.shape)
        shape[ax:ax + 1] = [n, k]
        a = a.reshape(shape).sum(axis=ax + 1)
    return a


def evaluate(R4, Gt, Rt, Mg, Mr, N4, k):
    """Purity/stability/fake/miss for a rho binning `k` fine bins wide."""
    Rk = merge(R4, k, (1, 3))
    diag = np.einsum("gigi->gi", Rk)                     # same pt bin AND same rho bin
    return dict(
        both=diag,
        m_gen=merge(Mg, k, (1,)), m_reco=merge(Mr, k, (1,)),
        gen=merge(Gt, k, (1,)), reco=merge(Rt, k, (1,)),
        n=np.einsum("gigi->gi", merge(N4, k, (1, 3))),
    )


def regroup(a, edges, fedges, axes):
    """Sum the fine grid onto arbitrary `edges` (which must be fine-grid edges)."""
    i = [int(np.argmin(np.abs(fedges - e))) for e in edges]
    for ax in axes:
        a = np.add.reduceat(a.take(range(i[0], i[-1]), axis=ax),
                            [k - i[0] for k in i[:-1]], axis=ax)
    return a


def pt_regroup(a, groups, axes):
    """Sum pT bins into `groups` (a list of index lists) along each of `axes`."""
    for ax in axes:
        a = np.stack([a.take(g, axis=ax).sum(axis=ax) for g in groups], axis=ax)
    return a


def pt_merge_scan(R4, fedges, rho_edges, groups_list, ptedges):
    """How much of the joint (pT, rho) impurity is pT migration we can bin away?

    rho-only purity is untouched by pT merging -- it is set by the rho
    resolution -- so the only thing a merge can buy is the pT-bin purity, and
    with it the joint purity that actually enters a 2-D unfold. Every candidate
    is one reduction on the SAME cached fine response, so this is free.
    """
    Rr = regroup(R4, rho_edges, fedges, (1, 3))
    out = []
    for name, groups in groups_list:
        Rk = pt_regroup(Rr, groups, (0, 2))
        den = Rk.sum(axis=(0, 1))
        joint = np.einsum("gigi->gi", Rk) / np.maximum(den, 1e-30)
        rho_only = np.einsum("gjij->ij", Rk) / np.maximum(den, 1e-30)
        P = Rk.sum(axis=(1, 3))                       # [pt_gen, pt_reco]
        pt_pur = np.diag(P) / np.maximum(P.sum(axis=0), 1e-30)
        labels = [f"{ptedges[g[0]]:.0f}-"
                  f"{'inf' if ptedges[g[-1] + 1] > 5000 else f'{ptedges[g[-1] + 1]:.0f}'}"
                  for g in groups]
        out.append(dict(name=name, labels=labels, pt_purity=pt_pur.tolist(),
                        joint_median=float(np.median(joint)),
                        joint_min=float(joint.min()),
                        rho_only_median=float(np.median(rho_only))))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skims", default="/Users/aritra/cernbox (2)/dijet_skims")
    p.add_argument("--year", default="2018")
    p.add_argument("--fine", type=float, default=0.05,
                   help="fine grid width; every scanned width must be a multiple")
    p.add_argument("--lo", type=float, default=-10.0)
    p.add_argument("--hi", type=float, default=0.0)
    p.add_argument("--widths", nargs="+", type=float,
                   default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0])
    p.add_argument("--mass-floor", type=float, default=2.0)
    p.add_argument("--min-frac", type=float, default=0.005,
                   help="ignore bins holding less than this of their pT slice")
    p.add_argument("--cache", default=None, help="npz to write/reuse the fine pass")
    p.add_argument("--pt-merge", nargs="+", default=None, metavar="GROUPS",
                   help="evaluate pT-bin merges on --rho-edges instead of "
                        "scanning widths, e.g. '0,1,2,3,4' '0,1,2+3,4'")
    p.add_argument("--rho-edges", nargs="+", type=float, default=None,
                   help="rho binning for --pt-merge (default: the proposal)")
    p.add_argument("--pt-edges", nargs="+", type=float, default=None,
                   help="override PT_EDGES. The defaults are the DIJET edges; "
                        "the trijet third jet is far softer (median 233 GeV, "
                        "99th pct 522) so they must not be reused there")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    if args.pt_edges:
        globals()["PT_EDGES"] = np.array(args.pt_edges, dtype=float)

    nf = int(round((args.hi - args.lo) / args.fine))
    fedges = np.linspace(args.lo, args.hi, nf + 1)
    bad = [w for w in args.widths
           if abs(w / args.fine - round(w / args.fine)) > 1e-9]
    if bad:
        raise SystemExit(f"widths {bad} are not multiples of --fine {args.fine}")

    if args.cache and Path(args.cache).exists():
        z = np.load(args.cache)
        R4, Gt, Rt, Mg, Mr, N4 = (z[k] for k in ("R4", "Gt", "Rt", "Mg", "Mr", "N4"))
        print(f"reusing fine pass from {args.cache}")
    else:
        paths = sorted(Path(args.skims).glob("*.pkl"))
        print(f"GROOMED rho, {len(paths)} files, fine grid {args.fine} "
              f"({nf} bins over [{args.lo},{args.hi}]), m_g > {args.mass_floor}")
        print(f"pt edges: {list(PT_EDGES)}\n")
        R4, Gt, Rt, Mg, Mr, N4 = accumulate(paths, PT_EDGES, fedges,
                                            args.year, args.mass_floor)
        if args.cache:
            np.savez_compressed(args.cache, R4=R4, Gt=Gt, Rt=Rt, Mg=Mg, Mr=Mr, N4=N4)
            print(f"cached fine pass -> {args.cache}")

    if args.pt_merge:
        rho_edges = np.array(args.rho_edges if args.rho_edges else PROPOSED_RHO)
        groups_list = []
        for spec in args.pt_merge:
            groups = [[int(x) for x in g.split("+")] for g in spec.split(",")]
            groups_list.append((spec, groups))
        res = pt_merge_scan(R4, fedges, rho_edges, groups_list, PT_EDGES)
        print(f"\n{'='*78}\npT-bin merges on rho edges "
              f"{[round(e, 2) for e in rho_edges]}\n{'='*78}")
        print(f"{'grouping':>16s} {'worst pT pur':>13s} {'median joint':>13s} "
              f"{'min joint':>10s} {'median rho-only':>16s}   per-bin pT purity")
        for r in res:
            print(f"{r['name']:>16s} {min(r['pt_purity']):13.3f} "
                  f"{r['joint_median']:13.3f} {r['joint_min']:10.3f} "
                  f"{r['rho_only_median']:16.3f}   "
                  + "  ".join(f"{l}:{v:.3f}"
                              for l, v in zip(r["labels"], r["pt_purity"])))
        print("\nrho-only purity is essentially invariant under pT merging, as it "
              "must be:\nmerging pT cannot change the rho resolution. The gain is "
              "all in pT purity.")
        if args.json_out:
            json.dump(res, open(args.json_out, "w"))
            print(f"wrote {args.json_out}")
        return 0

    #### summary: for each pT bin and width, the lowest rho above which EVERY
    #### populated bin clears 0.5 in both purity and stability
    print(f"\n{'='*78}\nlowest rho edge above which purity AND stability both "
          f"clear 0.5 everywhere\n{'='*78}")
    hdr = "".join(f"{w:>9.2f}" for w in args.widths)
    print(f"{'pT_gen bin':>16s}{hdr}")
    detail = {}
    for w in args.widths:
        k = int(round(w / args.fine))
        E = evaluate(R4, Gt, Rt, Mg, Mr, N4, k)
        edges = args.lo + args.fine * k * np.arange(E["both"].shape[1] + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pur, sta = E["both"] / E["m_reco"], E["both"] / E["m_gen"]
        detail[w] = (E, edges, pur, sta)

    for i in range(len(PT_EDGES) - 1):
        row = f"{PT_EDGES[i]:6.0f}-{PT_EDGES[i+1]:<9.0f}"
        for w in args.widths:
            E, edges, pur, sta = detail[w]
            tot = E["gen"][i].sum()
            frac = E["gen"][i] / tot if tot else np.zeros_like(E["gen"][i])
            good = ~((frac >= args.min_frac) & np.isfinite(pur[i]) &
                     ((pur[i] < 0.5) | (sta[i] < 0.5)))
            #### walk down from the top: the threshold is where the first
            #### failing bin appears, so everything above it is usable
            thr = None
            for j in range(len(good) - 1, -1, -1):
                if frac[j] < args.min_frac and edges[j] < -6:
                    continue
                if not good[j]:
                    thr = edges[j + 1]
                    break
            row += f"{'  none' if thr is None else f'{thr:9.2f}'}"
        print(row)

    #### per-width detail for the pT bins that carry the yield
    for w in args.widths:
        E, edges, pur, sta = detail[w]
        with np.errstate(divide="ignore", invalid="ignore"):
            fake, miss = 1 - E["m_reco"] / E["reco"], 1 - E["m_gen"] / E["gen"]
        print(f"\n{'='*78}\nrho width {w}\n{'='*78}")
        for i in range(len(PT_EDGES) - 1):
            tot = E["gen"][i].sum()
            print(f"\n  pT_gen {PT_EDGES[i]:.0f}-{PT_EDGES[i+1]:.0f} "
                  f"(yield {tot/E['gen'].sum():.2%})")
            print(f"  {'rho':>15s} {'purity':>8s} {'stab':>8s} {'fake':>8s} "
                  f"{'miss':>8s} {'rawN':>9s} {'yield':>8s}")
            for j in range(len(edges) - 1):
                if E["gen"][i, j] <= 0 or E["gen"][i, j] / tot < args.min_frac:
                    continue
                flag = "  <-- fails" if (pur[i, j] < 0.5 or sta[i, j] < 0.5) else ""
                print(f"  [{edges[j]:6.2f},{edges[j+1]:6.2f}] {pur[i,j]:8.3f} "
                      f"{sta[i,j]:8.3f} {fake[i,j]:8.3f} {miss[i,j]:8.3f} "
                      f"{E['n'][i,j]:9.0f} {E['gen'][i,j]/tot:8.2%}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
