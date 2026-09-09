#!/usr/bin/env python
"""Data/MC at RECO level in a candidate groomed-rho binning.

Purity and the gen/reco ratio both come from MC alone, so they can only say the
binning is self-consistent -- not that it describes the data. This is the check
that uses data: the same `reco` table, the same binning, 2018 JetHT against
2018 QCD madgraphMLM+pythia8.

Two normalization facts make this comparison honest per pT bin:

  * data carries a trigger PRESCALE weight that jumps by four orders of
    magnitude across the pT range (10 -> 3.4e5 in 2018). It is a function of pT
    only, so within one pT bin it is effectively a constant and the rho SHAPE is
    untouched -- across pT bins it is not, which is why everything here is
    normalized inside a pT bin and never across.
  * both skims were written with the same rho_skim10 prescale, so that factor
    cancels in any ratio.

The absolute data/MC yield ratio is reported too, but it is the weaker number:
it folds in the trigger-weight bookkeeping and the QCD cross sections, neither
of which this study is trying to validate.

    python scripts/rho_data_mc.py --mass-floor 2.0
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
from scripts.studies.unfold.rho_width_scan import PROPOSED_RHO, PT_EDGES, idx, rho, xs_weight


def read_reco(path, year, is_data, mfloor):
    """(pt, rho, weight) from a skim's `reco` table, m_g floor applied."""
    sk = pickle.load(open(path, "rb"))["skim"]
    t = sk["reco"]
    pt = np.asarray(t["ptreco"].value)
    if not len(pt):
        return None
    mg = np.asarray(t["mreco_g"].value)
    w = np.asarray(t["weight"].value)
    if not is_data:
        #### MC: the skim stores the raw generator weight, so the XS/sumw
        #### normalization that postprocess would have applied is done here
        wmap = {zlib.crc32(d.encode()) & 0xFFFFFFFF: (xs_weight(d, year) or 0.0)
                for d in sk.get("datasets", [])}
        ds = np.asarray(t["dataset_id"].value)
        lut = {i: wmap.get(i, 0.0) for i in np.unique(ds)}
        w = w * np.array([lut[i] for i in ds]) * corr.lumi[year] * 1000.0
    ok = np.isfinite(pt) & (mg > mfloor)
    return pt[ok], rho(mg[ok], pt[ok]), w[ok]


def spectrum(paths, year, is_data, mfloor, rho_edges):
    npt, nr = len(PT_EDGES) - 1, len(rho_edges) - 1
    H = np.zeros((npt, nr))
    N = np.zeros((npt, nr))
    for p in sorted(paths):
        got = read_reco(p, year, is_data, mfloor)
        if got is None:
            print(f"  {Path(p).name:<52s} EMPTY")
            continue
        pt, rr, w = got
        i, j = idx(pt, PT_EDGES), idx(rr, np.asarray(rho_edges))
        ok = (i >= 0) & (j >= 0)
        np.add.at(H, (i[ok], j[ok]), w[ok])
        np.add.at(N, (i[ok], j[ok]), 1.0)
        print(f"  {Path(p).name:<52s} {ok.sum():>9d} jets")
    return H, N


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mc", default="/Users/aritra/cernbox (2)/dijet_skims")
    p.add_argument("--data", default="/Users/aritra/Downloads/dijet_data_skims")
    p.add_argument("--year", default="2018")
    p.add_argument("--mass-floor", type=float, default=2.0)
    p.add_argument("--rho-edges", nargs="+", type=float, default=None)
    p.add_argument("--pt-edges", nargs="+", type=float, default=None,
                      help="override PT_EDGES. The defaults are the DIJET edges; the "
                           "trijet third jet is far softer and must not reuse them")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    if args.pt_edges:
        globals()["PT_EDGES"] = np.array(args.pt_edges, dtype=float)

    E = np.array(args.rho_edges if args.rho_edges else PROPOSED_RHO)
    print("MC:")
    MC, MCn = spectrum(Path(args.mc).glob("*.pkl"), args.year, False,
                       args.mass_floor, E)
    print("data:")
    DA, DAn = spectrum(Path(args.data).glob("*.pkl"), args.year, True,
                       args.mass_floor, E)

    out = {"pt_edges": PT_EDGES.tolist(), "rho_edges": E.tolist(), "bins": []}
    for i in range(len(PT_EDGES) - 1):
        hi = PT_EDGES[i + 1]
        m, d = MC[i], DA[i]
        if m.sum() <= 0 or d.sum() <= 0:
            print(f"\n=== pT {PT_EDGES[i]:.0f}: EMPTY (mc {m.sum():.3g} "
                  f"data {d.sum():.3g})")
            continue
        #### unit-normalize INSIDE the pT bin: this is a shape test, and the
        #### prescale weight is only pT-flat within a bin
        ms, ds = m / m.sum(), d / d.sum()
        print(f"\n=== pT_reco {PT_EDGES[i]:.0f}-"
              f"{'inf' if hi > 5000 else f'{hi:.0f}'} GeV   "
              f"absolute data/MC = {d.sum() / m.sum():.3f}")
        #### bin widths run 0.10 to 0.60 here, so the raw fractions are not
        #### comparable across bins -- quote the density (1/N)dN/drho alongside
        #### them. The ratio is unaffected: the width cancels in data/MC.
        print(f"  {'rho bin':>16s} {'w':>5s} {'MC frac':>9s} {'data frac':>10s} "
              f"{'MC dens':>9s} {'data dens':>10s} {'data/MC':>8s} "
              f"{'rawN data':>10s}")
        for j in range(len(E) - 1):
            r = ds[j] / ms[j] if ms[j] > 0 else np.nan
            w = E[j + 1] - E[j]
            print(f"  [{E[j]:6.2f},{E[j+1]:6.2f}] {w:5.2f} "
                  f"{ms[j]:9.4f} {ds[j]:10.4f} {ms[j]/w:9.4f} {ds[j]/w:10.4f} "
                  f"{r:8.3f} {DAn[i, j]:10.0f}")
            out["bins"].append(dict(ipt=i, irho=j, lo=float(E[j]), hi=float(E[j + 1]),
                                    mc_frac=float(ms[j]), data_frac=float(ds[j]),
                                    mc_dens=float(ms[j] / w), data_dens=float(ds[j] / w),
                                    ratio=float(r), n_data=float(DAn[i, j]),
                                    n_mc=float(MCn[i, j])))
        out.setdefault("abs_ratio", []).append(float(d.sum() / m.sum()))

    if args.json_out:
        json.dump(out, open(args.json_out, "w"))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
