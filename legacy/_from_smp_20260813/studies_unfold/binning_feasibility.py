#!/usr/bin/env python
"""Purity / stability / fake / miss at ANY candidate binning, from the rho skim.

This is what the three-table skim exists for: the production histograms can only
ever be MERGED, so a binning question that needs finer or shifted edges cannot be
answered from them. The flat `matched`/`reco`/`gen` tables can be re-binned
arbitrarily.

    matched -> final_seq   (both gen and reco survive)   -> purity, stability
    reco    -> recoTot_seq (fakes included)              -> fake rate
    gen     -> genTot_seq  (misses included)             -> miss rate

Definitions, all evaluated on the GEN (truth) binning -- reco is mapped into the
truth bin it lands in, which is the quantity that matters for whether a truth bin
can be unfolded:

    purity_i    = N(gen in i AND reco in i) / N(reco in i)      [matched only]
    stability_i = N(gen in i AND reco in i) / N(gen in i)       [matched only]
    fake_i      = 1 - N(matched, reco in i) / N(reco table in i)
    miss_i      = 1 - N(matched, gen  in i) / N(gen  table in i)

Rule of thumb: a bin is unfoldable if purity and stability both clear ~0.5 (a bin
narrower than the resolution drives them down and the unfolded stat error up).

    python scripts/binning_feasibility.py --groomed
    python scripts/binning_feasibility.py --groomed --rho-edges -10 -8 -6 -5 -4 -3 -2 -1 0

Files are streamed one at a time and only the needed columns are kept, so the
21M-row tables never all sit in memory at once.
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
from smp_jetmass_run2.hist_utils import util_binning

JET_R = 0.8


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


def accumulate(paths, ptedges, rhoedges, groomed, year, mfloor=0.0):
    npt, nrho = len(ptedges) - 1, len(rhoedges) - 1
    shape = (npt, nrho)
    A = {k: np.zeros(shape) for k in
         ("both", "m_reco", "m_gen", "reco", "gen", "both_n")}
    mr = "mreco_g" if groomed else "mreco_u"
    mg = "mgen_g" if groomed else "mgen_u"

    for p in paths:
        sk = pickle.load(open(p, "rb"))["skim"]
        wmap = {}
        for ds in sk.get("datasets", []):
            w = xs_weight(ds, year)
            if w is None:
                print(f"  !! no xs/sumw for {ds} -- SKIPPED")
            wmap[zlib.crc32(ds.encode()) & 0xFFFFFFFF] = w

        def cols(tab, names):
            t = sk[tab]
            if not len(t[names[0]].value):
                return None
            out = {c: np.asarray(t[c].value) for c in names + ["weight", "dataset_id"]}
            sf = np.array([wmap.get(d) or 0.0 for d in
                           np.unique(out["dataset_id"])])
            lut = dict(zip(np.unique(out["dataset_id"]), sf))
            out["w"] = out["weight"] * np.array([lut[d] for d in out["dataset_id"]])
            return out

        #### The groomed-mass floor is applied downstream in the unfold repo, not
        #### in the processor, so it has to be re-imposed here or the low-rho
        #### bins look far worse than the analysis actually sees them: a jet at
        #### rho=-4, pt=200 has m_g = 1.6 GeV, which is below any sane floor.
        m = cols("matched", ["ptreco", mr, "ptgen", mg])
        if m is not None:
            ir = (idx(m["ptreco"], ptedges), idx(rho(m[mr], m["ptreco"]), rhoedges))
            ig = (idx(m["ptgen"], ptedges), idx(rho(m[mg], m["ptgen"]), rhoedges))
            okr = (ir[0] >= 0) & (ir[1] >= 0) & (m[mr] > mfloor)
            okg = (ig[0] >= 0) & (ig[1] >= 0) & (m[mg] > mfloor)
            same = okr & okg & (ir[0] == ig[0]) & (ir[1] == ig[1])
            np.add.at(A["m_reco"], (ir[0][okr], ir[1][okr]), m["w"][okr])
            np.add.at(A["m_gen"], (ig[0][okg], ig[1][okg]), m["w"][okg])
            np.add.at(A["both"], (ig[0][same], ig[1][same]), m["w"][same])
            np.add.at(A["both_n"], (ig[0][same], ig[1][same]), 1.0)

        r = cols("reco", ["ptreco", mr])
        if r is not None:
            i = (idx(r["ptreco"], ptedges), idx(rho(r[mr], r["ptreco"]), rhoedges))
            ok = (i[0] >= 0) & (i[1] >= 0) & (r[mr] > mfloor)
            np.add.at(A["reco"], (i[0][ok], i[1][ok]), r["w"][ok])

        g = cols("gen", ["ptgen", mg])
        if g is not None:
            i = (idx(g["ptgen"], ptedges), idx(rho(g[mg], g["ptgen"]), rhoedges))
            ok = (i[0] >= 0) & (i[1] >= 0) & (g[mg] > mfloor)
            np.add.at(A["gen"], (i[0][ok], i[1][ok]), g["w"][ok])

        print(f"  {Path(p).name:<50s} done")
    return A


def report(A, ptedges, rhoedges, minfrac):
    with np.errstate(divide="ignore", invalid="ignore"):
        pur = A["both"] / A["m_reco"]
        sta = A["both"] / A["m_gen"]
        fake = 1 - A["m_reco"] / A["reco"]
        miss = 1 - A["m_gen"] / A["gen"]
    bad = []
    for i in range(len(ptedges) - 1):
        tot = A["gen"][i].sum()
        print(f"\n=== pT_gen {ptedges[i]:.0f}-{ptedges[i+1]:.0f} GeV "
              f"(gen yield fraction {tot/A['gen'].sum():.3%}) ===")
        print(f"{'rho bin':>16s} {'purity':>8s} {'stab':>8s} {'fake':>8s} "
              f"{'miss':>8s} {'rawN':>9s} {'yield':>8s}")
        for j in range(len(rhoedges) - 1):
            if A["gen"][i, j] <= 0 and A["reco"][i, j] <= 0:
                continue
            frac = A["gen"][i, j] / tot if tot else 0
            flag = ""
            if frac >= minfrac and np.isfinite(pur[i, j]):
                if pur[i, j] < 0.5 or sta[i, j] < 0.5:
                    flag = "  <-- below 0.5"
                    bad.append((ptedges[i], rhoedges[j], rhoedges[j + 1],
                                pur[i, j], sta[i, j], frac))
            print(f"[{rhoedges[j]:6.3f},{rhoedges[j+1]:6.3f}] {pur[i,j]:8.3f} "
                  f"{sta[i,j]:8.3f} {fake[i,j]:8.3f} {miss[i,j]:8.3f} "
                  f"{A['both_n'][i,j]:9.0f} {frac:8.2%}{flag}")
    return bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skims", default="/Users/aritra/cernbox (2)/dijet_skims")
    p.add_argument("--groomed", action="store_true",
                   help="use msoftdrop / groomed gen mass (the measurement)")
    p.add_argument("--year", default="2018")
    p.add_argument("--rho-edges", nargs="+", type=float, default=None,
                   help="override the truth rho edges (default: production)")
    p.add_argument("--pt-edges", nargs="+", type=float, default=None)
    p.add_argument("--mass-floor", type=float, default=0.0,
                   help="drop jets with groomed mass <= this (GeV); the unfold "
                        "repo applies m_g > 2, the processor does not")
    p.add_argument("--min-frac", type=float, default=0.005,
                   help="only flag bins holding at least this gen fraction")
    args = p.parse_args(argv)

    b = util_binning(channel="dijet")
    ptedges = np.array(args.pt_edges if args.pt_edges else b.ptgen_axis.edges)
    if args.rho_edges:
        rhoedges = np.array(args.rho_edges)
    else:
        rhoedges = np.array((b.mgen_over_pt_g_axis if args.groomed
                             else b.mgen_over_pt_axis).edges)

    paths = sorted(Path(args.skims).glob("*.pkl"))
    print(f"{'GROOMED' if args.groomed else 'UNGROOMED'} rho, {len(paths)} files")
    print(f"pt  edges: {list(ptedges)}")
    print(f"rho edges: {list(rhoedges)}\n")

    print(f"mass floor: {args.mass_floor} GeV\n")
    A = accumulate(paths, ptedges, rhoedges, args.groomed, args.year,
                   args.mass_floor)
    bad = report(A, ptedges, rhoedges, args.min_frac)

    print(f"\n{'='*70}\n{len(bad)} populated bins (>={args.min_frac:.1%} of their "
          f"pT slice) fall below 0.5 purity or stability")
    for pt, lo, hi, pu, st, fr in bad:
        print(f"  pT>{pt:.0f}  rho[{lo:.3f},{hi:.3f}]  purity={pu:.3f} "
              f"stability={st:.3f}  ({fr:.1%} of slice)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
