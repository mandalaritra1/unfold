#!/usr/bin/env python
"""Gen<->reco matching quality per (pT, rho) bin, from the rho skim.

Purity/stability say how much of a bin's content came from that bin. They do NOT
say WHY: a bin can be impure because the observable is badly resolved, or because
the gen<->reco association itself is loose. This separates the two, per bin of a
candidate binning:

    dR(gen, reco)        the association quality -- the matched_gen cut is dR<0.4,
                         so a distribution pushing against 0.4 means the pairing
                         is marginal, not that the mass is mismeasured
    d(rho) = rho_reco - rho_gen    the observable resolution. Compared against the
                         bin WIDTH: a bin narrower than ~2x the rho resolution
                         cannot be unfolded stably no matter what purity says
    pt_reco / pt_gen     the pT response, which is what drives the joint
                         (pT, rho) purity through pT-bin migration

Quantiles come from per-bin histograms rather than moments, because the dR and
d(rho) tails are long enough that a plain RMS is dominated by them.

    python scripts/rho_matching_quality.py --mass-floor 2.0
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
PT_EDGES = np.array([200., 290., 400., 480., 570., 13000.])

#### Default is the binning the width scan converged on under Aritra's relaxed
#### criterion (most bins >= 0.5 rho-only purity, a couple allowed near 0.4 to
#### keep shape information in the tail) -- see scripts/rho_width_scan.py.
RHO_EDGES = np.array([-4.0, -3.4, -2.85, -2.25, -1.8, -1.5, -1.3, -1.1, -0.9,
                      -0.75, -0.65, -0.55, 0.0])

#### per-bin histogram grids for the robust quantiles
DR_EDGES = np.linspace(0.0, 0.4, 81)
DRHO_EDGES = np.linspace(-2.0, 2.0, 201)
PTR_EDGES = np.linspace(0.4, 1.6, 241)


def rho(m, pt):
    """rho = 2*log10(m/(pt*R)) -- the repo definition, never 2*log10(m/pt)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(m / (pt * JET_R))


def idx(v, edges):
    i = np.digitize(v, edges) - 1
    return np.where(np.isfinite(v) & (i >= 0) & (i < len(edges) - 1), i, -1)


def xs_weight(dataset, year):
    key = dataset.split("/")[1] if dataset.startswith("/") else dataset
    sumw = corr.sumw_qcd_mg[year]
    if key not in sumw or key not in corr.xsdb:
        return None
    return corr.xsdb[key] / sumw[key]


def quantiles(counts, edges, qs=(0.16, 0.5, 0.84, 0.95)):
    """Quantiles of a weighted histogram, linearly interpolated within a bin."""
    tot = counts.sum()
    if tot <= 0:
        return [np.nan] * len(qs)
    c = np.concatenate([[0.0], np.cumsum(counts)]) / tot
    return [float(np.interp(q, c, edges)) for q in qs]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skims", default="/Users/aritra/cernbox (2)/dijet_skims")
    p.add_argument("--year", default="2018")
    p.add_argument("--mass-floor", type=float, default=2.0)
    p.add_argument("--rho-edges", nargs="+", type=float, default=None,
                   help="override the default rho binning")
    p.add_argument("--pt-edges", nargs="+", type=float, default=None,
                      help="override PT_EDGES. The defaults are the DIJET edges; the "
                           "trijet third jet is far softer and must not reuse them")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    if args.pt_edges:
        globals()["PT_EDGES"] = np.array(args.pt_edges, dtype=float)

    global RHO_EDGES
    if args.rho_edges:
        RHO_EDGES = np.array(args.rho_edges)

    npt, nrho = len(PT_EDGES) - 1, len(RHO_EDGES) - 1
    H = {
        "dr":   np.zeros((npt, nrho, len(DR_EDGES) - 1)),
        "drho": np.zeros((npt, nrho, len(DRHO_EDGES) - 1)),
        "ptr":  np.zeros((npt, nrho, len(PTR_EDGES) - 1)),
    }
    N = np.zeros((npt, nrho))

    for path in sorted(Path(args.skims).glob("*.pkl")):
        sk = pickle.load(open(path, "rb"))["skim"]
        wmap = {}
        for ds in sk.get("datasets", []):
            wmap[zlib.crc32(ds.encode()) & 0xFFFFFFFF] = xs_weight(ds, args.year) or 0.0
        t = sk["matched"]
        cols = ["ptreco", "etareco", "phireco", "mreco_g",
                "ptgen", "etagen", "phigen", "mgen_g", "weight", "dataset_id"]
        d = {c: np.asarray(t[c].value) for c in cols}
        if not len(d["ptreco"]):
            continue
        lut = {i: wmap.get(i, 0.0) for i in np.unique(d["dataset_id"])}
        w = d["weight"] * np.array([lut[i] for i in d["dataset_id"]])

        #### dphi wrapped into [0, pi] before the quadrature -- a raw subtraction
        #### puts every pair that straddles phi = +-pi at dR ~ 2pi
        dphi = np.abs(d["phireco"] - d["phigen"]) % (2 * np.pi)
        dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
        dr = np.hypot(d["etareco"] - d["etagen"], dphi)
        rg = rho(d["mgen_g"], d["ptgen"])
        rr = rho(d["mreco_g"], d["ptreco"])
        drho = rr - rg
        ptr = d["ptreco"] / d["ptgen"]

        #### bin at GEN, and apply the same m_g floor the measurement uses on
        #### both sides -- otherwise sub-floor jets dominate the d(rho) tail
        ip, ir = idx(d["ptgen"], PT_EDGES), idx(rg, RHO_EDGES)
        ok = ((ip >= 0) & (ir >= 0)
              & (d["mgen_g"] > args.mass_floor) & (d["mreco_g"] > args.mass_floor)
              & np.isfinite(drho) & np.isfinite(dr))
        for key, val, edges in (("dr", dr, DR_EDGES), ("drho", drho, DRHO_EDGES),
                                ("ptr", ptr, PTR_EDGES)):
            j = idx(val, edges)
            m = ok & (j >= 0)
            np.add.at(H[key], (ip[m], ir[m], j[m]), w[m])
        np.add.at(N, (ip[ok], ir[ok]), 1.0)
        print(f"  {path.name:<50s} done")

    out = {"pt_edges": PT_EDGES.tolist(), "rho_edges": RHO_EDGES.tolist(), "bins": []}
    for i in range(npt):
        hi = PT_EDGES[i + 1]
        print(f"\n=== pT_gen {PT_EDGES[i]:.0f}-{'inf' if hi > 5000 else f'{hi:.0f}'} GeV ===")
        print(f"{'rho bin':>15s} {'width':>6s} | {'dR med':>7s} {'dR 95%':>7s} | "
              f"{'drho med':>9s} {'drho sigma':>11s} {'w/sigma':>8s} | "
              f"{'pt resp':>8s} {'pt sigma':>9s} {'rawN':>9s}")
        for j in range(nrho):
            width = RHO_EDGES[j + 1] - RHO_EDGES[j]
            _, drmed, _, dr95 = quantiles(H["dr"][i, j], DR_EDGES)
            lo, med, up, _ = quantiles(H["drho"][i, j], DRHO_EDGES)
            sig = (up - lo) / 2.0
            plo, pmed, pup, _ = quantiles(H["ptr"][i, j], PTR_EDGES)
            psig = (pup - plo) / 2.0
            print(f"[{RHO_EDGES[j]:6.2f},{RHO_EDGES[j+1]:6.2f}] {width:6.2f} | "
                  f"{drmed:7.4f} {dr95:7.4f} | {med:9.3f} {sig:11.3f} "
                  f"{width/sig if sig > 0 else np.nan:8.2f} | "
                  f"{pmed:8.3f} {psig:9.3f} {N[i, j]:9.0f}")
            out["bins"].append(dict(
                ipt=i, irho=j, lo=RHO_EDGES[j], hi=RHO_EDGES[j + 1], width=width,
                dr_med=drmed, dr_95=dr95, drho_med=med, drho_sigma=sig,
                w_over_sigma=(width / sig if sig > 0 else None),
                pt_resp=pmed, pt_sigma=psig, n=N[i, j]))

    print("\nw/sigma is the bin width in units of the rho resolution: below ~2 the "
          "bin is\nnarrower than the smearing, which is what caps purity near 0.5.")
    if args.json_out:
        json.dump(out, open(args.json_out, "w"))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
