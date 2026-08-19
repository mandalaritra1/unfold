#!/usr/bin/env python
"""Is the 12-bin groomed-rho proposal stable under a tau=0 TUnfold?

Purity said the binning is resolvable and the data/MC comparison said it is
describable. Neither says the response is INVERTIBLE at tau=0, which is the
production choice ("coarsen, not regularize") -- that is what this asks.

Three questions, in order of how much they can kill the binning:

  1. self-closure -- unfold nominal MC with itself; at tau=0 this must return
     gen to machine precision on every constrained bin. A failure here is a
     construction bug, not a binning verdict.
  2. error amplification -- the unfolded stat error against the bin-by-bin
     error the same data would give (|x_i| * sigma_y/y averaged over the reco
     bins feeding it), and the TUnfold global correlation rho_i. This is where
     a too-fine GEN binning dies: closure stays exact and the errors blow up.
  3. the data unfold itself -- negative bins, the per-pT normalized shape ratio
     against MC gen, and sign flips of (ratio-1) between rho neighbours.

The proposal is a RECO binning, derived from reco-vs-gen purity. Using it
unchanged as the gen binning makes a SQUARE system -- zero degrees of freedom,
nothing left to test the model with -- so the candidates below walk the
"coarsen" lever from square down to the production-equivalent 2:1 nesting, and
include the production rho binning itself at the same gen-bin count as a
like-for-like reference.

    source ~/Projects/unfold/scripts/setup_root.sh
    ~/Projects/unfold/.venv/bin/python scripts/rho_unfold_stability.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def gen_map(n_pt, n_rho, pt_groups, rho_groups):
    """(Ng_fine x Ng_coarse) 0/1 map merging pT and rho groups at gen level."""
    M = np.zeros((n_pt * n_rho, len(pt_groups) * len(rho_groups)))
    for a, pg in enumerate(pt_groups):
        for b, rg in enumerate(rho_groups):
            for p in pg:
                for r in rg:
                    M[p * n_rho + r, a * len(rho_groups) + b] = 1.0
    return M


def tunfold(
    A,
    A_w2,
    misses,
    misses_w2,
    y,
    Vy,
    tag,
    *,
    area_constraint=True,
    quiet=False,
    include_rho=True,
):
    """One tau=0 TUnfoldDensity pass.

    ``area_constraint`` is exposed for validation studies.  Production keeps
    the historical area constraint; the no-area result is useful as an
    independent generalized-least-squares cross-check at tau=0.
    """
    import ROOT

    ROOT.TH1.AddDirectory(False)
    if quiet:
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
    Nr, Ng = A.shape
    hA = ROOT.TH2D(f"A{tag}", "", Ng, 0.0, float(Ng), Nr, 0.0, float(Nr))
    for j in range(Ng):
        for i in range(Nr):
            hA.SetBinContent(j + 1, i + 1, A[i, j])
            hA.SetBinError(j + 1, i + 1, np.sqrt(A_w2[i, j]))
        hA.SetBinContent(j + 1, 0, misses[j])
        hA.SetBinError(j + 1, 0, np.sqrt(misses_w2[j]))

    hy = ROOT.TH1D(f"y{tag}", "", Nr, 0.0, float(Nr))
    for i in range(Nr):
        hy.SetBinContent(i + 1, y[i])
        hy.SetBinError(i + 1, np.sqrt(max(Vy[i, i], 0.0)))
    hV = ROOT.TH2D(f"V{tag}", "", Nr, 0.0, float(Nr), Nr, 0.0, float(Nr))
    for i in range(Nr):
        for j in range(Nr):
            hV.SetBinContent(i + 1, j + 1, Vy[i, j])

    constraint = (
        ROOT.TUnfold.kEConstraintArea
        if area_constraint
        else ROOT.TUnfold.kEConstraintNone
    )
    unf = ROOT.TUnfoldDensity(
        hA,
        ROOT.TUnfold.kHistMapOutputHoriz,
        ROOT.TUnfold.kRegModeCurvature,
        constraint,
        ROOT.TUnfoldDensity.kDensityModeNone,
    )
    status = unf.SetInput(hy, 0.0, 0.0, hV)
    unf.DoUnfold(0.0)

    def t1(h, n):
        return np.array([h.GetBinContent(i + 1) for i in range(n)])

    def t2(h, n):
        return np.array([[h.GetBinContent(i + 1, j + 1) for j in range(n)]
                         for i in range(n)])

    rho_i = (
        t1(unf.GetRhoItotal(f"r{tag}"), Ng)
        if include_rho
        else np.full(Ng, np.nan)
    )
    return dict(status=int(status), x=t1(unf.GetOutput(f"o{tag}"), Ng),
                Ein=t2(unf.GetEmatrixInput(f"ei{tag}"), Ng),
                Esys=t2(unf.GetEmatrixSysUncorr(f"es{tag}"), Ng),
                rho_i=rho_i, chi2A=float(unf.GetChi2A()),
                ndf=int(unf.GetNdf()))


def analyse(z, label, pt_groups, rho_groups, floor, out, norm=(-2.85, -0.55)):
    n_pt = len(z["pt_edges"]) - 1
    redges = z["rho_edges"]
    n_rho = len(redges) - 1
    if pt_groups is None:
        pt_groups = [[p] for p in range(n_pt)]
    if rho_groups is None:
        rho_groups = [[r] for r in range(n_rho)]

    M = gen_map(n_pt, n_rho, pt_groups, rho_groups)
    A, Aw2 = z["A"] @ M, z["A_w2"] @ M
    gen, gen_w2 = z["gen"] @ M, z["gen_w2"] @ M
    reco = z["reco"]
    n_pg, n_rg = len(pt_groups), len(rho_groups)
    Nr, Ng = A.shape

    fakes = reco - A.sum(axis=1)
    misses = gen - A.sum(axis=0)
    misses_w2 = np.maximum(gen_w2 - Aw2.sum(axis=0), 0.0)

    gedges = np.r_[[redges[g[0]] for g in rho_groups], redges[-1]]
    shown = np.tile(gedges[:-1] >= floor, (n_pg, 1)).reshape(-1)
    constrained = A.sum(axis=0) > 0
    s = shown & (gen > 0) & constrained

    print(f"\n{'=' * 76}\n{label}")
    print(f"  reco {Nr} = {n_pt}pt x {n_rho}rho | gen {Ng} = {n_pg}pt x "
          f"{n_rg}rho | reported {int(s.sum())} (rho >= {floor})")

    mc = tunfold(A, Aw2, misses, misses_w2, reco - fakes,
                 np.diag(z["reco_w2"]), f"_mc{label}")
    okg = (gen > 0) & constrained
    dev = np.abs(mc["x"][okg] / gen[okg] - 1.0).max()

    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(reco > 0, fakes / reco, 0.0)
    surv = 1.0 - np.clip(f, 0.0, 1.0)
    y = z["data_reco"] * surv
    Vy = z["data_V"] * np.outer(surv, surv)
    da = tunfold(A, Aw2, misses, misses_w2, y, Vy, f"_da{label}")

    err = np.sqrt(np.maximum(np.diag(da["Ein"]), 0.0))
    err_r = np.sqrt(np.maximum(np.diag(da["Esys"]), 0.0))
    #### bin-by-bin reference: the relative data error of the reco bins that
    #### feed each gen bin, weighted by how much of the response they carry.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_y = np.where(y > 0, np.sqrt(np.diag(Vy)) / y, np.nan)
    Wt = A / np.maximum(A.sum(axis=0), 1e-30)
    rel_y_g = np.nansum(np.where(np.isnan(rel_y[:, None]), 0.0,
                                 rel_y[:, None] * Wt), axis=0)
    bbb = np.abs(da["x"]) * rel_y_g
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(bbb > 0, err / bbb, np.nan)
        rel = np.where(da["x"] > 0, err / da["x"], np.nan)
        rel_resp = np.where(da["x"] > 0, err_r / da["x"], np.nan)

    x2, g2 = da["x"].reshape(n_pg, n_rg), gen.reshape(n_pg, n_rg)
    s2 = s.reshape(n_pg, n_rg)
    #### Normalize the shape over a fixed rho WINDOW, not over every reported
    #### bin. The lowest reported bin is a merged resolution-limited catch-all
    #### holding 25-30% of the yield in one bin -- letting it into the
    #### normalization lets it set the scale for the whole slice, which is what
    #### made the core look 10-15% low at high pT. Same window for every
    #### candidate so the rows stay comparable.
    inwin = (gedges[:-1] >= norm[0] - 1e-9) & (gedges[1:] <= norm[1] + 1e-9)
    flips, tot, ratios, edge = 0, 0, [], []
    for p in range(n_pg):
        m, w = s2[p], s2[p] & inwin
        if m.sum() < 2 or w.sum() < 1:
            continue
        sc = (g2[p][w].sum() / g2[p][m].sum()) / (x2[p][w].sum() / x2[p][m].sum())
        r = (x2[p][m] / x2[p][m].sum()) / (g2[p][m] / g2[p][m].sum()) * sc
        ratios.append(r[w[m]])
        edge.append(r[~w[m]])
        d = np.sign(r[w[m]] - 1.0)
        flips += int((d[1:] * d[:-1] < 0).sum())
        tot += max(len(d) - 1, 0)
    allr = np.concatenate(ratios)
    alledge = np.concatenate(edge) if any(len(e) for e in edge) else np.array([np.nan])

    rec = dict(
        label=label, n_reco=Nr, n_gen=Ng, n_pt_gen=n_pg, n_rho_gen=n_rg,
        n_shown=int(s.sum()), closure_max=float(dev),
        rel_stat_median=float(np.nanmedian(rel[s])),
        rel_stat_max=float(np.nanmax(rel[s])),
        rel_resp_median=float(np.nanmedian(rel_resp[s])),
        amp_median=float(np.nanmedian(amp[s])), amp_max=float(np.nanmax(amp[s])),
        rho_median=float(np.median(np.abs(da["rho_i"][s]))),
        rho_max=float(np.abs(da["rho_i"][s]).max()),
        negative=int((da["x"][s] < 0).sum()), flips=flips, flips_of=tot,
        shape_min=float(allr.min()), shape_max=float(allr.max()),
        shape_median=float(np.median(allr)),
        norm_window=[float(norm[0]), float(norm[1])],
        n_norm_bins=int(inwin.sum()),
        edge_min=float(np.nanmin(alledge)), edge_max=float(np.nanmax(alledge)),
        chi2A=float(da["chi2A"]), ndf=int(da["ndf"]),
        gen_rho_edges=[float(v) for v in gedges],
        pt_groups=[[int(v) for v in g] for g in pt_groups],
        unfolded=da["x"].tolist(), gen=gen.tolist(), err=err.tolist(),
        shown=s.tolist(), rho_i=da["rho_i"].tolist(), amp=amp.tolist(),
    )
    print(f"  self-closure max|unf/gen-1| = {dev:.2e}   (status {mc['status']})")
    print(f"  rel stat: median {rec['rel_stat_median']:.2%}, max "
          f"{rec['rel_stat_max']:.2%}; response MC-stat median "
          f"{rec['rel_resp_median']:.2%}")
    print(f"  error amplification vs bin-by-bin: median {rec['amp_median']:.2f}, "
          f"max {rec['amp_max']:.2f}")
    print(f"  global correlation |rho_i|: median {rec['rho_median']:.3f}, max "
          f"{rec['rho_max']:.3f}")
    print(f"  shape ratio IN WINDOW [{norm[0]}, {norm[1]}] "
          f"({rec['n_norm_bins']} bins): median {rec['shape_median']:.3f}, range "
          f"[{rec['shape_min']:.3f}, {rec['shape_max']:.3f}]; "
          f"neighbour sign flips {flips}/{tot}; negatives {rec['negative']}")
    print(f"  catch-all bins outside the window: "
          f"[{rec['edge_min']:.3f}, {rec['edge_max']:.3f}]")
    print(f"  chi2A/ndf = {rec['chi2A']:.1f}/{rec['ndf']}")
    #### per-pT amplification: a single bad pT slice is a pT-merge problem,
    #### not a rho-binning problem, and the two need different fixes
    a2 = amp.reshape(n_pg, n_rg)
    print("  amp by pT slice: " + "  ".join(
        f"{np.nanmedian(a2[p][s2[p]]):.2f}/{np.nanmax(a2[p][s2[p]]):.1f}"
        for p in range(n_pg) if s2[p].any()))
    out.append(rec)


#### Trijet reco binning is 7 rho bins [-4,-3,-2.4,-1.9,-1.5,-1.1,-0.7,0] plus
#### the [-10,-4] buffer at index 0, over 3 pT bins [200,290,430,inf]. Those
#### edges come from the same rho-only-purity rule the dijet proposal used, but
#### rerun on trijet: 7 bins is the finest binning where EVERY bin clears 0.5 in
#### every pT slice. The core window excludes both catch-alls: [-4,-3] is the
#### resolution-limited low-rho sink and [-0.7,0] is the overflow bin.
TRIJET_NORM = (-3.0, -0.7)


def _trijet(zp, zq, out, json_out):
    #### square: the reco binning used unchanged as gen -- zero d.o.f.
    analyse(zp, "proposal_square", None, None, -4.0, out, norm=TRIJET_NORM)
    #### merge the two lowest physics bins, where sigma(drho) is largest
    analyse(zp, "proposal_coarse_tail", None,
            [[0], [1, 2], [3], [4], [5], [6], [7]], -4.0, out,
            norm=TRIJET_NORM)
    #### one merge further up the tail
    analyse(zp, "proposal_coarse3", None,
            [[0], [1, 2], [3, 4], [5], [6], [7]], -4.0, out, norm=TRIJET_NORM)
    #### full 2:1 nesting -- the production convention
    analyse(zp, "proposal_2to1", None,
            [[0], [1, 2], [3, 4], [5, 6], [7]], -4.0, out, norm=TRIJET_NORM)
    #### tail-coarsened with the 290-430 + 430-inf pT merge on top; the pT-merge
    #### scan put those two slices at 0.771/0.835 pT purity, the weakest pair
    analyse(zp, "proposal_coarse_tail_ptmerge", [[0], [1, 2]],
            [[0], [1, 2], [3], [4], [5], [6], [7]], -4.0, out,
            norm=TRIJET_NORM)
    #### like-for-like: the PRODUCTION rho axis 2:1 nested (6 reported bins),
    #### same pT bins, same data, same machinery
    analyse(zq, "production_2to1", None,
            [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
            -3.5, out, norm=TRIJET_NORM)
    json.dump(out, open(json_out, "w"))
    print(f"\nwrote {json_out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="unfold_inputs.npz")
    ap.add_argument("--npz-prod", default="unfold_inputs_prod.npz")
    ap.add_argument("--json-out", default="unfold_stability.json")
    ap.add_argument("--channel", default="dijet", choices=["dijet", "trijet"],
                    help="which candidate ladder to walk; the two channels "
                         "have different reco binnings so the gen-group "
                         "indices below are not interchangeable")
    args = ap.parse_args(argv)
    zp = {k: v for k, v in np.load(args.npz).items()}
    zq = {k: v for k, v in np.load(args.npz_prod).items()}

    #### rho-group indices are into the RECO bins of the corresponding npz;
    #### index 0 is the [-10, -4] buffer in both.
    P = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
    out = []
    if args.channel == "trijet":
        return _trijet(zp, zq, out, args.json_out)
    #### the proposal used unchanged as the gen binning: square, zero d.o.f.
    analyse(zp, "proposal_square", None, None, -4.0, out)
    #### coarsen the low-rho half only, where the resolution is worst
    analyse(zp, "proposal_coarse_tail", None,
            [[0], [1, 2], [3, 4], [5], [6], [7], [8], [9], [10], [11], [12]],
            -4.0, out)
    #### one merge further down the tail
    analyse(zp, "proposal_coarse3", None,
            [[0], [1, 2], [3, 4], [5, 6], [7], [8], [9], [10], [11], [12]],
            -4.0, out)
    #### the tail-coarsened binning with the 400-570 pT merge on top
    analyse(zp, "proposal_coarse_tail_ptmerge", [[0], [1], [2, 3], [4]],
            [[0], [1, 2], [3, 4], [5], [6], [7], [8], [9], [10], [11], [12]],
            -4.0, out)
    #### full 2:1 nesting -- the production convention, 6 reported rho bins
    analyse(zp, "proposal_2to1", None,
            [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], -4.0, out)
    #### 2:1 nesting plus the 400-570 pT merge the pT-purity scan asked for
    analyse(zp, "proposal_2to1_ptmerge", [[0], [1], [2, 3], [4]],
            [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], -4.0, out)
    #### like-for-like reference: the PRODUCTION rho binning, 2:1 nested, same
    #### 6 reported gen bins, same pT bins, same data, same machinery
    analyse(zq, "production_2to1", None,
            [[0], [1], [2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
            -3.5, out)

    json.dump(out, open(args.json_out, "w"))
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
