#!/usr/bin/env python3
"""Z+jet-aligned binning study for the pair-split dijet/trijet channels.

Accumulates the event-level pair-split skims (all Run-2 eras, matched +
inclusive reco/gen tables, data reco) onto a fine 0.125-wide grid of the
transformed coordinate 2*log10(rho), rho = m/(pT*R), then evaluates nested
candidate binnings whose edges all live on the Z+jet half/quarter-integer
grid — so dijet (finer), trijet (coarser), and Z+jet rebin exactly onto a
common axis for combined plots.

Two hard-won facts encoded here (see ai-wiki repos/unfold.md):
* the stored skim ``weight`` is FULLY normalized (genWeight * xs*lumi/sumw *
  prescale) — never multiply by an xs weight again (the archived width-scan
  scripts did, and that is wrong for these skims);
* the production nominal response applies NO mass floor (the ``_mfloor2``
  histograms are a separate variant), so the default here is no floor.

Usage:
  # one-time accumulation pass (writes one NPZ per channel x mode)
  python scripts/studies/study_aligned_binning.py accumulate
  # candidate evaluation from the NPZs
  python scripts/studies/study_aligned_binning.py evaluate
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIM_ROOT = Path("/Users/aritra/cernbox (2)/hadronic_skims_pairsplit")
ERAS = ("2016APV", "2016", "2017", "2018")
OUTPUT_DIR = REPO_ROOT / "outputs" / "studies" / "aligned_binning"
JET_R = 0.8

PT_EDGES = {
    "dijet": np.array([200.0, 290.0, 400.0, 480.0, 570.0, 13000.0]),
    "trijet": np.array([200.0, 290.0, 400.0, 13000.0]),
}
MASS_COLUMNS = {"groomed": ("mreco_g", "mgen_g"), "ungroomed": ("mreco_u", "mgen_u")}

# Fine accumulation grid: 0.125-wide from -6 to 0 (every half-, quarter- and
# eighth-integer candidate edge is a subset), coarse tail below.
FINE_EDGES = np.concatenate(([-10.0, -8.0], np.arange(-6.0, 0.0001, 0.125)))


def rho(mass, pt):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.log10(mass / (pt * JET_R))


def idx(values, edges):
    i = np.digitize(values, edges) - 1
    return np.where(np.isfinite(values) & (i >= 0) & (i < len(edges) - 1), i, -1)


def _columns(skim, table, names):
    t = skim[table]
    first = t[names[0]].value
    if not len(first):
        return None
    return {name: np.asarray(t[name].value) for name in names}


def _fill_2d(pt_i, rho_i, weights, out_w, out_w2):
    ok = (pt_i >= 0) & (rho_i >= 0)
    flat = pt_i[ok] * (len(FINE_EDGES) - 1) + rho_i[ok]
    out_w += np.bincount(flat, weights=weights[ok], minlength=out_w.size).reshape(out_w.shape)
    out_w2 += np.bincount(flat, weights=weights[ok] ** 2, minlength=out_w.size).reshape(out_w.shape)


def accumulate(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nf = len(FINE_EDGES) - 1
    for channel in ("dijet", "trijet"):
        pt_edges = PT_EDGES[channel]
        npt = len(pt_edges) - 1
        for mode, (mreco_col, mgen_col) in MASS_COLUMNS.items():
            R4w = np.zeros((npt, nf, npt, nf))
            R4w2 = np.zeros_like(R4w)
            tables = {
                name: (np.zeros((npt, nf)), np.zeros((npt, nf)))
                for name in ("gen_inclusive", "reco_inclusive", "data_reco")
            }
            sources = []
            mc_paths = sorted(
                p for era in ERAS
                for p in glob.glob(str(SKIM_ROOT / era / f"{channel}_mc" / "*.pkl"))
            )
            data_paths = sorted(
                p for era in ERAS
                for p in glob.glob(str(SKIM_ROOT / era / f"{channel}_data" / "*.pkl"))
            )
            for path in mc_paths:
                skim = pickle.load(open(path, "rb"))["skim"]
                sources.append(path)
                matched = _columns(
                    skim, "matched",
                    ["ptreco", mreco_col, "ptgen", mgen_col, "weight"],
                )
                if matched is not None:
                    w = matched["weight"]  # fully normalized already — do NOT rescale
                    ipg = idx(matched["ptgen"], pt_edges)
                    ipr = idx(matched["ptreco"], pt_edges)
                    irg = idx(rho(matched[mgen_col], matched["ptgen"]), FINE_EDGES)
                    irr = idx(rho(matched[mreco_col], matched["ptreco"]), FINE_EDGES)
                    both = (ipg >= 0) & (ipr >= 0) & (irg >= 0) & (irr >= 0)
                    flat = ((ipg[both] * nf + irg[both]) * npt + ipr[both]) * nf + irr[both]
                    R4w += np.bincount(flat, weights=w[both], minlength=R4w.size).reshape(R4w.shape)
                    R4w2 += np.bincount(flat, weights=w[both] ** 2, minlength=R4w.size).reshape(R4w.shape)
                gen = _columns(skim, "gen", ["ptgen", mgen_col, "weight"])
                if gen is not None:
                    _fill_2d(idx(gen["ptgen"], pt_edges),
                             idx(rho(gen[mgen_col], gen["ptgen"]), FINE_EDGES),
                             gen["weight"], *tables["gen_inclusive"])
                reco = _columns(skim, "reco", ["ptreco", mreco_col, "weight"])
                if reco is not None:
                    _fill_2d(idx(reco["ptreco"], pt_edges),
                             idx(rho(reco[mreco_col], reco["ptreco"]), FINE_EDGES),
                             reco["weight"], *tables["reco_inclusive"])
                print(f"  {channel} {mode}: {Path(path).name}", flush=True)
            for path in data_paths:
                skim = pickle.load(open(path, "rb"))["skim"]
                sources.append(path)
                reco = _columns(skim, "reco", ["ptreco", mreco_col, "weight"])
                if reco is not None:
                    _fill_2d(idx(reco["ptreco"], pt_edges),
                             idx(rho(reco[mreco_col], reco["ptreco"]), FINE_EDGES),
                             reco["weight"], *tables["data_reco"])
            out = OUTPUT_DIR / f"fine_accumulation_{channel}_{mode}.npz"
            np.savez_compressed(
                out,
                fine_edges=FINE_EDGES,
                pt_edges=pt_edges,
                response_w=R4w,           # (ptgen, rhogen, ptreco, rhoreco)
                response_w2=R4w2,
                gen_inclusive_w=tables["gen_inclusive"][0],
                gen_inclusive_w2=tables["gen_inclusive"][1],
                reco_inclusive_w=tables["reco_inclusive"][0],
                reco_inclusive_w2=tables["reco_inclusive"][1],
                data_reco_w=tables["data_reco"][0],
                data_reco_w2=tables["data_reco"][1],
                sources=np.asarray(sources),
            )
            print(f"wrote {out}")


# --------------------------------------------------------------------------
# Candidate definitions.  Every edge lies on the 0.125 fine grid; shown gen
# edges are chosen so all three channels (and Z+jet) share a common nested
# grid: Z+jet groomed shown [-3.5,-3,-2.5,-2,-1.5,-1,0], ungroomed
# [-2.5,-2,-1.5,-1,0].  The hidden buffer below the shown floor is 0.5-wide
# down to -5 (groomed) / a single catch-all (ungroomed), tail sinks below.
# Reco candidates are strict 2:1 refinements of shown gen bins; the LAST gen
# bin stays 1:1 (kinematic-edge sparsity, 2026-08-18 lesson) unless noted.
# --------------------------------------------------------------------------

def _two_to_one_reco(gen_edges, shown_floor, last_bin_one_to_one=True):
    """Nested 2:1 reco refinement of the shown gen bins; buffer kept 1:1."""
    gen_edges = list(gen_edges)
    reco = [gen_edges[0]]
    for low, high in zip(gen_edges[:-1], gen_edges[1:]):
        is_last = high == gen_edges[-1]
        if low >= shown_floor - 1e-9 and not (is_last and last_bin_one_to_one):
            reco.append((low + high) / 2.0)
        reco.append(high)
    return tuple(reco)


GROOMED_BUFFER = (-10.0, -5.0, -4.5, -4.0, -3.5)
CANDIDATES = {
    "dijet": {
        "groomed": {
            # Z+jet parity: identical shown gen bins (last bin [-1,0] merged).
            "zjet_parity": (-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 0.0),
            # 0.25-wide through the peak, 0.5 at the edges.
            "quarter_peak": (-3.5, -3.0, -2.5, -2.25, -2.0, -1.75, -1.5,
                             -1.25, -1.0, -0.75, -0.5, 0.0),
            # 0.25-wide everywhere shown.
            "quarter_full": (-3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0,
                             -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, 0.0),
            # Aligned hybrids: merged half-integer tail (still nested in the
            # Z+jet grid), 0.25 only where resolution can support it.
            "hybrid_wide_tail": (-3.5, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0),
            "hybrid_quarter_peak": (-3.5, -2.5, -2.0, -1.5, -1.25, -1.0,
                                    -0.75, -0.5, 0.0),
            "hybrid_quarter_wide": (-3.5, -2.5, -2.0, -1.75, -1.5, -1.25,
                                    -1.0, -0.75, -0.5, 0.0),
            # The unified common grid (merged [-1, 0] top, per Z+jet):
            "aligned_final": (-3.5, -2.5, -2.0, -1.5, -1.0, 0.0),
            # Same, with one 0.25 split in the best-resolution peak bin.
            "aligned_final_qpeak": (-3.5, -2.5, -2.0, -1.5, -1.25, -1.0, 0.0),
            # Quarter splits across the peak but NO edge at -0.5/-0.25 (the
            # near-empty kinematic-edge bin killed hybrid_quarter_peak) and no
            # quarter bins below -1.5.  Purity matches the legacy coarse_tail
            # grid (0.36-0.85) with edges on the aligned lattice; merging the
            # two quarter pairs recovers aligned_final exactly.  APPROVED as
            # the dijet groomed grid 2026-08-27.
            "aligned_seven": (-3.5, -2.5, -2.0, -1.5, -1.25, -1.0, -0.75, 0.0),
            # One more split at -1.75: worst bin drops to 0.35 at mid pT.
            "aligned_eight": (-3.5, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0,
                              -0.75, 0.0),
        },
        "ungroomed": {
            "current": (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0),
            "zjet_parity": (-2.5, -2.0, -1.5, -1.0, 0.0),
            "quarter_peak": (-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0,
                             -0.5, 0.0),
        },
    },
    "trijet": {
        "groomed": {
            "zjet_parity": (-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 0.0),
            "half_from3": (-3.0, -2.5, -2.0, -1.5, -1.0, 0.0),
            "coarse": (-3.0, -2.0, -1.5, -1.0, 0.0),
            "wide_tail_deep": (-3.5, -2.5, -2.0, -1.5, -1.0, 0.0),
            "wide_tail_split_top": (-3.5, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0),
        },
        "ungroomed": {
            "current": (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0),
            "zjet_parity": (-2.5, -2.0, -1.5, -1.0, 0.0),
        },
    },
}
UNGROOMED_BUFFER = (-10.0, -2.5)


def _candidate_edges(mode, shown):
    buffer_edges = GROOMED_BUFFER if mode == "groomed" else UNGROOMED_BUFFER
    assert abs(buffer_edges[-1] - shown[0]) < 1e-9 or buffer_edges[-1] < shown[0]
    if abs(buffer_edges[-1] - shown[0]) > 1e-9:
        buffer_edges = buffer_edges + (shown[0],)
    gen = tuple(buffer_edges) + tuple(shown[1:])
    reco = _two_to_one_reco(gen, shown_floor=shown[0])
    return gen, reco


def _regroup_axis(fine_w, fine_edges, target_edges, axis):
    """Sum a fine axis onto nested target edges."""
    fine_edges = np.asarray(fine_edges)
    positions = [int(np.flatnonzero(np.isclose(fine_edges, e))[0]) for e in target_edges]
    pieces = [
        fine_w.take(range(a, b), axis=axis).sum(axis=axis, keepdims=True)
        for a, b in zip(positions[:-1], positions[1:])
    ]
    return np.concatenate(pieces, axis=axis)


def evaluate(args):
    lines = ["# Z+jet-aligned pair-split binning study", ""]
    payload = {}
    for channel in ("dijet", "trijet"):
        for mode in ("groomed", "ungroomed"):
            f = np.load(OUTPUT_DIR / f"fine_accumulation_{channel}_{mode}.npz",
                        allow_pickle=False)
            fine_edges = f["fine_edges"]
            npt = len(f["pt_edges"]) - 1
            lines.append(f"## {channel} {mode}")
            lines.append("")
            for name, shown in CANDIDATES[channel][mode].items():
                gen_edges, reco_edges = _candidate_edges(mode, shown)
                shown_mask_gen = np.asarray(gen_edges[:-1]) >= shown[0] - 1e-9
                # Regroup the 4D response: gen axis -> gen_edges, reco -> reco_edges
                R = _regroup_axis(f["response_w"], fine_edges, gen_edges, axis=1)
                R = _regroup_axis(R, fine_edges, reco_edges, axis=3)
                Gi = _regroup_axis(f["gen_inclusive_w"], fine_edges, gen_edges, axis=1)
                Di_w = _regroup_axis(f["data_reco_w"], fine_edges, gen_edges, axis=1)
                Di_w2 = _regroup_axis(f["data_reco_w2"], fine_edges, gen_edges, axis=1)
                # Map reco onto the gen layout (reco edges nest in gen edges)
                Rg = _regroup_axis(R, np.asarray(reco_edges), gen_edges, axis=3)
                # Flatten (pt x genbin) treating pt migration too
                ngen = len(gen_edges) - 1
                M = Rg.reshape(npt * ngen, npt * ngen)   # rows gen, cols reco(gen layout)
                diag = np.diag(M)
                purity = np.divide(diag, M.sum(axis=0), out=np.zeros_like(diag),
                                   where=M.sum(axis=0) > 0)
                stability = np.divide(diag, M.sum(axis=1), out=np.zeros_like(diag),
                                      where=M.sum(axis=1) > 0)
                matched_gen = Rg.sum(axis=(2, 3)).reshape(-1)
                efficiency = np.divide(matched_gen, Gi.reshape(-1),
                                       out=np.zeros_like(matched_gen),
                                       where=Gi.reshape(-1) > 0)
                data_rel = np.divide(np.sqrt(Di_w2), Di_w,
                                     out=np.full_like(Di_w, np.nan),
                                     where=Di_w > 0).reshape(-1)
                # Column-normalized condition of the (reco x gen) response
                K = R.transpose(2, 3, 0, 1).reshape(npt * (len(reco_edges) - 1),
                                                    npt * ngen)
                colsum = K.sum(axis=0)
                Kn = np.divide(K, colsum, out=np.zeros_like(K), where=colsum > 0)
                sv = np.linalg.svd(Kn, compute_uv=False)
                condition = float(sv[0] / sv[sv > 1e-12][-1])
                shown_flat = np.tile(shown_mask_gen, npt)
                per_pt_min = [
                    float(purity.reshape(npt, ngen)[i][shown_mask_gen].min())
                    for i in range(npt)
                ]
                per_bin = {
                    f"pt{i}": [round(float(v), 3)
                               for v in purity.reshape(npt, ngen)[i][shown_mask_gen]]
                    for i in range(npt)
                }
                p, s = purity[shown_flat], stability[shown_flat]
                d = data_rel[shown_flat]
                summary = {
                    "shown_gen_edges": list(shown),
                    "gen_edges": list(gen_edges),
                    "reco_edges": list(reco_edges),
                    "purity_min": float(p.min()), "purity_mean": float(p.mean()),
                    "stability_min": float(s.min()), "stability_mean": float(s.mean()),
                    "bins_below_p05": int((p < 0.5).sum()),
                    "shown_bins": int(shown_flat.sum()),
                    "efficiency_min": float(efficiency[shown_flat].min()),
                    "data_relstat_max": float(np.nanmax(d)),
                    "data_relstat_median": float(np.nanmedian(d)),
                    "condition": condition,
                    "purity_min_per_pt": per_pt_min,
                    "purity_per_bin": per_bin,
                }
                payload[f"{channel}_{mode}_{name}"] = summary
                lines.append(
                    f"- **{name}** ({int(shown_flat.sum())} shown bins): "
                    f"purity min/mean {p.min():.3f}/{p.mean():.3f}, "
                    f"stability min/mean {s.min():.3f}/{s.mean():.3f}, "
                    f"bins<0.5 purity: {(p < 0.5).sum()}, "
                    f"eff min {efficiency[shown_flat].min():.3f}, "
                    f"data rel-stat max/median {np.nanmax(d)*100:.1f}%/{np.nanmedian(d)*100:.1f}%, "
                    f"condition {condition:.1f}; per-pT purity min "
                    + "/".join(f"{v:.2f}" for v in per_pt_min)
                )
            lines.append("")
    report = OUTPUT_DIR / "aligned_binning_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "aligned_binning_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(report)
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("accumulate", "evaluate"))
    args = parser.parse_args()
    if args.stage == "accumulate":
        accumulate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
