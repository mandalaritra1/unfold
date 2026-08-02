#!/usr/bin/env python3
"""Derive the Vincia gen-space reweight for the z+jet rho unfolding.

Compares the nominal analysis MC (PYTHIA8 madgraphMLM DY, gen level) against a
standalone Vincia-shower prediction (same madgraphMLM ME + HT binning, Rivet
route), and emits a per-pT gen reweight  w(rho) = Vincia / Pythia  on an axis
that nests inside the analysis gen binning (each analysis bin split in half; the
wide low-rho tail kept whole). This w feeds the response-reweighting machinery
(cf. study_response_reweight.py) to build the Vincia shower-model uncertainty.

Nominal Pythia gen  : inputs/zjet/rho/finebins/minimal_rho_fine_pythia_2018.pkl
                      (mode minimal_rho_fine; produce on casa via
                       smp_jetmass_run2 configs/zjet_pythia_2018_finebins.json)
Vincia standalone   : HT-binned per-event ntuples under VINCIA_DIR, stitched with
                      per-event weight xsec_HT/N_gen_HT (xsec + ScaledBy read from
                      each paired YODA).
Analysis gen binning: taken live from unfold.tools.binning.bin_edges (so the
                      [-1,0] merge etc. stay in sync).

Outputs -> outputs/zjet/rho/vincia_reweight/:
    overlay_rho_{g,u}_coarse.png   analysis-binning shape overlay + ratio
    reweight_rho_{g,u}_fine.png    reweight-axis overlay + w(rho)
    vincia_reweight.npz            coarse_{g,u}, rw_edges_{g,u}, w_{g,u}, pt_edges
"""
import argparse
import os, re, glob, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from unfold.tools.binning import bin_edges

DEFAULT_VINCIA_DIR = Path(os.environ.get(
    "VINCIA_DIR", "/Users/aritra/cernbox (2)/vincia_prod/vincia"))
DEFAULT_PYTHIA_INPUT = REPO / "inputs/zjet/rho/finebins/minimal_rho_fine_pythia_2018.pkl"
DEFAULT_OUTPUT_DIR = REPO / "outputs/zjet/rho/vincia_reweight"
VINCIA_DIR = DEFAULT_VINCIA_DIR
PKL = DEFAULT_PYTHIA_INPUT
OUT = DEFAULT_OUTPUT_DIR
CACHE = OUT / "vincia_gen_cache.npz"

PT_EDGES = np.array([200., 290., 400., 13000.])          # analysis pt bins 1,2,3
PT_LABELS = ["200-290 GeV", "290-400 GeV", r"400-$\infty$ GeV"]

# analysis gen binning, pulled live (pt-bin rows are identical for zjet -> row 1).
COARSE = {
    "g": np.array(bin_edges(groomed=True).gen_rho_edges_by_pt[1], float),
    "u": np.array(bin_edges(groomed=False).gen_rho_edges_by_pt[1], float),
}


def reweight_axis(coarse):
    """Reweight (fine) axis = coarse binning with every bin except the wide low-rho
    tail split in half; split points fall on existing fine-axis edges so it nests
    in both the response fine gen axis and the coarse analysis binning."""
    edges = [coarse[0]]
    for a, b in zip(coarse[:-1], coarse[1:]):
        if b - a > 1.0:
            edges.append(b)
        else:
            edges.append(0.5 * (a + b)); edges.append(b)
    return np.array(edges)


def read_yoda_norm(path):
    """(xsec, ScaledBy) from a finalized Rivet YODA (xsec from /_XSEC block)."""
    sb = xsec = None
    with open(path) as fh:
        in_xsec = seen = False
        for ln in fh:
            if sb is None and ln.startswith("ScaledBy:"):
                sb = float(ln.split(":", 1)[1])
            if ln.startswith("BEGIN YODA_ESTIMATE0D_V3 /_XSEC"):
                in_xsec, seen = True, False
            elif in_xsec:
                if ln.startswith("# value"):
                    seen = True
                elif seen:
                    tok = ln.split()
                    if tok and tok[0] not in ("---", "nan"):
                        xsec = float(tok[0]); in_xsec = False
            if sb is not None and xsec is not None:
                break
    return xsec, sb


def load_vincia():
    """HT-stitched per-event arrays jet_pt, rho_u, rho_g, weight."""
    pts, ru, rg, ws = [], [], [], []
    for htdir in sorted(glob.glob(os.path.join(VINCIA_DIR, "HT-*"))):
        ev_pt, ev_ru, ev_rg, xsec_f, ngen_f = [], [], [], [], []
        for nt in sorted(glob.glob(os.path.join(htdir, "ntuple_*.txt"))):
            yo = os.path.join(htdir, os.path.basename(nt)
                              .replace("ntuple_", "out_").replace(".txt", ".yoda"))
            xs, sb = read_yoda_norm(yo)
            if not xs or not sb:
                continue
            arr = np.loadtxt(nt, comments="#")
            if arr.size == 0:
                continue
            arr = np.atleast_2d(arr)                     # jet_pt m_u m_g rho_u rho_g w
            xsec_f.append(xs); ngen_f.append(xs / sb)
            ev_pt.append(arr[:, 0]); ev_ru.append(arr[:, 3]); ev_rg.append(arr[:, 4])
        if not ev_pt:
            continue
        xsec_f = np.array(xsec_f); ngen_f = np.array(ngen_f)
        w_ht = (np.sum(xsec_f * ngen_f) / np.sum(ngen_f)) / np.sum(ngen_f)
        pt = np.concatenate(ev_pt)
        pts.append(pt); ru.append(np.concatenate(ev_ru)); rg.append(np.concatenate(ev_rg))
        ws.append(np.full(pt.shape, w_ht))
        print(f"  {os.path.basename(htdir):16s} nfile={len(xsec_f):4d} "
              f"Ngen={np.sum(ngen_f):11.1f} nsel={pt.size:6d} w={w_ht:.3e}")
    return dict(jet_pt=np.concatenate(pts), rho_u=np.concatenate(ru),
                rho_g=np.concatenate(rg), weight=np.concatenate(ws))


def load_pythia():
    """Analysis Pythia gen -> per-mode (3 pt, nfine) values + fine edges."""
    d = pickle.load(open(PKL, "rb"))
    out = {}
    for mode, key in [("g", "ptjet_rhojet_g_gen"), ("u", "ptjet_rhojet_u_gen")]:
        h = d[key]
        vals = np.asarray(h.values())[0, :, :, 0]        # (4 pt, nfine)
        out[mode] = (vals[1:4], np.asarray(h.axes[2].edges))
    return out


def rebin(fine_vals, fine_edges, target_edges):
    idx = [int(np.argmin(np.abs(fine_edges - e))) for e in target_edges]
    return np.array([fine_vals[idx[i]:idx[i + 1]].sum() for i in range(len(idx) - 1)])


def hist_vincia(V, mode, edges):
    rho = V["rho_g"] if mode == "g" else V["rho_u"]
    out = np.zeros((3, len(edges) - 1))
    for i in range(3):
        m = (V["jet_pt"] >= PT_EDGES[i]) & (V["jet_pt"] < PT_EDGES[i + 1])
        out[i], _ = np.histogram(rho[m], bins=edges, weights=V["weight"][m])
    return out


def fine_reweight(V, mode, py_target, edges, min_frac=1e-4, min_count=25, clip=(0.2, 5.0)):
    """w(rho) = norm-Vincia / norm-Pythia per pt bin; w=1 where unmeasurable."""
    rho = V["rho_g"] if mode == "g" else V["rho_u"]
    nb = len(edges) - 1
    w = np.ones((3, nb)); pyn = np.zeros((3, nb)); vinn = np.zeros((3, nb))
    for i in range(3):
        m = (V["jet_pt"] >= PT_EDGES[i]) & (V["jet_pt"] < PT_EDGES[i + 1])
        vcnt, _ = np.histogram(rho[m], bins=edges)
        vw, _ = np.histogram(rho[m], bins=edges, weights=V["weight"][m])
        vn = vw / vw.sum(); pn = py_target[i] / py_target[i].sum()
        pyn[i], vinn[i] = pn, vn
        good = (pn > min_frac) & (vcnt >= min_count)
        w[i, good] = np.clip(vn[good] / pn[good], *clip)
    return w, pyn, vinn


def _panels(mode, edges, top_py, top_vin, ratio, ratio_label, fname, zoom, ylim_r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.CMS)
    wdt = np.diff(edges); cen = 0.5 * (edges[:-1] + edges[1:])
    sym = r"\rho_g" if mode == "g" else r"\rho_u"
    fig = plt.figure(figsize=(31, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.05, wspace=0.18)
    for i in range(3):
        ax = fig.add_subplot(gs[0, i]); axr = fig.add_subplot(gs[1, i], sharex=ax)
        hep.cms.label("Preliminary", data=False, loc=0, ax=ax, rlabel=PT_LABELS[i])
        pd = top_py[i] / top_py[i].sum() / wdt
        vd = top_vin[i] / top_vin[i].sum() / wdt
        ax.stairs(pd, edges, color="#5790fc", lw=2.4, label="Pythia (nominal MC)")
        ax.stairs(vd, edges, color="#e42536", lw=2.4, label="Vincia (standalone)")
        vis = cen > zoom
        ax.set_xlim(zoom, 0); ax.grid(alpha=0.25)
        ax.set_ylim(0, max(pd[vis].max(), vd[vis].max()) * 1.45)
        ax.set_ylabel(rf"$(1/N)\,dN/d{sym}$"); ax.tick_params(labelbottom=False)
        if i == 0:
            ax.legend(loc="upper left", framealpha=0.92)
        axr.stairs(ratio[i], edges, color="#7a21dd", lw=2.4, baseline=None)
        axr.axhline(1.0, color="k", lw=1, ls="--")
        axr.set_ylim(*ylim_r); axr.set_xlim(zoom, 0); axr.grid(alpha=0.25)
        axr.set_ylabel(ratio_label); axr.set_xlabel(rf"${sym}$")
    fig.savefig(OUT / fname, dpi=110, bbox_inches="tight"); plt.close(fig)
    return OUT / fname


def main():
    global VINCIA_DIR, PKL, OUT, CACHE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vincia-dir", type=Path, default=DEFAULT_VINCIA_DIR,
                        help=f"Vincia production directory (default: {DEFAULT_VINCIA_DIR})")
    parser.add_argument("--pythia-input", type=Path, default=DEFAULT_PYTHIA_INPUT,
                        help=f"fine-binned nominal Pythia pkl (default: {DEFAULT_PYTHIA_INPUT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"generated-output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--reload", action="store_true",
                        help="rebuild the cached Vincia gen arrays")
    args = parser.parse_args()

    VINCIA_DIR = args.vincia_dir.expanduser()
    PKL = args.pythia_input.expanduser()
    OUT = args.output_dir.expanduser()
    CACHE = OUT / "vincia_gen_cache.npz"
    if not PKL.is_file():
        raise SystemExit(f"missing Pythia input: {PKL}")
    if not VINCIA_DIR.is_dir():
        raise SystemExit(f"missing Vincia production directory: {VINCIA_DIR}")

    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not args.reload:
        z = np.load(CACHE); V = {k[2:]: z[k] for k in z if k.startswith("v_")}
        print("loaded", CACHE)
    else:
        print("Loading Vincia ntuples from", VINCIA_DIR)
        V = load_vincia(); np.savez(CACHE, **{f"v_{k}": v for k, v in V.items()})
    PY = load_pythia()

    W, RW = {}, {}
    for mode in ["g", "u"]:
        ce = COARSE[mode]; py_fine, fe = PY[mode]
        # coarse overlay (analysis binning)
        py_c = np.array([rebin(py_fine[i], fe, ce) for i in range(3)])
        vin_c = hist_vincia(V, mode, ce)
        ratio_c = np.divide(vin_c / vin_c.sum(1, keepdims=True),
                            py_c / py_c.sum(1, keepdims=True),
                            out=np.ones_like(vin_c),
                            where=py_c > 0)
        zoom = -5.0 if mode == "g" else -3.0
        print("  ->", _panels(mode, ce, py_c, vin_c, ratio_c, "Vin / Pyt",
                              f"overlay_rho_{mode}_coarse.png", zoom, (0.5, 1.6)))
        # reweight on the split-in-half axis
        rw = reweight_axis(ce)
        py_rw = np.array([rebin(py_fine[i], fe, rw) for i in range(3)])
        w, pyn, vinn = fine_reweight(V, mode, py_rw, rw)
        print("  ->", _panels(mode, rw, py_rw, hist_vincia(V, mode, rw), w,
                              r"$w=$Vin/Pyt", f"reweight_rho_{mode}_fine.png",
                              zoom, (0.4, 1.8)))
        W[mode], RW[mode] = w, rw
    np.savez(OUT / "vincia_reweight.npz",
             coarse_g=COARSE["g"], coarse_u=COARSE["u"],
             rw_edges_g=RW["g"], rw_edges_u=RW["u"],
             pt_edges=PT_EDGES, w_g=W["g"], w_u=W["u"])
    print("saved ->", OUT / "vincia_reweight.npz")


if __name__ == "__main__":
    main()
