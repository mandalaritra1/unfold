#!/usr/bin/env python3
"""Derive Herwig/Pythia (h/p) reweighting splines for the *dijet* channel.

This mirrors ``notebooks/derive_h_over_p.ipynb`` (which targets the zjet
channel) but reads the dijet rho inputs and writes channel-named artefacts.

For each gen pT bin we form the (sum-normalized) Herwig/Pythia ratio as a
function of rho = log10(rho^2) = ``mpt_gen``, fit a smoothing spline, evaluate
it on a fine grid and clamp it to a safe range. The resulting (pt_edges,
rho_grids, w_grids) bundle is consumed by ``corrections.PtVarWeighter`` exactly
like the zjet ``spline_*.npz`` files.

Run from the repo root:

    python scripts/derive_dijet_h_over_p.py
"""
import os
import pickle as pkl

import numpy as np
from scipy.interpolate import UnivariateSpline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(REPO_ROOT, "inputs", "dijet", "rho")
PYTHIA_PKL = os.path.join(IN_DIR, "minimal_rho_dijet_mg_pythia8_2018.pkl")
HERWIG_PKL = os.path.join(IN_DIR, "minimal_rho_dijet_herwig_2018.pkl")

# Where the splines are consumed by the processor.
OUT_DIR = os.path.join(
    REPO_ROOT, "..", "smp_jetmass_run2", "smp_jetmass_run2", "corrections"
)

# Clamp the weight to a safe range (matches the zjet derivation notebook).
W_CLIP = (0.5, 1.5)
N_GRID = 2000


def _load(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def _ratio_vs_rho(h_p, h_h, ipt):
    """Sum-normalized Herwig/Pythia ratio (+errors) for one pt bin."""
    # axes: (dataset, systematic, ptgen, mpt_gen). Sum datasets, pick nominal.
    p = h_p[{"systematic": "nominal"}].project("ptgen", "mpt_gen")
    h = h_h[{"systematic": "nominal"}].project("ptgen", "mpt_gen")

    p_vals = p[ipt, :].values()
    h_vals = h[ipt, :].values()
    p_errs = np.sqrt(p[ipt, :].variances())
    h_errs = np.sqrt(h[ipt, :].variances())

    p_sum = p_vals.sum()
    h_sum = h_vals.sum()
    if p_sum <= 0 or h_sum <= 0:
        return None

    p_vals, p_errs = p_vals / p_sum, p_errs / p_sum
    h_vals, h_errs = h_vals / h_sum, h_errs / h_sum

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = h_vals / p_vals
        ratio_errs = ratio * np.sqrt(
            (h_errs / h_vals) ** 2 + (p_errs / p_vals) ** 2
        )
    return ratio, ratio_errs


def derive(groomed):
    key = "ptjet_rhojet_g_gen" if groomed else "ptjet_rhojet_u_gen"
    out_pythia = _load(PYTHIA_PKL)
    out_herwig = _load(HERWIG_PKL)
    h_p = out_pythia[key]
    h_h = out_herwig[key]

    # pt edges and rho edges straight from the (shared) gen axes.
    pt_edges = np.asarray(h_p.axes["ptgen"].edges, dtype=float)
    rho_edges = np.asarray(h_p.axes["mpt_gen"].edges, dtype=float)
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    rho_min, rho_max = rho_edges[0], rho_edges[-1]

    npt = len(pt_edges) - 1
    rho_grids, w_grids = [], []
    for ipt in range(npt):
        rho_grid = np.linspace(rho_min, rho_max, N_GRID)
        res = _ratio_vs_rho(h_p, h_h, ipt)
        if res is None:
            # No stats in this pt bin -> neutral (no reweight).
            w_grids.append(np.ones_like(rho_grid))
            rho_grids.append(rho_grid)
            print(f"  pt bin {ipt} [{pt_edges[ipt]:.0f},{pt_edges[ipt+1]:.0f}]: "
                  "empty -> flat weight 1.0")
            continue
        ratio, ratio_errs = res
        mask = (ratio > 0) & (ratio_errs > 0) & np.isfinite(ratio) & np.isfinite(ratio_errs)
        if mask.sum() < 3:
            w_grids.append(np.ones_like(rho_grid))
            rho_grids.append(rho_grid)
            print(f"  pt bin {ipt} [{pt_edges[ipt]:.0f},{pt_edges[ipt+1]:.0f}]: "
                  f"only {int(mask.sum())} good points -> flat weight 1.0")
            continue

        s = mask.sum() * 2  # smoothing strength, as in the zjet notebook
        spl = UnivariateSpline(
            rho_centers[mask], ratio[mask], w=1.0 / ratio_errs[mask], k=2, s=s
        )
        w_grid = np.clip(spl(rho_grid), *W_CLIP)
        rho_grids.append(rho_grid)
        w_grids.append(w_grid)
        print(f"  pt bin {ipt} [{pt_edges[ipt]:.0f},{pt_edges[ipt+1]:.0f}]: "
              f"{int(mask.sum())} points, w in [{w_grid.min():.3f},{w_grid.max():.3f}]")

    return pt_edges, rho_grids, w_grids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for groomed in (False, True):
        label = "groomed" if groomed else "ungroomed"
        print(f"[{label}] deriving dijet h/p spline ...")
        pt_edges, rho_grids, w_grids = derive(groomed)
        assert not np.isnan(np.concatenate(w_grids)).any(), "NaN in weight grid"
        out_path = os.path.join(OUT_DIR, f"dijet_spline_{label}.npz")
        np.savez(
            out_path,
            pt_edges=pt_edges,
            rho_grids=np.array(rho_grids, dtype=object),
            w_grids=np.array(w_grids, dtype=object),
        )
        print(f"  -> wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
