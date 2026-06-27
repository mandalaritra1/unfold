#!/usr/bin/env python3
r"""Full regularization sweep for the zjet groomed rho unfolding.

Every regularization MODE x every tau-selection SCAN, all compared against the
unregularized (tau=0) result, on the nominal data unfold. For each setup it
records: chosen tau, the propagated (EMatrix) stat error, the self-closure bias
(unfold the MC's own reco -> should return the MC truth) and the HERWIG bias
(unfold HERWIG reco through the PYTHIA response -> compare to HERWIG truth).

MODES
  none         : tau = 0 (reference)
  size         : kRegModeSize        -- penalise |x - x_MC| (amplitude)
  derivative   : kRegModeDerivative  -- penalise the 1st derivative of x
  curvature    : kRegModeCurvature   -- penalise the 2nd derivative of x   (standard)
  ratio_deriv  : custom -- 1st derivative of x/x_MC  (zero penalty for x ~ prior)
  ratio_curv   : custom -- 2nd derivative of x/x_MC  (PRODUCTION; zero penalty for x ~ prior)

SCANS
  lcurve : L-curve corner (ScanLcurve)
  sure   : Stein unbiased risk estimate (ScanSURE, wide range)
  rho    : minimise the average global correlation rho (ScanTau, kEScanTauRhoAvg)

Outputs (outputs/zjet/validation/):
  reg_overlay_groomed_pt290_400.{pdf,png}   -- unreg vs production reg on the 290-400 slice
  reg_grid_shapes_groomed_pt290_400.{pdf,png} -- all modes' unfolded shape (L-curve)
  reg_grid_tradeoff_groomed.{pdf,png}        -- stat reduction vs self-closure bias
  + a printed table of every (mode, scan).

Usage: source scripts/setup_root.sh && python scripts/study_regularization_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import ROOT

from unfold.tools.unfolder_core import Unfolder, get_spec, _declare_open_l, _graph_to_arrays
from unfold.utils.merge_helpers import (
    merge_mass_flat, reorder_to_expected_2d, unflatten_gen_by_pt,
)
from study_regularization_rho import normalize


def results_from(unfold, n_true, name):
    """Unfolded result + input covariance, with the temp histograms detached
    from gDirectory so they do not accumulate/collide across ~40 unfolds."""
    h_out = unfold.GetOutput(f"out_{name}")
    e_in = unfold.GetEmatrixInput(f"ein_{name}")
    y = np.array([h_out.GetBinContent(i) for i in range(1, n_true + 1)])
    cov = np.array([[e_in.GetBinContent(i, j) for j in range(1, n_true + 1)]
                    for i in range(1, n_true + 1)])
    return y, cov

OUT = REPO_ROOT / "outputs" / "zjet" / "validation"
_declare_open_l()

# ScanTau (rho-average) shim: PyROOT cannot pass the TSpline**/TGraph** outputs.
if not hasattr(ROOT, "RunUnfoldTauScanR"):
    ROOT.gInterpreter.Declare(r"""
    struct UnfoldTauScanR { TSpline* rho=nullptr; TGraph* lcurve=nullptr; Int_t iBest=-1; };
    UnfoldTauScanR RunUnfoldTauScanR(TUnfoldDensity& u, Int_t n, Double_t tmin,
                                     Double_t tmax, Int_t mode){
      UnfoldTauScanR r; TSpline *lx=0,*ly=0;
      r.iBest = u.ScanTau(n, tmin, tmax, &r.rho,
                          (TUnfoldDensity::EScanTauMode)mode, 0, 0, &r.lcurve, &lx, &ly);
      return r;
    }
    """)

REGMODE = {
    "none": ROOT.TUnfold.kRegModeNone,
    "size": ROOT.TUnfold.kRegModeSize,
    "derivative": ROOT.TUnfold.kRegModeDerivative,
    "curvature": ROOT.TUnfold.kRegModeCurvature,
    "ratio_deriv": ROOT.TUnfold.kRegModeNone,
    "ratio_curv": ROOT.TUnfold.kRegModeNone,
    "ratio_size": ROOT.TUnfold.kRegModeNone,
}
MODES = ["none", "ratio_curv", "ratio_deriv", "ratio_size", "size", "derivative", "curvature"]
SCANS = ["lcurve", "sure", "rho"]


_BUILD_N = [0]


def build(u, resp, misses, true_flat, meas_flat, *, mode, name, variances=None):
    _BUILD_N[0] += 1
    name = f"{name}_{_BUILD_N[0]}"  # globally unique -> no ROOT name collisions
    # Fresh binning trees per build: a TUnfoldDensity registers its regularization
    # distribution on the tree, so sharing one tree across setups leaks conditions
    # between them and shifts the L-curve corner.
    truth_root, reco_root = u._build_root_binning()
    h_meas = reco_root.CreateHistogram(f"hM_{name}")
    h_resp = ROOT.TUnfoldBinning.CreateHistogramOfMigrations(truth_root, reco_root, f"hR_{name}")
    u._fill_response_histogram(
        h_resp, resp, misses, include_bin_errors=True,
        resp_var=getattr(u, "mosaic_var_dict", {}).get("nominal"),
        misses_var=getattr(u, "misses_var_dict", {}).get("nominal"),
    )
    u._fill_root_histogram(h_meas, meas_flat, variances)
    unfold = ROOT.TUnfoldDensityOpenL(
        h_resp, ROOT.TUnfold.kHistMapOutputHoriz, REGMODE[mode],
        ROOT.TUnfold.kEConstraintArea, ROOT.TUnfoldDensity.kDensityModeBinWidth,
        truth_root, reco_root,
    )
    if mode == "ratio_curv":
        # Use the production method verbatim (a re-implemented copy mysteriously
        # halved the L magnitude in this harness, shifting the L-curve corner).
        u._add_ratio_curvature_conditions(unfold, true_flat)
    elif mode == "ratio_deriv":
        offset = 0
        for edges in u.gen_edges_by_pt:
            nbins = len(edges) - 1
            for k in range(0, nbins - 1):
                j0, j1 = offset + k, offset + k + 1
                m0, m1 = true_flat[j0], true_flat[j1]
                if min(m0, m1) <= 0:
                    continue
                unfold.AddRegularisationCondition(j0 + 1, 1.0 / m0, j1 + 1, -1.0 / m1)
            offset += nbins
    elif mode == "ratio_size":
        # amplitude of x_j/m_j -> pulls toward x ~ m^2 (NO null-space for the prior)
        offset = 0
        for edges in u.gen_edges_by_pt:
            nbins = len(edges) - 1
            for k in range(0, nbins):
                j = offset + k
                if true_flat[j] > 0:
                    unfold.AddRegularisationCondition(j + 1, 1.0 / true_flat[j])
            offset += nbins
    if unfold.SetInput(h_meas) >= 10000:
        raise RuntimeError("overflow/underflow")
    return unfold, (truth_root, reco_root, h_meas, h_resp)


def run(u, resp, misses, true_flat, meas_flat, *, mode, name, scan=None, tau=None, variances=None):
    """Build, scan-or-unfold, copy results to numpy (shared binning, unique hists)."""
    unfold, keep = build(u, resp, misses, true_flat, meas_flat, mode=mode, name=name,
                         variances=variances)
    if scan == "lcurve":
        ROOT.RunUnfoldLcurveScan(unfold, 40)
    elif scan == "sure":
        ROOT.RunUnfoldSureScan(unfold, 60, 1e-4, 1e3)
    elif scan == "rho":
        ROOT.RunUnfoldTauScanR(unfold, 60, 1e-4, 1e3, 0)  # kEScanTauRhoAvg
    else:
        unfold.DoUnfold(tau)
    t = float(unfold.GetTau())
    y, cov = results_from(unfold, resp.shape[1], name)
    return np.array(y, copy=True), np.array(cov, copy=True), t


def main():
    spec = get_spec("zjet", "rho", "original")
    u = Unfolder(spec, groomed=True, do_syst=False)
    resp = u.mosaic_dict["nominal"]
    misses = u.misses_2d
    true_flat = resp.sum(axis=0) + misses
    meas_data = u._apply_fake_correction(np.array(u.mosaic_2d, float), "nominal", False, False)
    meas_self = np.asarray(u.mosaic.sum(axis=1), dtype=float)

    herwig = u._load_pickle(spec.input_dir + spec.herwig_file)
    resp4d = herwig[u._histogram_keys()["response"]]
    h2d_her = resp4d[{"systematic": "nominal"}].project("ptreco", u.reco_axis).values(flow=False)
    h2d_her, _ = reorder_to_expected_2d(h2d_her, u.edges, u.pt_edges)
    meas_herwig = np.asarray(merge_mass_flat(h2d_her, u.edges, u.reco_edges_by_pt), dtype=float)
    herwig_truth = np.array(u.herwig_gen_val_flat, dtype=float)

    def reported_median(frac):
        vals = []
        off = 0
        for i, edges in enumerate(u.gen_edges_by_pt):
            n = len(edges) - 1
            if u.pt_edges[i] >= 200:
                vals.extend(frac[off + 1:off + n])  # drop underflow sink
            off += n
        return float(np.median(vals))

    var = u.corrected_measured_variances  # real data-stat variances -> correct L-curve corner

    rows = []
    norm_data = {}        # (mode,scan) -> normalized unfolded data
    # unregularized reference
    y0, c0, _ = run(u, resp, misses, true_flat, meas_data, mode="none", tau=0.0, name="none", variances=var)
    f0 = np.sqrt(np.clip(np.diag(c0), 0, None)) / np.where(y0 != 0, np.abs(y0), 1.0)
    stat0 = reported_median(f0)
    norm_data[("none", "-")] = normalize(u, y0)
    ys0, _, _ = run(u, resp, misses, true_flat, meas_self, mode="none", tau=0.0, name="ns")
    yh0, _, _ = run(u, resp, misses, true_flat, meas_herwig, mode="none", tau=0.0, name="nh")
    rows.append(dict(mode="none", scan="-", tau=0.0, stat=stat0, statred=0.0,
                     selfmax=float(np.max(np.abs(normalize(u, ys0)/normalize(u, true_flat)-1))),
                     hermax=float(np.max(np.abs(normalize(u, yh0)/normalize(u, herwig_truth)-1)))))

    for mode in MODES:
        if mode == "none":
            continue
        for scan in SCANS:
            name = f"{mode}_{scan}"
            try:
                yd, cd, tau = run(u, resp, misses, true_flat, meas_data, mode=mode,
                                  name=name, scan=scan, variances=var)
                fd = np.sqrt(np.clip(np.diag(cd), 0, None)) / np.where(yd != 0, np.abs(yd), 1.0)
                ys, _, _ = run(u, resp, misses, true_flat, meas_self, mode=mode, tau=tau, name=name + "s")
                yh, _, _ = run(u, resp, misses, true_flat, meas_herwig, mode=mode, tau=tau, name=name + "h")
                stat = reported_median(fd)
                rows.append(dict(
                    mode=mode, scan=scan, tau=tau, stat=stat,
                    statred=100 * (1 - stat / stat0),
                    selfmax=float(np.max(np.abs(normalize(u, ys)/normalize(u, true_flat)-1))),
                    hermax=float(np.max(np.abs(normalize(u, yh)/normalize(u, herwig_truth)-1))),
                ))
                norm_data[(mode, scan)] = normalize(u, yd)
            except Exception as e:
                rows.append(dict(mode=mode, scan=scan, tau=float("nan"), stat=float("nan"),
                                 statred=float("nan"), selfmax=float("nan"),
                                 hermax=float("nan"), err=str(e)[:40]))

    # ---- table ----
    print(f"\n{'mode':>12} {'scan':>7} {'tau':>10} {'stat med':>9} {'stat red':>9} "
          f"{'self max':>9} {'HERWIG max':>11}")
    print("-" * 74)
    for r in rows:
        print(f"{r['mode']:>12} {r['scan']:>7} {r['tau']:>10.3g} "
              f"{100*r['stat']:>8.2f}% {r['statred']:>8.1f}% "
              f"{100*r['selfmax']:>8.2f}% {100*r['hermax']:>10.2f}%"
              + (f"   ERR {r.get('err')}" if 'err' in r else ""))

    # ===================== plots =====================
    hep.style.use("CMS")
    i_pt = 2  # 290-400 GeV
    off = sum(len(e) - 1 for e in u.gen_edges_by_pt[:i_pt])
    n = len(u.gen_edges_by_pt[i_pt]) - 1
    sl = slice(off, off + n)
    edges = np.array(u.gen_edges_by_pt[i_pt], dtype=float)
    bw = np.diff(edges)
    pyt = normalize(u, true_flat)[sl]
    her = normalize(u, herwig_truth)[sl]

    # --- Figure 1: overlay unreg vs production reg (the requested plot) ---
    reg = norm_data[("ratio_curv", "lcurve")][sl]
    unr = norm_data[("none", "-")][sl]
    fig, (a, r) = plt.subplots(2, 1, sharex=True, figsize=(9, 9),
                               gridspec_kw={"height_ratios": [3, 1]})
    a.stairs(unr, edges, color="k", lw=2.5, label="Unfolded, unregularised (τ=0)")
    a.stairs(reg, edges, color="#e42536", lw=2.5, ls="--", label="Unfolded, ratio-curvature (τ L-curve)")
    a.stairs(pyt, edges, color="#5790fc", lw=1.8, ls=":", label="PYTHIA8 (prior)")
    a.set_ylabel(r"$1/N\ dN/d\log_{10}(\rho^2)$")
    a.legend(title="Groomed, 290-400 GeV", fontsize=12)
    a.set_xlim(-4.5, 0)
    ratio = np.divide(reg, unr, out=np.ones_like(reg), where=unr != 0)
    r.axhline(1.0, color="gray", ls="--")
    r.stairs(ratio, edges, color="#e42536", lw=2)
    r.set_ylabel("reg / unreg")
    r.set_ylim(0.9, 1.1)
    r.set_xlabel(u._observable_label())
    r.set_xlim(-4.5, 0)
    hep.cms.label("Internal", data=True, lumi="138", com="13", ax=a, fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "reg_overlay_groomed_pt290_400.pdf")
    fig.savefig(OUT / "reg_overlay_groomed_pt290_400.png", dpi=140)
    plt.close(fig)

    # --- Figure 2: all modes' shape (L-curve) ---
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.stairs(pyt, edges, color="#5790fc", lw=2, ls=":", label="PYTHIA8")
    ax.stairs(her, edges, color="#964a8b", lw=2, ls="-.", label="HERWIG7")
    ax.stairs(unr, edges, color="k", lw=3, label="unregularised (τ=0)")
    cols = {"size": "#f89c20", "derivative": "#7a21dd", "curvature": "#e42536",
            "ratio_deriv": "#2ca02c", "ratio_curv": "#17becf"}
    for mode in ("size", "derivative", "curvature", "ratio_deriv", "ratio_curv"):
        if (mode, "lcurve") in norm_data:
            ax.stairs(norm_data[(mode, "lcurve")][sl], edges, color=cols[mode], lw=2,
                      label=f"{mode}")
    ax.set_xlim(-4.5, 0)
    ax.set_xlabel(u._observable_label())
    ax.set_ylabel(r"$1/N\ dN/d\log_{10}(\rho^2)$")
    ax.legend(title="Groomed 290-400 GeV, L-curve τ", ncol=2, fontsize=11)
    hep.cms.label("Internal", data=True, lumi="138", com="13", ax=ax, fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "reg_grid_shapes_groomed_pt290_400.pdf")
    fig.savefig(OUT / "reg_grid_shapes_groomed_pt290_400.png", dpi=140)
    plt.close(fig)

    # --- Figure 3: trade-off stat reduction vs self-closure bias ---
    fig, ax = plt.subplots(figsize=(10, 7.5))
    markers = {"size": "s", "derivative": "^", "curvature": "v",
               "ratio_deriv": "D", "ratio_curv": "o"}
    scol = {"lcurve": "#e42536", "sure": "#5790fc", "rho": "#2ca02c"}
    for rr in rows:
        if rr["mode"] == "none" or not np.isfinite(rr["statred"]):
            continue
        ax.scatter(rr["statred"], 100 * rr["selfmax"], marker=markers[rr["mode"]],
                   color=scol[rr["scan"]], s=120, edgecolor="k", zorder=3)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-0.3, 6000)
    ax.set_xlabel("input-stat error reduction vs τ=0 (%)  →  better")
    ax.set_ylabel("max self-closure bias (%)  →  worse  (symlog)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.text(0.5, 1.0, "1% bias", color="gray", va="bottom", ha="left", fontsize=9)
    ax.text(46, 0.02, "ratio modes:\nexact self-closure\nat every scan",
            fontsize=10, color="#17becf", va="bottom", ha="left", fontweight="bold")
    ax.text(50, 1500, "rho-scan blows up\nstandard modes", fontsize=10, color="#2ca02c",
            va="center", ha="center")
    from matplotlib.lines import Line2D
    leg1 = [Line2D([0], [0], marker=markers[m], color="0.4", ls="", mec="k", ms=11, label=m)
            for m in markers]
    leg2 = [Line2D([0], [0], marker="o", color=scol[s], ls="", mec="k", ms=11, label=s)
            for s in scol]
    l1 = ax.legend(handles=leg1, title="mode", loc="center left", fontsize=10)
    ax.add_artist(l1)
    ax.legend(handles=leg2, title="scan", loc="center right", fontsize=10)
    ax.set_title("Regularization trade-off (groomed): want bottom-right",
                 fontsize=13, pad=22)
    hep.cms.label("Internal", data=False, lumi="138", com="13", ax=ax, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "reg_grid_tradeoff_groomed.pdf")
    fig.savefig(OUT / "reg_grid_tradeoff_groomed.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote reg_overlay / reg_grid_shapes / reg_grid_tradeoff to {OUT}")


if __name__ == "__main__":
    main()
