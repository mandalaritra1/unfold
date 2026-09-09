#!/usr/bin/env python3
r"""Quantitative bottom-line test for the Z+jet rho unfolding.

The bottom-line test asks whether unfolding *degrades* our ability to reject a
model: unfolding cannot create information, so

    chi2_unfold  <=  chi2_smeared

must hold, where (TUnfold TWiki / Cowan):

    chi2_smeared = (y - K x')^T V_y^{-1}   (y - K x')      [reco/measured space]
    chi2_unfold  = (xhat - x')^T V_xhat^{-1} (xhat - x')    [unfolded/truth space]

Here the *model* x' is the PYTHIA8 gen prior (``y_true``), K x' is the folded
model = matched reco MC (``mosaic.sum(axis=1)``), y is the fake-subtracted data
(``y_meas``), V_y is the data stat covariance in reco space
(``corrected_measured_variances``, diagonal), xhat is the unfolded data
(``y_unf``) and V_xhat is the data covariance propagated to truth space by
TUnfold (``cov_data_np`` = GetEmatrixInput; ``cov_np`` = GetEmatrixTotal adds
the response-MC stat).

At tau = 0 with a well-conditioned response the expectation tightens to
chi2_unfold ~= chi2_smeared (the propagation is exact), so this is the clean,
principled statement to put in front of the ARC. Systematics do not affect the
validity of the test, so only the statistical covariance is used.

Singular V_xhat (the area constraint kEConstraintArea adds one global zero mode)
is handled with an eigenvalue-truncated pseudo-inverse for the chi2 value. The
unfolded-space ndf follows the CMS Stat Committee / TUnfold TWiki prescription:
the effective (participation-ratio) rank of K J J^T K^T, which drops below the
gen bin count under regularization. Run ``*_noarea`` for a full-rank V_xhat.

Usage:
    source scripts/setup_root.sh
    python scripts/diagnostics/study_bottom_line_chi2.py --tag original
    python scripts/diagnostics/study_bottom_line_chi2.py --tag original_merge25
    python scripts/diagnostics/study_bottom_line_chi2.py --tag original_merge25_noarea
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if os.environ.get("ROOTSYS"):
    sys.path.insert(0, str(Path(os.environ["ROOTSYS"]) / "lib"))

try:
    from scipy.stats import chi2 as _chi2_dist

    def _pvalue(chi2_val, ndf):
        if ndf <= 0:
            return float("nan")
        return float(_chi2_dist.sf(chi2_val, ndf))
except Exception:  # scipy optional
    def _pvalue(chi2_val, ndf):
        return float("nan")


def _offsets(edges_by_pt):
    """Flat start offset of each pt slice given its per-pt edge lists."""
    counts = [len(e) - 1 for e in edges_by_pt]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    return starts, counts


def chi2_dense(residual, cov, rcond=1e-10):
    """chi2 = r^T pinv(cov) r with an eigenvalue-truncated inverse.

    Returns (chi2, ndf) where ndf is the number of retained eigenmodes.
    Symmetric covariance assumed.
    """
    residual = np.asarray(residual, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    cutoff = rcond * evals.max() if evals.size and evals.max() > 0 else 0.0
    keep = evals > cutoff
    if not np.any(keep):
        return float("nan"), 0
    proj = evecs[:, keep].T @ residual
    chi2_val = float(np.sum(proj**2 / evals[keep]))
    return chi2_val, int(keep.sum())


def chi2_diag(residual, variance):
    """chi2 for a diagonal covariance; ndf = number of bins with var > 0."""
    residual = np.asarray(residual, dtype=float)
    variance = np.asarray(variance, dtype=float)
    good = variance > 0
    chi2_val = float(np.sum(residual[good] ** 2 / variance[good]))
    return chi2_val, int(good.sum())


def build_unfolder(tag, groomed):
    import matplotlib

    matplotlib.use("Agg")
    from unfold.tools.unfolder_core import Unfolder, get_spec

    spec = get_spec("zjet", "rho", tag)
    # do_syst=False -> nominal unfold + jackknife stat only; no plots, so the
    # HERWIG-systematic plot (which needs syst inputs) is never reached.
    return Unfolder(spec, groomed, do_syst=False)


def _bin_sel(edges, drop_underflow, drop_last, rho_min):
    """Local indices of the bins to keep for one pt slice.

    ``rho_min`` (when not None) keeps only bins whose *lower* edge is at or
    above rho_min, i.e. it drops every bin lying entirely below rho_min. This
    supersedes ``drop_underflow`` (the rho_min cut already removes the low-rho
    sink). ``drop_last`` still trims the top (rho->0 spike) bin on top.
    """
    n = len(edges) - 1
    idx = list(range(n))
    if rho_min is not None:
        idx = [k for k in idx if float(edges[k]) >= rho_min - 1e-6]
    elif drop_underflow:
        idx = idx[1:]
    if drop_last and idx:
        idx = idx[:-1]
    return idx


def analyze(unf, drop_underflow=True, drop_last=False, use_total_cov=False,
            rho_min=None):
    """Return per-slice and global bottom-line chi2 for an unfolded result."""
    gen_by_pt = unf.gen_edges_by_pt
    reco_by_pt = unf.reco_edges_by_pt
    gen_starts, gen_counts = _offsets(gen_by_pt)
    reco_starts, reco_counts = _offsets(reco_by_pt)

    y_unf = np.asarray(unf.y_unf, dtype=float)
    y_true = np.asarray(unf.y_true, dtype=float)
    y_meas = np.asarray(unf.y_meas, dtype=float)
    reco_mc = np.asarray(unf.mosaic.sum(axis=1), dtype=float)
    var_y = np.asarray(
        getattr(unf, "corrected_measured_variances", None)
        if getattr(unf, "corrected_measured_variances", None) is not None
        else unf.measured_variances,
        dtype=float,
    )
    cov_x = np.asarray(unf.cov_np if use_total_cov else unf.cov_data_np, dtype=float)

    reported = [i for i in unf._reported_pt_indices() if i >= unf.first_reported_pt_bin]
    # zjet: pt bin 0 is the 0-200 GeV migration sink; report 200+ only.
    reported = [i for i in reported if unf.pt_edges[i] >= 200]

    rows = []
    glob_r_unf, glob_r_sm, glob_var = [], [], []
    glob_gen_idx = []  # absolute flat gen indices kept (for the global cov block)
    glob_reco_idx = []  # absolute flat reco indices kept (for effective ndof)
    for i in reported:
        gs, gc = gen_starts[i], gen_counts[i]
        rs, rc = reco_starts[i], reco_counts[i]
        gsel = _bin_sel(gen_by_pt[i], drop_underflow, drop_last, rho_min)
        rsel = _bin_sel(reco_by_pt[i], drop_underflow, drop_last, rho_min)
        gidx = [gs + k for k in gsel]
        ridx = [rs + k for k in rsel]

        r_unf = y_unf[gidx] - y_true[gidx]
        Vx = cov_x[np.ix_(gidx, gidx)]
        c_unf, ndf_unf = chi2_dense(r_unf, Vx)
        # TWiki N_dof for the unfolded space: effective rank of K J J^T K^T.
        eff = unf._effective_ndof(ridx)
        if eff is not None:
            ndf_unf = eff

        r_sm = y_meas[ridx] - reco_mc[ridx]
        c_sm, ndf_sm = chi2_diag(r_sm, var_y[ridx])

        rows.append(
            dict(
                pt=f"{int(unf.pt_edges[i])}-"
                + (f"{int(unf.pt_edges[i+1])}" if i + 1 < len(unf.pt_edges) - 1 else "inf"),
                c_sm=c_sm, ndf_sm=ndf_sm, c_unf=c_unf, ndf_unf=ndf_unf,
            )
        )
        glob_r_unf.append(r_unf)
        glob_r_sm.append(r_sm)
        glob_var.append(var_y[ridx])
        glob_gen_idx.extend(gidx)
        glob_reco_idx.extend(ridx)

    # Global test: one chi2 over all reported bins (the rigorous statement).
    r_unf = np.concatenate(glob_r_unf)
    Vx = cov_x[np.ix_(glob_gen_idx, glob_gen_idx)]
    gc_unf, gndf_unf = chi2_dense(r_unf, Vx)
    g_eff = unf._effective_ndof(glob_reco_idx)
    if g_eff is not None:
        gndf_unf = g_eff
    r_sm = np.concatenate(glob_r_sm)
    gc_sm, gndf_sm = chi2_diag(r_sm, np.concatenate(glob_var))
    glob = dict(c_sm=gc_sm, ndf_sm=gndf_sm, c_unf=gc_unf, ndf_unf=gndf_unf)
    return rows, glob


def _fmt(c, ndf):
    ndf_s = f"{ndf:<5.1f}" if abs(ndf - round(ndf)) > 1e-6 else f"{int(round(ndf)):<5d}"
    if ndf <= 0 or not np.isfinite(c):
        return f"{c:8.2f} / {ndf_s}  (   nan)"
    return f"{c:8.2f} / {ndf_s}  (p={_pvalue(c, ndf):5.3f})"


def print_report(tag, groomed, rows, glob, cov_label):
    mode = "groomed" if groomed else "ungroomed"
    print(f"\n=== {tag}  [{mode}]   V_xhat = {cov_label} ===")
    print(f"{'pt slice':>12} | {'chi2_smeared / ndf':>26} | {'chi2_unfold / ndf':>26} | check")
    print("-" * 86)
    for r in rows:
        ok = "PASS" if (r["c_unf"] <= r["c_sm"] + 1e-9) else "FAIL"
        print(f"{r['pt']:>12} | {_fmt(r['c_sm'], r['ndf_sm']):>26} | "
              f"{_fmt(r['c_unf'], r['ndf_unf']):>26} | {ok}")
    print("-" * 86)
    ok = "PASS" if (glob["c_unf"] <= glob["c_sm"] + 1e-9) else "FAIL"
    print(f"{'GLOBAL':>12} | {_fmt(glob['c_sm'], glob['ndf_sm']):>26} | "
          f"{_fmt(glob['c_unf'], glob['ndf_unf']):>26} | {ok}")
    # chi2/ndf comparison (the headline for the ARC at tau=0).
    su = glob["c_sm"] / max(glob["ndf_sm"], 1)
    uu = glob["c_unf"] / max(glob["ndf_unf"], 1)
    print(f"  global chi2/ndf:  smeared={su:.3f}   unfold={uu:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="original")
    ap.add_argument("--modes", nargs="+", default=["groomed", "ungroomed"],
                    choices=["groomed", "ungroomed"])
    ap.add_argument("--keep-underflow", action="store_true",
                    help="include the [-10,-4.5] low-rho underflow sink bin")
    ap.add_argument("--drop-last", action="store_true",
                    help="also drop the last (rho->0 spike) bin")
    ap.add_argument("--rho-min", type=float, default=None,
                    help="drop every bin lying entirely below this rho value "
                         "(e.g. --rho-min -2.5 for groomed); supersedes the "
                         "index-based --keep-underflow/underflow drop")
    ap.add_argument("--total-cov", action="store_true",
                    help="use GetEmatrixTotal (input+matrix stat) for V_xhat "
                         "instead of GetEmatrixInput (data stat only)")
    args = ap.parse_args()

    cov_label = "cov_np (input+matrix stat)" if args.total_cov else "cov_data_np (data stat)"
    for mode in args.modes:
        groomed = mode == "groomed"
        unf = build_unfolder(args.tag, groomed)
        rows, glob = analyze(
            unf,
            drop_underflow=not args.keep_underflow,
            drop_last=args.drop_last,
            use_total_cov=args.total_cov,
            rho_min=args.rho_min,
        )
        print_report(args.tag, groomed, rows, glob, cov_label)


if __name__ == "__main__":
    main()
