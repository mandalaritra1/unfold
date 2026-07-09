"""Can the groomed second-peak/dip structure at log10(rho^2) in [-3,-1]
survive unfolding, given the detector kernel (RMS ~ 0.5-0.65 there)?

Test: build two truth hypotheses per pT slice
  - DIP     = nominal PYTHIA gen (has the dip at ~ -2.25)
  - DIPLESS = same, but with log-linear interpolation across the dip region
fold the DIP truth, generate data-statistics toys, unfold each toy, and ask
whether the unfolded result separates the two hypotheses:
  dchi2 = chi2(u, DIPLESS) - chi2(u, DIP)   using the toy covariance.
Also: D'Agostini with the DIPLESS *prior* unfolding DIP pseudo-data — does the
dip survive a wrong prior? (bounds the prior-dependence objection).

Run: .venv/bin/python -m scripts.dijet_methods.study2_dip_resolvability
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.dijet_methods.loader import Binning
from scripts.dijet_methods import methods as M
from scripts.dijet_methods.result import prepare

OUT = "outputs/dijet/2018/rho/method_study2/dip_resolvability"
os.makedirs(OUT, exist_ok=True)

RECO_EDGES = np.array([-10, -8, -7, -6, -5.5, -5, -4.75, -4.5, -4.25, -4,
                       -3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75,
                       -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0])
PT_EDGES = np.array([0, 200, 290, 400, 570, 760, 13000.0])

BINNINGS = {
    # tail merged below -4, 0.5-wide bins across the dip region
    "tailmerged_0p5": np.array([-10, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.0]),
    # resolution-matched: 1.0-wide bins in the dip region
    "coarse_1p0": np.array([-10, -4, -3, -2, -1, 0.0]),
    # current study binning (fine tail kept)
    "fine_study": np.array([-10, -6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.0]),
}

DIP_REGION = (-3.0, -1.0)
N_TOYS = 300
RNG = np.random.default_rng(1234)


def make_dipless(prob):
    """Per pT slice, log-linearly interpolate gen across the dip region."""
    gen = prob.gen.copy()
    edges = prob.binning.rho_edges_gen
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    out = gen.copy()
    for pt in np.unique(prob.gen_pt_idx):
        idx = np.where(prob.gen_pt_idx == pt)[0]
        c = centers[prob.gen_rho_idx[idx]]
        w = widths[prob.gen_rho_idx[idx]]
        dens = np.where(gen[idx] > 0, gen[idx] / w, 0)
        inside = (c > DIP_REGION[0]) & (c < DIP_REGION[1]) & (dens > 0)
        left = np.where((c <= DIP_REGION[0]) & (dens > 0))[0]
        right = np.where((c >= DIP_REGION[1]) & (dens > 0))[0]
        if not inside.any() or len(left) == 0 or len(right) == 0:
            continue
        i0, i1 = left[-1], right[0]
        f = (c[inside] - c[i0]) / (c[i1] - c[i0])
        interp = np.exp(np.log(dens[i0]) * (1 - f) + np.log(dens[i1]) * f)
        out[idx[inside]] = interp * w[inside]
    return out


def weighted_lstsq(prob, matched, var):
    v = np.maximum(var, np.maximum(matched, 1.0))
    w = 1.0 / np.sqrt(v)
    keep = matched >= 0  # all rows; empty rows get weight from floor
    A = prob.P * w[:, None]
    b = matched * w
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def reported_mask(prob):
    return prob.gen_pt_idx >= 1


def dip_mask(prob):
    edges = prob.binning.rho_edges_gen
    centers = 0.5 * (edges[:-1] + edges[1:])
    c = centers[prob.gen_rho_idx]
    return (prob.gen_pt_idx >= 1) & (c > DIP_REGION[0]) & (c < DIP_REGION[1])


def run(mode="groomed"):
    results = {}
    for bname, gen_edges in BINNINGS.items():
        prep = prepare(mode, Binning(pt_edges=PT_EDGES, rho_edges_reco=RECO_EDGES,
                                     rho_edges_gen=gen_edges))
        prob = prep.prob
        truth_dip = prob.gen.copy()
        truth_dipless = make_dipless(prob)

        # pseudo-data at real-data statistical power
        pseudo = prob.P @ truth_dip
        scale = prep.data_matched.sum() / pseudo.sum()
        pseudo *= scale
        t_dip = truth_dip * scale
        t_dls = truth_dipless * scale
        var = np.maximum(prep.data_var, 1.0)  # per-bin variance at data stats

        methods = {
            "lstsq_unreg": lambda m, prior=None: weighted_lstsq(prob, m, var),
            "dago4_dipprior": lambda m, prior=None: M.dagostini(prob, m, n_iter=4,
                                                                prior=t_dip),
            "dago4_diplessprior": lambda m, prior=None: M.dagostini(prob, m, n_iter=4,
                                                                    prior=t_dls),
            "dago10_diplessprior": lambda m, prior=None: M.dagostini(prob, m, n_iter=10,
                                                                     prior=t_dls),
        }
        rep = reported_mask(prob)
        dm = dip_mask(prob)
        res_b = {}
        for name, fn in methods.items():
            u0 = fn(pseudo)
            toys = np.empty((N_TOYS, len(u0)))
            for t in range(N_TOYS):
                m_toy = np.clip(pseudo + RNG.normal(size=pseudo.shape) * np.sqrt(var), 0, None)
                toys[t] = fn(m_toy)
            cov = np.cov(toys, rowvar=False)
            # hypothesis separation in the dip bins
            sub = np.where(dm)[0]
            Csub = cov[np.ix_(sub, sub)]
            Cinv = np.linalg.pinv(Csub, rcond=1e-10)
            d_dip = (toys[:, sub] - t_dip[sub])
            d_dls = (toys[:, sub] - t_dls[sub])
            chi2_dip = np.einsum("ti,ij,tj->t", d_dip, Cinv, d_dip)
            chi2_dls = np.einsum("ti,ij,tj->t", d_dls, Cinv, d_dls)
            dchi2 = chi2_dls - chi2_dip
            # bias of central vs dip truth in dip bins
            with np.errstate(divide="ignore", invalid="ignore"):
                bias = np.where(t_dip[rep] != 0, u0[rep] / t_dip[rep] - 1, np.nan)
            spread = np.where(t_dip[rep] != 0,
                              toys[:, rep].std(axis=0) / np.abs(t_dip[rep]), np.nan)
            res_b[name] = dict(
                dchi2_median=float(np.median(dchi2)),
                dchi2_p16=float(np.percentile(dchi2, 16)),
                frac_toys_prefer_dip=float(np.mean(dchi2 > 0)),
                sep_sigma=float(np.sign(np.median(dchi2)) * np.sqrt(abs(np.median(dchi2)))),
                central_bias_median=float(np.nanmedian(np.abs(bias))),
                central_bias_max=float(np.nanmax(np.abs(bias))),
                toy_spread_median=float(np.nanmedian(spread)),
                n_dip_bins=int(dm.sum()),
            )
            print(f"[{bname}] {name}: sep={res_b[name]['sep_sigma']:.1f} sigma "
                  f"(frac_prefer_dip={res_b[name]['frac_toys_prefer_dip']:.2f}) "
                  f"bias med={res_b[name]['central_bias_median']*100:.1f}% "
                  f"max={res_b[name]['central_bias_max']*100:.1f}% "
                  f"spread med={res_b[name]['toy_spread_median']*100:.2f}%")
        results[bname] = res_b

        # overlay figure per binning: truth pair + unfolded (dipless-prior dago + lstsq)
        edges = prob.binning.rho_edges_gen
        widths = edges[1:] - edges[:-1]
        nslices = len(PT_EDGES) - 2
        fig, axs = plt.subplots(1, nslices, figsize=(3.6 * nslices, 3.4), sharey=False)
        for k, pt in enumerate(range(1, nslices + 1)):
            ax = axs[k]
            idx = np.where(prob.gen_pt_idx == pt)[0]
            c = 0.5 * (edges[:-1] + edges[1:])[prob.gen_rho_idx[idx]]
            w = widths[prob.gen_rho_idx[idx]]
            norm = (t_dip[idx]).sum()
            for name, style in [("dago4_diplessprior", dict(fmt="o", color="tab:green")),
                                ("lstsq_unreg", dict(fmt="s", color="tab:red"))]:
                u0 = methods[name](pseudo)
                toys_std = None
                ax.errorbar(c, u0[idx] / w / norm, xerr=w / 2, ls="none",
                            label=name, **style, ms=3)
            ax.stairs(t_dip[idx] / w / norm, edges[np.r_[prob.gen_rho_idx[idx],
                      prob.gen_rho_idx[idx][-1] + 1]], color="k", label="truth (dip)")
            ax.stairs(t_dls[idx] / w / norm, edges[np.r_[prob.gen_rho_idx[idx],
                      prob.gen_rho_idx[idx][-1] + 1]], color="gray", ls="--",
                      label="truth (dipless)")
            ax.set_xlim(-4.5, 0)
            ax.set_title(f"pT slice {pt}", fontsize=9)
            if k == 0:
                ax.legend(fontsize=6)
        fig.suptitle(f"{mode} {bname}: dip recovery (pseudo-data = dip truth)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/dip_recovery_{mode}_{bname}.png", dpi=130)
        plt.close(fig)

    with open(f"{OUT}/metrics_{mode}.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", f"{OUT}/metrics_{mode}.json")


if __name__ == "__main__":
    run("groomed")
