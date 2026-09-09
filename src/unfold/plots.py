"""Figures.  Every function takes a run ``Unfolder`` (``u``) and draws from its state.

The plot code reads results only; nothing here changes the physics state of
``u``.  ``run_all_plots`` is the full Z+jet suite, ``run_channel_plots`` the
subset used for the prepared-input channels.  Figures land under
``u.spec.output_dir`` in the categorized layout of ``Unfolder._categorize_output``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import numpy as np
import ROOT
from scipy.stats import chi2 as scipy_chi2

from unfold.cms_plot import (
    PUB_ANNOTATION_FONTSIZE, PUB_LABEL_FONTSIZE, PUB_LEGEND_FONTSIZE, PUB_TICK_FONTSIZE, stamp_figure,
)
from unfold.histmath import unflatten_gen_by_pt


def plot_fakes_misses(u, show=True):
    title_list = []
    npt = len(u.pt_edges) - 1
    for i in range(npt):
        lo = int(u.pt_edges[i])
        if i + 1 < npt:
            hi = int(u.pt_edges[i + 1])
            title_list.append(rf"{lo} $<$ $p_T$ $<$ {hi} GeV")
        else:
            title_list.append(rf"{lo} $<$ $p_T$ $< \, \infty$ GeV")

    # Use the same MC fake fraction used by the unfolding fake correction.
    # Dividing by data here can produce infinities in sparse rho tail bins.
    fakerate = np.asarray(u.fake_fraction_2d, dtype=float)
    efficiency = 1 - (u.misses_2d/(u.misses_2d + u.mosaic.sum(axis=0)))
    efficiency_pt_binned = unflatten_gen_by_pt(efficiency, u.gen_edges_by_pt)
    fakerate_pt_binned = unflatten_gen_by_pt(fakerate, u.reco_edges_by_pt)

    _ratio_unc = u._subset_fraction_unc

    # MC-stat error bars from the pkl-hist sumw2 (when available).
    fakerate_unc_pt_binned = efficiency_unc_pt_binned = None
    mosaic_var = getattr(u, "mosaic_var_dict", {}).get("nominal")
    misses_var = getattr(u, "misses_var_dict", {}).get("nominal")
    fakes_var = getattr(u, "fakes_2d_var", None)
    if mosaic_var is not None and fakes_var is not None:
        matched_reco = u.mosaic.sum(axis=1)
        fakerate_unc = _ratio_unc(
            u.fakes_2d, matched_reco, fakes_var, mosaic_var.sum(axis=1)
        )
        fakerate_unc_pt_binned = unflatten_gen_by_pt(fakerate_unc, u.reco_edges_by_pt)
    if mosaic_var is not None and misses_var is not None:
        matched_gen = u.mosaic.sum(axis=0)
        efficiency_unc = _ratio_unc(
            u.misses_2d, matched_gen, misses_var, mosaic_var.sum(axis=0)
        )
        efficiency_unc_pt_binned = unflatten_gen_by_pt(efficiency_unc, u.gen_edges_by_pt)

    for i in u._reported_pt_indices():
        hep.histplot(
            1 - fakerate_pt_binned[i],
            u.reco_edges_by_pt[i],
            yerr=fakerate_unc_pt_binned[i] if fakerate_unc_pt_binned is not None else None,
            label=r"$1-$ fake rate $= N_{\mathrm{gen}\wedge\mathrm{reco}}/N_{\mathrm{reco}}$",
            lw=1.5,
            histtype="step",
        )
        hep.histplot(
            efficiency_pt_binned[i],
            u.gen_edges_by_pt[i],
            yerr=efficiency_unc_pt_binned[i] if efficiency_unc_pt_binned is not None else None,
            label=r"Acceptance $\times$ efficiency $= N_{\mathrm{gen}\wedge\mathrm{reco}}/N_{\mathrm{gen}}$",
            lw=1.5,
            histtype="step",
        )
        plt.legend(title = title_list[i])
        plt.xlabel(u._observable_short_label())
        plt.xlim(*u._display_xlim(i))
        plt.ylim(0,1.05)
        hep.cms.label(
            u.cms_label,
            data=False,
            lumi=u._lumi_label(),
            com=u._com_label(),
            fontsize=20,
        )
        if u.groomed:
            save_path = f"./{u.spec.output_dir}fakerates_groomed_{u._output_panel_index(i)}.pdf"
        else:
            save_path = f"./{u.spec.output_dir}fakerates_ungroomed_{u._output_panel_index(i)}.pdf"
        u._finalize_plot(save_path=save_path, show=show)


def plot_purity_stability(u, show=True):
    """Per-gen-bin purity and stability of the matched response.

    purity_j    = fraction of events reconstructed in bin j that were
                  also generated in bin j;
    stability_j = fraction of events generated in bin j that were also
                  reconstructed in bin j (matched events only).
    """
    title_list = [""]
    npt = len(u.pt_edges) - 1
    for i in range(1, npt):
        lo = int(u.pt_edges[i])
        if i + 1 < npt:
            hi = int(u.pt_edges[i + 1])
            title_list.append(rf"{lo} $<$ $p_T$ $<$ {hi} GeV")
        else:
            title_list.append(rf"{lo} $<$ $p_T$ $< \, \infty$ GeV")

    compressed = u._gen_binned_migration()
    diagonal = np.diag(compressed)
    reco_totals = compressed.sum(axis=1)
    gen_totals = compressed.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.divide(
            diagonal, reco_totals, out=np.zeros_like(diagonal), where=reco_totals > 0
        )
        stability = np.divide(
            diagonal, gen_totals, out=np.zeros_like(diagonal), where=gen_totals > 0
        )
    purity_pt_binned = unflatten_gen_by_pt(purity, u.gen_edges_by_pt)
    stability_pt_binned = unflatten_gen_by_pt(stability, u.gen_edges_by_pt)

    # MC-stat error bars: compress the sumw2 mosaic the same way, then
    # binomial-style propagation (the diagonal is a disjoint subset of
    # the row/column totals).
    purity_unc_pt_binned = stability_unc_pt_binned = None
    mosaic_var = getattr(u, "mosaic_var_dict", {}).get("nominal")
    if mosaic_var is not None:
        compressed_var = u._gen_binned_migration(mosaic_var)
        diag_var = np.diag(compressed_var)
        purity_unc = u._subset_fraction_unc(
            diagonal, reco_totals - diagonal,
            diag_var, compressed_var.sum(axis=1) - diag_var,
        )
        stability_unc = u._subset_fraction_unc(
            diagonal, gen_totals - diagonal,
            diag_var, compressed_var.sum(axis=0) - diag_var,
        )
        purity_unc_pt_binned = unflatten_gen_by_pt(purity_unc, u.gen_edges_by_pt)
        stability_unc_pt_binned = unflatten_gen_by_pt(stability_unc, u.gen_edges_by_pt)

    for i in u._reported_pt_indices():
        hep.histplot(
            purity_pt_binned[i],
            u.gen_edges_by_pt[i],
            yerr=purity_unc_pt_binned[i] if purity_unc_pt_binned is not None else None,
            label="Purity",
            lw=1.5,
            histtype="step",
        )
        hep.histplot(
            stability_pt_binned[i],
            u.gen_edges_by_pt[i],
            yerr=stability_unc_pt_binned[i] if stability_unc_pt_binned is not None else None,
            label="Stability",
            lw=1.5,
            histtype="step",
        )
        plt.axhline(0.5, color="gray", ls="dotted", lw=1)
        plt.legend(title=title_list[i])
        plt.xlabel(u._observable_short_label())
        plt.ylabel("Purity / Stability")
        plt.xlim(*u._display_xlim(i))
        plt.ylim(0, 1.05)
        hep.cms.label(
            u.cms_label,
            data=False,
            lumi=u._lumi_label(),
            com=u._com_label(),
            fontsize=20,
        )
        suffix = "groomed" if u.groomed else "ungroomed"
        panel_index = u._output_panel_index(i)
        save_path = (
            f"./{u.spec.output_dir}purity_stability_{suffix}_{panel_index}.pdf"
        )
        u._finalize_plot(save_path=save_path, show=show)


def plot_input_data_mc(u, show=True):
    """Reco-level overlay of Data vs PYTHIA vs HERWIG per pt bin, with a ratio pad."""
    hep.style.use("CMS")
    pt_edges = np.asarray(u.bins.pt_edges, dtype=float)
    mode = "groomed" if u.groomed else "ungroomed"
    x_label = u.spec.x_label_groomed if u.groomed else u.spec.x_label_ungroomed

    def _project_nominal(hist_2d):
        if "systematic" in hist_2d.axes.name:
            return hist_2d.project("ptreco", u.reco_axis, "systematic")[:, :, "nominal"]
        return hist_2d.project("ptreco", u.reco_axis)

    h_data = _project_nominal(u.data_2d)
    h_mc = _project_nominal(u.pythia_2d)
    h_her = _project_nominal(u.herwig_2d)

    pt_axis_edges = np.asarray(h_data.axes['ptreco'].edges, dtype=float)
    reco_axis_edges = np.asarray(h_data.axes[u.reco_axis].edges, dtype=float)

    def _slice(h, lo, hi):
        vals = np.asarray(h.values(), dtype=float)
        raw_var = h.variances()
        if raw_var is None:
            # Fallback for histograms that do not store variances explicitly.
            vars_ = np.clip(vals, 0.0, None)
        else:
            vars_ = np.asarray(raw_var, dtype=float)
        i_lo = int(np.searchsorted(pt_axis_edges, lo - 1e-9, side='left'))
        if np.isfinite(hi):
            i_hi = int(np.searchsorted(pt_axis_edges, hi - 1e-9, side='left'))
        else:
            i_hi = len(pt_axis_edges) - 1
        return vals[i_lo:i_hi, :].sum(axis=0), vars_[i_lo:i_hi, :].sum(axis=0)

    n_pt = len(pt_edges) - 1
    for bin_idx in range(n_pt):
        lo, hi = pt_edges[bin_idx], pt_edges[bin_idx + 1]
        pt_label = (rf"{lo:g}$-\infty$ GeV" if (not np.isfinite(hi) or hi >= 13000)
                    else f"{lo:g}-{hi:g} GeV")

        data_vals, data_var = _slice(h_data, lo, hi)
        mc_vals,   mc_var   = _slice(h_mc,   lo, hi)
        her_vals,  her_var  = _slice(h_her,  lo, hi)
        data_errs = np.sqrt(data_var)
        mc_errs   = np.sqrt(mc_var)
        her_errs  = np.sqrt(her_var)
        edges = reco_axis_edges

        # Normalize over the SHOWN reco window when the spec reports/normalizes
        # over the shown space (production default), so the displayed spectrum
        # integrates to 1 over exactly what is drawn; otherwise full range.
        # mask on the fine reco axis these hists are stored with (the per-pT
        # analysis mask has a different length and crashed here)
        shown = u._normalization_mask(reco_axis_edges, bin_idx)
        for vals, errs in ((data_vals, data_errs), (mc_vals, mc_errs), (her_vals, her_errs)):
            s = vals[shown].sum()
            if s != 0:
                vals /= s
                errs /= s

        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        plt.sca(ax_main)
        hep.histplot(data_vals, edges, yerr=data_errs, label="Data",    color="black")
        hep.histplot(mc_vals,   edges, yerr=mc_errs,   label="PYTHIA8", color="red")
        hep.histplot(her_vals,  edges, yerr=her_errs,  label="HERWIG7", color="blue", alpha=0.7)
        ax_main.set_ylabel("Normalized entries")
        ax_main.legend(title=pt_label)
        hep.cms.label(u.cms_label, data=True, lumi=138, com=13, fontsize=20, ax=ax_main)

        safe_d = np.where(data_vals != 0, data_vals, 1.0)
        safe_m = np.where(mc_vals != 0, mc_vals, 1.0)
        safe_h = np.where(her_vals != 0, her_vals, 1.0)
        r_mc = np.divide(mc_vals,  data_vals, out=np.zeros_like(data_vals), where=data_vals != 0)
        r_h  = np.divide(her_vals, data_vals, out=np.zeros_like(data_vals), where=data_vals != 0)
        r_mc_err = r_mc * np.sqrt((data_errs / safe_d) ** 2 + (mc_errs  / safe_m) ** 2)
        r_h_err  = r_h  * np.sqrt((data_errs / safe_d) ** 2 + (her_errs / safe_h) ** 2)

        plt.sca(ax_ratio)
        hep.histplot(r_mc, edges, yerr=r_mc_err, label="PYTHIA/Data", color="red",  ax=ax_ratio)
        hep.histplot(r_h,  edges, yerr=r_h_err,  label="HERWIG/Data", color="blue", ls="--", alpha=0.7, ax=ax_ratio)
        ax_ratio.axhline(1, color="gray", ls="--")
        ax_ratio.set_xlabel(x_label)
        ax_ratio.set_ylabel("Theory / Data")
        ax_ratio.set_ylim(0, 2)

        # Trim the display to the SHOWN rho window for this pT slice (drop the
        # low-rho migration buffer that is only there to catch feed-in). The
        # shared x-axis carries the limit to both pads.
        floor = u._bl_shown_floors()[bin_idx]
        ax_ratio.set_xlim(floor, reco_axis_edges[-1])

        save_path = f"./{u.spec.output_dir}input_{mode}_{bin_idx}.pdf"
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_L(u, show=True):
    lMatrix = u.L
    #try plotting the L matrix root way
    c = ROOT.TCanvas("c", "L-curve Matrix", 800, 600)
    lMatrix.Draw("colz")
    l_root_path = u._relocate_output("L_matrix_root.png")
    l_root_path.parent.mkdir(parents=True, exist_ok=True)
    c.SaveAs(str(l_root_path))
    nx, ny = lMatrix.GetNbinsX(), lMatrix.GetNbinsY() 
    l_np = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            l_np[i, j] = lMatrix.GetBinContent(i + 1, j + 1)
    #mask zeros for better visualization
    l_np_masked = np.ma.masked_where(l_np == 0, l_np)
    plt.imshow(l_np_masked, origin='lower', aspect='auto')
    plt.colorbar(label='L-curve Matrix Value')
    hep.cms.label(u.cms_label, data = False, lumi = 138 , com = 13, fontsize = 20)
    u._finalize_plot(save_path=f"{u.spec.output_dir}unfold/L_matrix_matplotlib.png", show=show)


def plot_folded(u, show=True, *, counts=False):
    """Plot measured data against the refolded result.

    ``counts=False`` retains the historical, independently normalized
    shape comparison.  The opt-in ``counts=True`` path is for prepared
    pair-split inputs: it compares fake-corrected measured counts to
    ``x_folded`` on the native reco binning and uses the diagonal of the
    fake-corrected measured covariance for the data error bars.
    """
    folded_pt_binned = unflatten_gen_by_pt(u.x_folded, u.reco_edges_by_pt)
    measured_pt_binned = unflatten_gen_by_pt(u.y_meas, u.reco_edges_by_pt)
    if not counts:
        reco_mc_pt_binned = unflatten_gen_by_pt(
            u.mosaic.sum(axis=1), u.reco_edges_by_pt
        )
    for i in u._reported_pt_indices():
        # two-panel plot: main + ratio
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        if counts:
            payload = u._folded_counts_payload(i)
            edges = payload["edges"]
            folded = payload["folded"]
            meas = payload["measured"]
            meas_err = payload["measured_error"]
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_top.stairs(
                folded, edges, label="Nominal refolded MC", color="#e42536",
                ls="dotted", lw=3,
            )
            ax_top.errorbar(
                centers, meas, yerr=meas_err, xerr=np.diff(edges) / 2,
                fmt="o", color="k", label="Fake-corrected measured data",
            )
            ratio = np.divide(
                meas, folded, out=np.full_like(meas, np.nan), where=folded != 0
            )
            ratio_err = np.divide(
                meas_err, np.abs(folded), out=np.full_like(meas_err, np.nan),
                where=folded != 0,
            )
            ax_top.set_ylabel(r"Events / unit $\log_{10}(\rho^2)$")
        else:
            bin_widths_reco = np.diff(u.reco_edges_by_pt[i])
            edges = np.array(u.reco_edges_by_pt[i], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            folded = folded_pt_binned[i] / bin_widths_reco / folded_pt_binned[i].sum()
            meas = measured_pt_binned[i] / bin_widths_reco / measured_pt_binned[i].sum()
            hep.histplot(
                folded, edges, label="Folded", color="#e42536", alpha=0.8,
                ls="dotted", lw=3, ax=ax_top,
            )
            hep.histplot(
                meas, edges, color="k", ls="--", alpha=1,
                label="Measured Data", ax=ax_top,
            )
            ratio = np.divide(
                meas, folded, out=np.full_like(meas, np.nan), where=folded != 0
            )
            ratio_err = None

        ax_bot.axhline(1.0, color='gray', ls='--')
        ax_bot.errorbar(centers, ratio, yerr=ratio_err, fmt='o', color='k')
        ax_bot.set_ylabel('Data / Folded')
        ax_bot.set_xlim(*u._display_xlim(i) if counts else (edges[0], edges[-1]))
        ax_bot.set_ylim(0.5, 1.5)
        ax_bot.set_xlabel(u._observable_label())

        title = f"pT bin: {int(u.pt_edges[i])}-{int(u.pt_edges[i+1]) if i+1 < len(u.pt_edges)-1 else '∞'} GeV"
        ax_top.legend(title=title)
        ax_top.set_xlim(*u._display_xlim(i) if counts else u._observable_xlim(i))
        hep.cms.label(
            u.cms_label, data=True, lumi=u._lumi_label(),
            com=u._com_label(), fontsize=20, ax=ax_top,
        )
        panel_index = u._output_panel_index(i)
        mode = "groomed" if u.groomed else "ungroomed"
        save_path = f"./{u.spec.output_dir}unfold/folded_{mode}_{panel_index}.pdf"
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_jk(u, show= True):
    # Outputs
    n_pt_bins = len(u.pt_edges) - 1
    jk_pt_binned = [
        unflatten_gen_by_pt(sample, u.gen_edges_by_pt)
        for sample in u.y_unf_jk_input_list
    ]

    for pt_index in range(n_pt_bins):
        fig, ax = plt.subplots()
        for jk_index, unfolded_pt_binned in enumerate(jk_pt_binned):
            ax.stairs(
                unfolded_pt_binned[pt_index],
                u.gen_edges_by_pt[pt_index],
                label=f"JK sample {jk_index}",
                alpha=0.6,
            )

        pt_low = int(u.pt_edges[pt_index])
        pt_high = u.pt_edges[pt_index + 1]
        pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

        ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
        
        ax.set_ylabel("Unfolded entries")
        if u.groomed:
            ax.set_xlim(*u._observable_xlim(pt_index))
            ax.set_xlabel(u._observable_label())
            save_path = f"./{u.spec.output_dir}unfold/jk_outputs_groomed_pt{pt_index-1}.pdf"
        else:
            ax.set_xlim(*u._observable_xlim(pt_index))
            ax.set_xlabel(u._observable_label())
            save_path = f"./{u.spec.output_dir}unfold/jk_outputs_ungroomed_pt{pt_index-1}.pdf"

        plt.sca(ax)
        hep.cms.label(u.cms_label, data=True, lumi=138, com=13, fontsize=20)
        plt.tight_layout()
        u._finalize_plot(save_path=save_path, show=show, fig=fig)
    # Inputs
    for pt_index in range(n_pt_bins):
        fig, ax = plt.subplots()
        for jk_index, mosaic_2d_jk in enumerate(u.mosaic_2d_jk_list):
            reco_pt_binned_jk = unflatten_gen_by_pt(u.mosaic_2d_jk_list[jk_index], u.reco_edges_by_pt)
            ax.stairs(
                reco_pt_binned_jk[pt_index],
                u.reco_edges_by_pt[pt_index],
                label=f"JK sample {jk_index}",
                alpha=0.6,
            )

        pt_low = int(u.pt_edges[pt_index])
        pt_high = u.pt_edges[pt_index + 1]
        pt_label = f"{pt_low}–∞ GeV" if pt_index == n_pt_bins - 1 else f"{pt_low}–{int(pt_high)} GeV"

        ax.legend(title=pt_label, fontsize=12, title_fontsize=14)
        
        ax.set_ylabel("Entries")
        if u.groomed:
            ax.set_xlim(*u._observable_xlim(pt_index))
            ax.set_xlabel(u._observable_label())
            save_path = f"./{u.spec.output_dir}unfold/jk_inputs_groomed_pt{pt_index-1}.pdf"
        else:
            ax.set_xlim(*u._observable_xlim(pt_index))
            ax.set_xlabel(u._observable_label())
            save_path = f"./{u.spec.output_dir}unfold/jk_inputs_ungroomed_pt{pt_index-1}.pdf"

        plt.sca(ax)
        hep.cms.label(u.cms_label, data=False, lumi=138, com=13, fontsize=20)
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_bottom_line(u, show=True, rebin_reco_to_gen=False,
                     annotate_chi2=True, include_herwig_band=False,
                     ratio_ylim=(0.0, 2.0),
                     show_isr_fsr_envelope=False):
    # This remains a visual diagnostic.  The quantitative test is evaluated
    # in the full unnormalised space below and printed on every panel so the
    # plot cannot be mistaken for a bin-by-bin chi-square test.
    # Evaluate the test only over the SHOWN space (each slice up to the rho
    # value it is cut at for display), matching what the panel displays.
    # ``rebin_reco_to_gen`` puts the detector level on the gen binning (the
    # ARC's alternative: square K, so chi2_smeared and chi2_unfold are
    # directly comparable). Both ratio curves then share the gen axis.
    hep.style.use("CMS")
    smeared_binning = "gen" if rebin_reco_to_gen else "reco"
    try:
        bl_rows, bl_global = u.bottom_line_test_by_pt(
            min_edge="shown", smeared_binning=smeared_binning)
    except RuntimeError:
        bl_rows, bl_global = [], None
    bl_by_i = {r["i"]: r for r in bl_rows}

    unfolded_pt_binned = unflatten_gen_by_pt(u.y_unf, u.gen_edges_by_pt)
    true_pt_binned = unflatten_gen_by_pt(u.y_true, u.gen_edges_by_pt)

    model_reco_shifts = u._model_reco_signed_shifts()

    measured_pt_binned = unflatten_gen_by_pt(u.y_meas, u.reco_edges_by_pt)
    reco_mc_pt_binned = unflatten_gen_by_pt(u.mosaic.sum(axis = 1), u.reco_edges_by_pt)

    # Optional: reco-level HERWIG spectrum per pT bin, projected exactly like
    # the input-distribution overlay (same reco binning as reco_mc). Used to
    # add the reco-level HERWIG-vs-PYTHIA difference to the model band.
    herwig_reco_pt_binned = None
    if include_herwig_band and getattr(u, "herwig_2d", None) is not None:
        _hh = u.herwig_2d
        if "systematic" in _hh.axes.name:
            _hh = _hh.project("ptreco", u.reco_axis, "systematic")[:, :, "nominal"]
        else:
            _hh = _hh.project("ptreco", u.reco_axis)
        _hpt = np.asarray(_hh.axes['ptreco'].edges, float)
        _hvals = np.asarray(_hh.values(), float)
        _hedges = np.asarray(_hh.axes[u.reco_axis].edges, float)
        _pt_edges = np.asarray(u.bins.pt_edges, float)
        herwig_reco_pt_binned = []
        for _k in range(len(_pt_edges) - 1):
            _a = int(np.searchsorted(_hpt, _pt_edges[_k] - 1e-9))
            _b = (len(_hpt) - 1 if not np.isfinite(_pt_edges[_k + 1])
                  else int(np.searchsorted(_hpt, _pt_edges[_k + 1] - 1e-9)))
            _global = _hvals[_a:_b, :].sum(axis=0)
            # Map the global fine reco axis onto this pT slice's reco binning
            # (identity for groomed where they coincide; a coarsening for the
            # nested ungroomed per-pT binning) so it matches reco_mc.
            herwig_reco_pt_binned.append(u._reco_to_gen_rebin_1d(
                _global, _hedges, u.reco_edges_by_pt[_k]))

    #now plot the ratio of unfolded to true and measured to reco mc in the same axis, just the ratio plot (no main panel)
    gen_offset = sum(
        len(edges) - 1
        for edges in u.gen_edges_by_pt[:getattr(u, "first_reported_pt_bin", 0)]
    )
    for i in u._reported_pt_indices():
        n_gen_bins = len(u.gen_edges_by_pt[i]) - 1
        gen_slice = slice(gen_offset, gen_offset + n_gen_bins)
        gen_offset += n_gen_bins
        fig, ax = plt.subplots(figsize=(12, 9))
        gen_display_slice, gen_edges = u._gen_display_slice(i)
        gen_display_indices = np.arange(n_gen_bins)[gen_display_slice]
        unfolded = u._normalized_slice(
            unfolded_pt_binned[i], u.gen_edges_by_pt[i], i
        )[gen_display_slice]
        true = u._normalized_slice(
            true_pt_binned[i], u.gen_edges_by_pt[i], i
        )[gen_display_slice]
        
        ratio_unf_true = np.divide(unfolded, true, out=np.full_like(unfolded, np.nan), where=true != 0)

        # Propagate the data-stat covariance through the same per-pT
        # normalisation used for the displayed ratio.  This is deliberately
        # not the total uncertainty band: the bottom-line test is a
        # data-stat-only diagnostic at this stage.
        ratio_unf_true_err = None
        covariance = getattr(u, "cov_data_np", None)
        if covariance is not None:
            raw_unfolded = np.asarray(u.y_unf[gen_slice], dtype=float)
            normalisation_jacobian = u._slice_normalization_jacobian(
                raw_unfolded, u.gen_edges_by_pt[i], i
            )
            if np.any(normalisation_jacobian):
                normalised_covariance = (
                    normalisation_jacobian
                    @ np.asarray(covariance)[gen_slice, gen_slice]
                    @ normalisation_jacobian.T
                )
                normalised_covariance = normalised_covariance[np.ix_(
                    gen_display_indices, gen_display_indices
                )]
                ratio_unf_true_err = np.divide(
                    np.sqrt(np.clip(np.diag(normalised_covariance), 0.0, None)),
                    np.abs(true),
                    out=np.zeros_like(true),
                    where=true != 0,
                )
        
        if rebin_reco_to_gen:
            # Detector level rebinned onto the gen binning: both curves share
            # the gen axis and the smeared side is a square-K comparison.
            meas_counts = u._reco_to_gen_rebin_1d(
                measured_pt_binned[i], u.reco_edges_by_pt[i], u.gen_edges_by_pt[i])
            recomc_counts = u._reco_to_gen_rebin_1d(
                reco_mc_pt_binned[i], u.reco_edges_by_pt[i], u.gen_edges_by_pt[i])
            measured = u._normalized_slice(
                meas_counts, u.gen_edges_by_pt[i], i
            )[gen_display_slice]
            reco_mc = u._normalized_slice(
                recomc_counts, u.gen_edges_by_pt[i], i
            )[gen_display_slice]
            meas_edges = gen_edges
            reco_display_slice = gen_display_slice
        else:
            reco_display_slice, meas_edges = u._reco_display_slice(i)
            measured = u._normalized_slice(
                measured_pt_binned[i], u.reco_edges_by_pt[i], i, reco=True
            )[reco_display_slice]
            reco_mc = u._normalized_slice(
                reco_mc_pt_binned[i], u.reco_edges_by_pt[i], i, reco=True
            )[reco_display_slice]
        ratio_meas_reco = np.divide(measured, reco_mc, out=np.full_like(measured, np.nan), where=reco_mc != 0)

        # Model uncertainty on the Reco_MC denominator. This is a
        # reconstruction-level band and is deliberately NOT the gen-level
        # PS/HAD envelope used for the unfolded result: it is the envelope
        # of the ISR and FSR shower-scale PSWeight variations, taken
        # directly at reco level.
        model_band = None
        if model_reco_shifts is not None:
            def _frac(source):
                frac = model_reco_shifts.get(source, {}).get(i)
                if frac is None:
                    n_bins = (
                        len(u.gen_edges_by_pt[i]) - 1
                        if rebin_reco_to_gen
                        else len(u.reco_edges_by_pt[i]) - 1
                    )
                    return np.zeros(n_bins)
                if rebin_reco_to_gen:
                    # Rebin the signed shift as a nominal-count-weighted
                    # mean: rebin nom*(1+frac) against rebinned nom.
                    nom_reb = u._reco_to_gen_rebin_1d(
                        reco_mc_pt_binned[i],
                        u.reco_edges_by_pt[i], u.gen_edges_by_pt[i])
                    var_reb = u._reco_to_gen_rebin_1d(
                        reco_mc_pt_binned[i] * (1.0 + frac),
                        u.reco_edges_by_pt[i], u.gen_edges_by_pt[i])
                    return np.divide(
                        var_reb, nom_reb,
                        out=np.ones_like(nom_reb), where=nom_reb != 0) - 1.0
                return frac
            isr_frac = np.maximum(
                np.abs(_frac("isrUp")), np.abs(_frac("isrDown")))
            fsr_frac = np.maximum(
                np.abs(_frac("fsrUp")), np.abs(_frac("fsrDown")))
            band_frac = np.maximum(isr_frac, fsr_frac)
            herwig_frac = None
            if include_herwig_band and herwig_reco_pt_binned is not None:
                her = herwig_reco_pt_binned[i]
                pyt = reco_mc_pt_binned[i]
                hn = her / max(u._shown_norm_total(her, i, reco=True), 1e-300)
                pn = pyt / max(u._shown_norm_total(pyt, i, reco=True), 1e-300)
                hfrac_reco = np.divide(hn, pn, out=np.ones_like(pn),
                                       where=pn != 0) - 1.0
                if rebin_reco_to_gen:
                    nom_reb = u._reco_to_gen_rebin_1d(
                        pyt, u.reco_edges_by_pt[i], u.gen_edges_by_pt[i])
                    var_reb = u._reco_to_gen_rebin_1d(
                        pyt * (1.0 + hfrac_reco),
                        u.reco_edges_by_pt[i], u.gen_edges_by_pt[i])
                    herwig_frac = np.divide(
                        var_reb, nom_reb, out=np.ones_like(nom_reb),
                        where=nom_reb != 0) - 1.0
                else:
                    herwig_frac = hfrac_reco
                band_frac = np.maximum(band_frac, np.abs(herwig_frac))
            band_frac = band_frac[reco_display_slice]
            model_band = band_frac * np.abs(ratio_meas_reco)
            print(f"  BLT band pt{i}: max|ISR|={isr_frac.max():.3f} "
                  f"max|FSR|={fsr_frac.max():.3f}"
                  + (f" max|HERWIG|={np.abs(herwig_frac).max():.3f}"
                     if herwig_frac is not None else ""))

        # Data statistical uncertainty on the normalized Measured/Reco_MC
        # ratio: Poisson on the reported data counts propagated through the
        # per-slice unit-area normalization, rel = sqrt((1 - p) / N).
        if rebin_reco_to_gen:
            raw_meas = u._reco_to_gen_rebin_1d(
                measured_pt_binned[i], u.reco_edges_by_pt[i],
                u.gen_edges_by_pt[i])
        else:
            raw_meas = np.asarray(measured_pt_binned[i], float)
        raw_meas = raw_meas[reco_display_slice]
        _Mtot = raw_meas.sum()
        _p = np.divide(raw_meas, _Mtot, out=np.zeros_like(raw_meas), where=_Mtot > 0)
        stat_frac = np.sqrt(np.clip(np.divide(
            1.0 - _p, raw_meas, out=np.zeros_like(raw_meas), where=raw_meas > 0),
            0.0, None))
        stat_band = stat_frac * np.abs(ratio_meas_reco)

        ax.axhline(1.0, color='gray', ls='--')
        x_steps = np.repeat(np.asarray(meas_edges, float), 2)[1:-1]
        if model_band is not None:
            total_band = np.sqrt(model_band ** 2 + stat_band ** 2)
            band_label = ('Model $\\oplus$ stat unc. (Reco MC, incl. HERWIG)'
                          if include_herwig_band and herwig_frac is not None
                          else 'Model $\\oplus$ stat unc. (Reco MC)')
        else:
            total_band = stat_band
            band_label = 'Stat unc. (Reco MC)'
        ax.fill_between(
            x_steps,
            np.repeat(ratio_meas_reco - total_band, 2),
            np.repeat(ratio_meas_reco + total_band, 2),
            color='#e42536', alpha=0.20, linewidth=0, label=band_label)
        # Shower-scale diagnostics are separate from the filled total
        # band.  The band takes their bin-wise maximum, while the two
        # outlines retain the individual ISR and FSR excursions.  This is
        # optional because the ISR weight is not result-ready.
        if model_band is not None and show_isr_fsr_envelope:
            for frac, color, label in (
                (isr_frac, '#ff7f0e', 'ISR envelope'),
                (fsr_frac, '#2ca02c', 'FSR envelope'),
            ):
                shower_band = frac * np.abs(ratio_meas_reco)
                for sign in (+1, -1):
                    ax.plot(
                        x_steps,
                        np.repeat(ratio_meas_reco + sign * shower_band, 2),
                        color=color, ls='-', lw=1.2, alpha=0.9,
                        label=(label if sign == +1 else None),
                    )
        if model_band is not None and include_herwig_band and herwig_frac is not None:
            herwig_band = np.abs(herwig_frac) * np.abs(ratio_meas_reco)
            for sign in (+1, -1):
                ax.plot(x_steps, np.repeat(ratio_meas_reco + sign * herwig_band, 2),
                        color='#3f90da', ls='--', lw=1.3,
                        label=('HERWIG reco diff' if sign == +1 else None))
        hep.histplot(ratio_unf_true, gen_edges, yerr=ratio_unf_true_err,
                     label='Unfolded / True', color='k', ls='--')
        reco_label = 'Measured / Reco_MC (rebinned to gen)' if rebin_reco_to_gen else 'Measured / Reco_MC'
        hep.histplot(ratio_meas_reco, meas_edges, yerr=stat_band,
                     label=reco_label, color='#e42536', ls=':')
        ax.set_ylabel('Ratio')
        ax.set_xlim(u.gen_edges_by_pt[i][0], u.gen_edges_by_pt[i][-1])
        # Keep the full ratio excursion visible by default.  Pass ``None``
        # for Matplotlib's automatic limits when inspecting outliers.
        if ratio_ylim is not None:
            ax.set_ylim(*ratio_ylim)
        plt.xlim(*u._display_xlim(i))
        plt.xlabel(u._observable_label())
        title = f"pT bin: {int(u.pt_edges[i])}-{int(u.pt_edges[i+1]) if i+1 < len(u.pt_edges)-1 else '∞'} GeV"
        # Pinned lower-left with an opaque frame (matching the chi2 text
        # box, which owns the top-left): 'best' placement can land on the
        # model band or the chi2 text in the busier panels.
        plt.legend(title=title, loc='lower left', frameon=True,
                   framealpha=0.85, facecolor='white', edgecolor='0.7')
        slice_row = bl_by_i.get(i)
        # The quantitative test lives in the chi2 summary bar charts; the
        # panels stamp it only when asked, so the ratio curves stay readable.
        if annotate_chi2 and (slice_row is not None or bl_global is not None):
            ax.text(
                0.03,
                0.97,
                u._bottom_line_panel_text(slice_row, bl_global),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PUB_ANNOTATION_FONTSIZE,
                bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
            )
        hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(), fontsize=20)
        mode = "groomed" if u.groomed else "ungroomed"
        tag = "_rebinned" if rebin_reco_to_gen else ""
        if include_herwig_band and herwig_reco_pt_binned is not None:
            tag += "_herwig"
        save_path = Path(u.spec.output_dir) / f"bottom_line{tag}_{mode}_{u._output_panel_index(i)}.pdf"
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_bottom_line_chi2_summary(u, show=True, normalized=False,
                                  min_edge=None):
    r"""Grouped bar chart of chi2_smeared vs chi2_unfold per pT slice.

    Direct visual of the bottom-line inequality: the (blue) unfolded bar
    must not exceed the (red) smeared bar. Raw chi2 -- not chi2/ndf -- is the
    quantity the test constrains; the ndf is annotated on each bar. The
    smeared ndf is the reco bin count; the unfolded ndf is the effective
    rank of K J J^T K^T (CMS Stat Committee / TUnfold TWiki), which under
    regularization is below the gen bin count. Data-stat-only, per
    bottom_line_test_by_pt.

    ``normalized=True`` plots chi2/ndf instead -- the quantity the ARC asked
    to compare (they no longer want the absolute chi2). The smeared ndf is
    the reco bin count and the unfolded ndf is the effective rank of
    K J J^T K^T (CMS Stat Committee / TUnfold TWiki); the single global
    chi2/ndf for each is annotated on the plot. The raw-chi2 inequality
    (the underlying bottom-line theorem) is still evaluated and printed to
    the log, but is no longer stamped on the figure.
    """
    rows, glob = u.bottom_line_test_by_pt(min_edge=min_edge)
    if not rows:
        return
    labels, groups = [], []
    for r in rows:
        labels.append(f"{r['pt_lo']}-{r['pt_hi']}" if r["pt_hi"] else f"{r['pt_lo']}-∞")
        groups.append(r)
    labels.append("Global")
    groups.append(glob)

    def val(g, key):
        m = g[key]
        return m["chi2"] / m["ndof"] if normalized and m["ndof"] > 0 else m["chi2"]

    x = np.arange(len(labels))
    w = 0.38
    c_sm = [val(g, "smeared") for g in groups]
    c_unf = [val(g, "unfolded") for g in groups]
    with_model = bool(glob.get("model_uncertainty_included", False))
    unf_label = (r"$\chi^2_\mathrm{unfold}$  (truth space, unfolded"
                 + (r"$\oplus$model" if with_model else "")
                 + " vs PYTHIA gen)")
    fig, ax = plt.subplots(figsize=(12, 9))
    b1 = ax.bar(x - w / 2, c_sm, w, color="#e42536", alpha=0.85,
                label=r"$\chi^2_\mathrm{smeared}$  (reco space, data vs folded PYTHIA)")
    b2 = ax.bar(x + w / 2, c_unf, w, color="#5790fc", alpha=0.9,
                label=unf_label)
    ax.set_yscale("log")
    # Push the log floor well below the smallest bar so the bottom-left
    # uncertainty-scope note never collides with a short bar's value
    # label (seen on the aligned-binning trijet summaries).
    positive_values = [v for v in (*c_sm, *c_unf) if v > 0]
    if positive_values:
        ax.set_ylim(bottom=min(positive_values) / 30.0)
    ax.set_ylabel(r"$\chi^2/n_\mathrm{dof}$ vs PYTHIA8" if normalized
                  else r"$\chi^2$ vs PYTHIA8")
    if with_model:
        ax.text(0.03, 0.03,
                r"smeared: data stat  ·  unfolded: stat$\oplus$model",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, color="#555555",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2), zorder=7)
    ax.set_xlabel(r"$p_T$ slice (GeV)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for bars, groups_ in ((b1, groups), (b2, groups)):
        key = "smeared" if bars is b1 else "unfolded"
        for rect, g in zip(bars, groups_):
            note = (fr"{g[key]['chi2']:.0f}/{u._fmt_ndof(g[key]['ndof'])}"
                    if normalized else f"ndf={u._fmt_ndof(g[key]['ndof'])}")
            ax.annotate(note,
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=9)
    # Resolve a human/file label for the cut region.
    is_shown = (min_edge == "shown")
    is_seq = isinstance(min_edge, (list, tuple, np.ndarray))
    if min_edge is None:
        cut_word, cut_tag = "full range", ""
    elif is_shown or is_seq:
        cut_word, cut_tag = "shown range", "_shown"
    else:
        cut_word = rf"$\log_{{10}}(\rho^2)\geq{min_edge:g}$"
        cut_tag = f"_above{abs(min_edge):g}".replace(".", "")

    passed = all(g["unfolded"]["chi2"] <= g["smeared"]["chi2"] + 1e-9 for g in groups)
    mode = "Groomed" if u.groomed else "Ungroomed"
    gsm_r = glob["smeared"]["chi2"] / max(glob["smeared"]["ndof"], 1e-9)
    gun_r = glob["unfolded"]["chi2"] / max(glob["unfolded"]["ndof"], 1e-9)
    print(f"[bottom-line {mode.lower()} ({cut_word})] "
          + ("PASS" if passed else "FAIL")
          + f" | GLOBAL chi2/ndf: smeared={gsm_r:.2f} unfolded={gun_r:.2f}")
    _box = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2)
    ax.text(0.03, 0.97, f"{mode}, {cut_word}", transform=ax.transAxes,
            ha="left", va="top", fontsize=18, bbox=_box, zorder=7)
    # The single global chi2/ndof number the ARC asked to compare.
    ax.text(0.03, 0.88,
            fr"Global $\chi^2/n_\mathrm{{dof}}$:  smeared $={gsm_r:.2f}$,  "
            fr"unfolded $={gun_r:.2f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, color="#333333", bbox=_box, zorder=7)
    ax.legend(loc="upper right", fontsize=16, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    # headroom so the bar annotations stay clear of the texts and the legend
    ax.set_ylim(top=max(max(c_sm), max(c_unf)) * (60 if ax.get_yscale() == "log" else 2.2))
    hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(), fontsize=20)
    suffix = "groomed" if u.groomed else "ungroomed"
    stem = f"bottom_line_chi2{'_perndf' if normalized else ''}_summary{cut_tag}_{suffix}.pdf"
    u._finalize_plot(
        save_path=Path(u.spec.output_dir) / stem,
        show=show, fig=fig,
    )


def _mark_offscale_ratio_band(ax, centers, frac_up, frac_down, *, top=2.0, bottom=0.0, xlim=None):
    """Flag ratio bins whose uncertainty band exceeds the fixed panel range.

    The ratio panel range is deliberately fixed so ordinary bins keep a
    readable band scale; an off-scale band (e.g. the trijet ungroomed
    first bin at ~180%) is truncated by the frame.  Mark such bins with
    edge arrows and print the true band extent inside the bin so the
    truncation is visible rather than silent.

    Only bins inside ``xlim`` (the displayed window) are marked: hidden
    buffer bins sit outside the frame, and a label placed there would be
    swept into the saved figure by ``bbox_inches="tight"``.
    """
    centers = np.asarray(centers, dtype=float)
    frac_up = np.asarray(frac_up, dtype=float)
    frac_down = np.asarray(frac_down, dtype=float)
    span = top - bottom
    for center, up, down in zip(centers, frac_up, frac_down):
        if xlim is not None and not (xlim[0] <= center <= xlim[1]):
            continue
        # Compare the DRAWN band edges, not the signed fractions: a
        # negative unfolded central value flips the fractions' signs and
        # inverts the drawn band (its "top" curve 1+up sits below the
        # baseline 1-down), which is exactly the consistent-with-zero
        # case this cue exists for.
        band_high = max(1.0 + up, 1.0 - down)
        band_low = min(1.0 + up, 1.0 - down)
        clipped_up = band_high > top + 1e-9
        clipped_down = band_low < bottom - 1e-9
        if not (clipped_up or clipped_down):
            continue
        up_extent = band_high - 1.0
        down_extent = 1.0 - band_low
        if clipped_up:
            ax.plot(center, top - 0.05 * span, marker="^", color="black",
                    markersize=6, linestyle="none", zorder=6, clip_on=True)
        if clipped_down:
            ax.plot(center, bottom + 0.05 * span, marker="v", color="black",
                    markersize=6, linestyle="none", zorder=6, clip_on=True)
        if clipped_up and clipped_down and np.isclose(
            up_extent, down_extent, rtol=0.05, atol=0.005
        ):
            label = f"band ±{100.0 * up_extent:.0f}%"
        else:
            parts = []
            if clipped_up:
                parts.append(f"+{100.0 * up_extent:.0f}%")
            if clipped_down:
                parts.append(f"\N{MINUS SIGN}{100.0 * down_extent:.0f}%")
            label = "band " + "/".join(parts)
        ax.text(center, bottom + 0.5 * span, label, rotation=90,
                ha="center", va="center", fontsize=11, color="black", zorder=6, clip_on=True)


def plot_unfolded_fancy(u, log=False, show=True):
    # The focused pair-split runner calls this before any other core plot.
    # Establish the CMS 10x10 canvas here rather than inheriting Matplotlib's
    # 6.4x4.8 default: the latter cannot accommodate the public-size labels,
    # CMS/pT annotation, and uncertainty legend without clipping.  This is
    # idempotent for the legacy ``run_all_plots`` path, which already sets
    # the same style before calling us.
    hep.style.use("CMS")
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
    npt = len(u.pt_edges)-1
    has_herwig = getattr(u, "has_herwig", True)
    # Self-closure figures compare unfolded MC to its own truth: the
    # alternate-generator overlays are noise there, and the points are not
    # data.  Keep the PYTHIA truth curve (its chi2 is the closure metric).
    is_closure = bool(u.closure or u.herwig_closure)
    if is_closure:
        has_herwig = False
    points_label = 'Unfolded' if is_closure else 'Data'
    # Stats-only run (self-closure): the total band would just repeat the
    # stat band, so only the stat band is drawn.
    stat_only = u._band_is_stat_only
    stat_label = "Stat. Unc." if u.response_matrix_stat_available else "Input Stat. Unc."
    total_label = (
        r"Syst. $\oplus$ Stat. Unc."
        if u.response_matrix_stat_available and has_herwig
        else r"Partial Syst. $\oplus$ Stat. Unc."
    )
    title_list = []
    normalization_window = u._normalization_window()
    display_window = u._display_window()
    normalization_note = None
    if (
        normalization_window is not None
        and normalization_window != display_window
    ):
        normalization_note = (
            f"Norm. [{normalization_window[0]:g}, "
            f"{normalization_window[1]:g}]"
        )
    for i in range(npt):
        lo = int(u.pt_edges[i])
        if i + 1 < npt:
            hi = int(u.pt_edges[i + 1])
            pt_title = rf"${lo} < p_{{\mathrm{{T}}}} < {hi}$ GeV"
        else:
            pt_title = rf"${lo} < p_{{\mathrm{{T}}}} < \infty$ GeV"
        title_list.append(pt_title)
    true_herwig_pt_binned = (
        unflatten_gen_by_pt(u.y_true_herwig, u.gen_edges_by_pt)
        if has_herwig
        else None
    )
    # True standalone-Vincia gen prediction (rho only).  Pair-split runs
    # attach an audited MESS+Vincia prediction explicitly; their required
    # flag prevents any accidental fallback to this core's Z+jet cache.
    vincia_truth = None
    vincia_label = "Vincia"
    if u.spec.name == "rho" and not is_closure:
        if getattr(u, "pairsplit_vincia_required", False):
            prediction = getattr(u, "pairsplit_vincia_prediction", None)
            if prediction is None:
                raise RuntimeError(
                    "pair-split MESS+Vincia prediction was required but not attached"
                )
            vincia_truth = prediction.truth_by_pt()
            vincia_label = prediction.label
        else:
            try:
                from unfold.model import vincia_truth_by_pt
                vincia_truth = vincia_truth_by_pt(u)
            except Exception:
                vincia_truth = None
    for i in u._reported_pt_indices():
        # CMS-default canvas (forced figsize shrinks fonts in the scaled
        # PDF); ratio panel close to the main panel (ARC round-2).
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07},
            layout="constrained",
        )
        display_slice, rho_edges = u._gen_display_slice(i)
        display_indices = np.arange(len(u.gen_edges_by_pt[i]) - 1)[display_slice]
        bin_widths = np.diff(u.gen_edges_by_pt[i])
        if has_herwig:
            herwig_norm = (
                true_herwig_pt_binned[i] / bin_widths
                / u._shown_norm_total(true_herwig_pt_binned[i], i)
            )[display_slice]
        unfolded = np.array(
            u.normalized_results[i]['unfolded'], dtype=float
        )[display_slice]
        stat_unc = np.array(
            u.normalized_results[i]['stat_unc'], dtype=float
        )[display_slice]
        syst_up = np.array(
            u.normalized_results[i]['syst_unc']['up'], dtype=float
        )[display_slice]
        syst_down = np.array(
            u.normalized_results[i]['syst_unc']['down'], dtype=float
        )[display_slice]
        pythia = np.array(
            u.normalized_results[i]['true'], dtype=float
        )[display_slice]
        if vincia_truth is not None:
            vincia_norm, vincia_err = vincia_truth[i]
            vincia_norm = np.asarray(vincia_norm, dtype=float)[display_slice]
            vincia_err = np.asarray(vincia_err, dtype=float)[display_slice]
        centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
        plt.sca(ax_main)
        if not stat_only:
            plt.stairs( unfolded + syst_up,
                rho_edges,
                baseline = unfolded - syst_down,
                fill = True, color = u.spec.band_color_total , label = total_label)
            plt.stairs( unfolded + stat_unc,
                rho_edges,
                baseline = unfolded - stat_unc,
                fill = True, color = u.spec.band_color_stat , label = stat_label)
        # Track the tallest drawn curve so the legend headroom clears the
        # predictions too, not just the data band.
        curve_max = float(np.max(unfolded + syst_up))
        curve_max = max(curve_max, float(np.max(pythia)))
        if has_herwig:
            curve_max = max(curve_max, float(np.max(herwig_norm)))
        if vincia_truth is not None:
            curve_max = max(curve_max, float(np.max(vincia_norm)))
        # Data-vs-MC chi2/ndf per panel using the full covariance of the
        # normalized result (ARC round-3: stat through the normalization
        # Jacobian + rank-1 systematics + projected PS/HAD model term)
        # PLUS each prediction's own uncertainty (Pythia: theory + MC
        # stat; Herwig: MC stat; Vincia: sample stat). The per-pT sum
        # constraint makes the block singular by one, so chi2 uses the
        # pseudo-inverse and ndf = n_shown - 1. Restricted to the shown
        # bins so the quoted agreement covers exactly what the panel
        # displays.
        shown = u._shown_gen_mask(i)[display_slice]
        chi2_offset = sum(
            len(edges) - 1 for edges in u.gen_edges_by_pt[:i]
        )
        chi2_block = chi2_offset + display_indices
        results_cov = u._results_chi2_covariance()
        def _mc_chi2_label(name, prediction, pred_cov=None):
            sigma = np.maximum(syst_up, syst_down)
            good = (sigma > 0) & np.isfinite(prediction) & shown
            ndf = max(int(good.sum()) - 1, 1)
            idx = chi2_block[good]
            residual = unfolded[good] - prediction[good]
            cov_sub = results_cov[np.ix_(idx, idx)]
            if pred_cov is not None:
                pred_idx = display_indices[good]
                cov_sub = cov_sub + pred_cov[np.ix_(pred_idx, pred_idx)]
            chi2 = float(
                residual @ np.linalg.pinv(cov_sub, rcond=1e-10) @ residual
            )
            return rf"{name} ($\chi^2$/ndf = {chi2:.1f}/{ndf})"
        plt.stairs(pythia, rho_edges,
                   label=_mc_chi2_label(
                       'Pythia8', pythia,
                       u._prediction_chi2_covariance(i, "pythia")),
                   color='#5790fc', ls='dotted', lw=3, baseline=None)
        py_unc_up, py_unc_down = u._prediction_uncertainty(i, "pythia")
        if py_unc_up is not None:
            py_unc_up = py_unc_up[display_slice]
            py_unc_down = py_unc_down[display_slice]
            plt.errorbar(centers, pythia, yerr=[py_unc_down, py_unc_up], fmt='none',
                         ecolor='#5790fc', elinewidth=1.5, capsize=3)
        if has_herwig:
            plt.stairs(herwig_norm, rho_edges,
                       label=_mc_chi2_label(
                           'Herwig7', herwig_norm,
                           u._prediction_chi2_covariance(i, "herwig")),
                       color='#e42536', ls='dashdot', lw=2, baseline=None)
            hw_unc_up, hw_unc_down = u._prediction_uncertainty(i, "herwig")
            if hw_unc_up is not None:
                hw_unc_up = hw_unc_up[display_slice]
                hw_unc_down = hw_unc_down[display_slice]
                plt.errorbar(centers, herwig_norm, yerr=[hw_unc_down, hw_unc_up], fmt='none',
                             ecolor='#e42536', elinewidth=1.5, capsize=3)
        if vincia_truth is not None:
            plt.stairs(vincia_norm, rho_edges,
                       label=_mc_chi2_label(
                           vincia_label, vincia_norm,
                           (u.vincia_stat_covariance_by_pt[i]
                            if hasattr(u, "vincia_stat_covariance_by_pt") else
                            np.diag(np.asarray(vincia_truth[i][1], float) ** 2))),
                       color='#964a8b', ls='dashed', lw=2, baseline=None)
            plt.errorbar(centers, vincia_norm, yerr=vincia_err, fmt='none',
                         ecolor='#964a8b', elinewidth=1.5, capsize=3)
        # ARC round-2: data as black points (not per-panel markers), with
        # vertical (stat) AND horizontal (bin-width) error bars; the
        # errorbar handle carries the bars into the legend.
        plt.errorbar(centers, unfolded, yerr=stat_unc, xerr=np.diff(rho_edges) / 2,
                     fmt='o', color='k', markersize=7, label=points_label)

        # Legend headroom, then the pT-bin info below the legend and
        # left-aligned to it (ARC round-2).
        ax_main.set_ylim(0, 1.9 * curve_max)
        # Approval comments (A. Meyer): Data at the top of the legend.
        handles, labels = ax_main.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda k: labels[k] != points_label)
        legend = ax_main.legend([handles[k] for k in order],
                                [labels[k] for k in order],
                                loc="upper right", fontsize=PUB_LEGEND_FONTSIZE)
        # In-frame CMS tag (loc=2), with the pT range right under it
        # (Aritra's preference).
        fig.canvas.draw()
        label_art = hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(),
                                  com=u._com_label(), loc=2, ax=ax_main)
        fig.canvas.draw()
        inv = ax_main.transAxes.inverted()
        boxes = [t.get_window_extent().transformed(inv) for t in label_art
                 if getattr(t, "get_text", lambda: "")()]
        boxes = [b for b in boxes if b.y0 < 1.0]  # skip the lumi text above the frame
        cms_x0 = min(b.x0 for b in boxes)
        cms_y0 = min(b.y0 for b in boxes)
        panel_title = title_list[i]
        if normalization_note is not None:
            panel_title += "\n" + normalization_note
        pt_text = ax_main.text(cms_x0, cms_y0 - 0.04, panel_title,
                               transform=ax_main.transAxes, ha="left", va="top",
                               fontsize=PUB_ANNOTATION_FONTSIZE)
        # Final headroom: every curve stays below the CMS+pT block and the
        # legend (all live in axes fractions, unaffected by ylim).
        fig.canvas.draw()
        tbox = pt_text.get_window_extent().transformed(inv)
        lbox = legend.get_window_extent().transformed(inv)
        clearance = min(tbox.y0, lbox.y0)
        ax_main.set_ylim(0, curve_max / max(0.35, clearance - 0.04))
        plt.ylabel(u._normalized_ylabel(), fontsize=PUB_LABEL_FONTSIZE)
        ax_main.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)

        # Ratio Plot
        plt.sca(ax_ratio)
        plt.axhline(1.0, color='gray', ls='--')
        ratio_pythia = np.divide(unfolded, pythia)
        stat_frac = np.divide(stat_unc, unfolded, out=np.zeros_like(stat_unc), where=unfolded != 0)
        total_frac_up = np.divide(syst_up, unfolded, out=np.zeros_like(syst_up), where=unfolded != 0)
        total_frac_down = np.divide(syst_down, unfolded, out=np.zeros_like(syst_down), where=unfolded != 0)

        if stat_only:
            # No band: the ratio carries the same stat uncertainty as the
            # points above, drawn once as error bars on the ratio markers.
            # No x error bars here: at bin width they merge into a solid
            # line that buries the unity reference.
            # abs: a negative unfolded bin flips the sign of the fraction
            plt.errorbar(centers, ratio_pythia, yerr=np.abs(stat_frac), fmt='o',
                         color='k', markersize=5)
        else:
            plt.stairs(1.0 + total_frac_up, rho_edges, baseline=1.0 - total_frac_down, fill=True, color=u.spec.band_color_total, label=total_label)
            plt.stairs(1.0 + stat_frac, rho_edges, baseline=1.0 - stat_frac, fill=True, color=u.spec.band_color_stat, label=stat_label)
            plt.stairs(ratio_pythia, rho_edges, color='#5790fc', ls='dotted', lw=2,
                       label='Unfolded / Pythia8' if is_closure else 'Data / Pythia8', baseline=None)
        # PYTHIA8 uncertainty propagated onto Data/PYTHIA8 (ratio ~ 1/PYTHIA,
        # so a +sigma on PYTHIA pulls the ratio down by ratio*sigma/PYTHIA).
        if py_unc_up is not None:
            rel_py = np.divide(np.abs(ratio_pythia), pythia, out=np.zeros_like(pythia), where=pythia != 0)
            plt.errorbar(
                centers, ratio_pythia, yerr=[rel_py * py_unc_down, rel_py * py_unc_up],
                fmt='none', ecolor='#5790fc', elinewidth=1.2, capsize=2,
            )
        if has_herwig:
            ratio_herwig = np.divide(unfolded, herwig_norm)
            plt.stairs(ratio_herwig, rho_edges, color='#e42536', ls='dashdot', lw=2, label='Data / Herwig7', baseline=None)
            if hw_unc_up is not None:
                rel_hw = np.divide(np.abs(ratio_herwig), herwig_norm, out=np.zeros_like(herwig_norm), where=herwig_norm != 0)
                plt.errorbar(
                    centers, ratio_herwig, yerr=[rel_hw * hw_unc_down, rel_hw * hw_unc_up],
                    fmt='none', ecolor='#e42536', elinewidth=1.2, capsize=2,
                )
        if vincia_truth is not None:
            ratio_vincia = np.divide(
                unfolded, vincia_norm,
                out=np.zeros_like(unfolded), where=vincia_norm != 0)
            plt.stairs(ratio_vincia, rho_edges, color='#964a8b', ls='dashed', lw=2,
                       label=f'Data / {vincia_label}', baseline=None)
        plt.ylim(0, 2)
        if not stat_only:
            _mark_offscale_ratio_band(
                ax_ratio, centers, total_frac_up, total_frac_down,
                top=2.0, bottom=0.0, xlim=u._display_xlim(i),
            )
        # No tick label at 0 or 2 (as in the data/MC figure): the corner "0"
        # collides with the first x tick label and the top "2" crowds the
        # main panel's "0.0" above.
        ax_ratio.set_yticks([0.5, 1.0, 1.5])
        plt.xlabel(u._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
        plt.ylabel(r"$\frac{Unfolded}{Truth}$" if is_closure else r"$\frac{Data}{Simulation}$",
                   fontsize=PUB_LABEL_FONTSIZE)
        ax_ratio.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        plt.xlim(*u._display_xlim(i))
        panel_index = u._output_panel_index(i)
        if u.closure:
            save_path = (
                f"./{u.spec.output_dir}closure_groomed_{panel_index}.pdf"
                if u.groomed
                else f"./{u.spec.output_dir}closure_ungroomed_{panel_index}.pdf"
            )
        else:
            save_path = (
                f"./{u.spec.output_dir}unfold/groomed_{panel_index}.pdf"
                if u.groomed
                else f"./{u.spec.output_dir}unfold/ungroomed_{panel_index}.pdf"
            )
        u._finalize_plot(save_path=save_path, show=show, fig=fig)
    
    # Summary plot: all pT slices together, offset by 10^n. ARC round-2:
    # ratio panel added (Data / Pythia8 per slice) "to better understand
    # the level of agreement"; CMS-default canvas; Petroff colors;
    # Pythia8/Herwig7/Vincia casing; data as black markers with vertical
    # + horizontal error bars carried into the legend.
    fig, ax_main = plt.subplots()
    ratio_inputs = []  # per-slice payload for the separate ratio figure
    pt_key_entries = []  # (handle, label) rows for the SMP-24-010-style pT key
    for i in u._summary_pt_indices():
        exponent = 2 * i - 1
        scale = 10 ** exponent
        display_slice, rho_edges = u._gen_display_slice(
            i, legacy_low_prefix=True
        )
        unfolded = np.array(
            u.normalized_results[i]['unfolded'], dtype=float
        )[display_slice]
        syst_up = np.array(
            u.normalized_results[i]['syst_unc']['up'], dtype=float
        )[display_slice]
        syst_down = np.array(
            u.normalized_results[i]['syst_unc']['down'], dtype=float
        )[display_slice]
        stat_unc = np.array(
            u.normalized_results[i]['stat_unc'], dtype=float
        )[display_slice]
        bin_widths = np.diff(u.gen_edges_by_pt[i])[display_slice]
        if has_herwig:
            herwig_norm = (
                true_herwig_pt_binned[i] / np.diff(u.gen_edges_by_pt[i])
                / u._shown_norm_total(true_herwig_pt_binned[i], i)
            )[display_slice]

        y_syst_up = scale * (unfolded + syst_up)
        y_syst_down = scale * (unfolded - syst_down)
        y_syst_down = np.maximum(y_syst_down, scale * unfolded * 1e-1)
        y_stat_up = scale * (unfolded + stat_unc)
        y_stat_down = scale * (unfolded - stat_unc)
        pythia = np.array(
            u.normalized_results[i]['true'], dtype=float
        )[display_slice]
        centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
        summary_max = max(getattr(u, "_summary_max", 0.0), float(np.max(y_syst_up)), float(np.max(scale * pythia)))
        u._summary_max = summary_max
        positive = y_syst_down[y_syst_down > 0]
        if positive.size:
            u._summary_min = min(getattr(u, "_summary_min", np.inf), float(np.min(positive)))
        ax_main.stairs(scale * pythia, rho_edges, label='Pythia8', color='#5790fc', ls='dotted', lw=3, baseline=None)
        py_unc_up, py_unc_down = u._prediction_uncertainty(i, "pythia")
        if py_unc_up is not None:
            py_unc_up, py_unc_down = (
                py_unc_up[display_slice], py_unc_down[display_slice]
            )
            ax_main.errorbar(centers, scale * pythia, yerr=[scale * py_unc_down, scale * py_unc_up],
                             fmt='none', ecolor='#5790fc', elinewidth=1.2, capsize=2)
        if has_herwig:
            ax_main.stairs(scale * herwig_norm, rho_edges, label='Herwig7', color='#e42536', ls='dashdot', lw=2, baseline=None)
            hw_unc_up, hw_unc_down = u._prediction_uncertainty(i, "herwig")
            if hw_unc_up is not None:
                hw_unc_up, hw_unc_down = (
                    hw_unc_up[display_slice], hw_unc_down[display_slice]
                )
                ax_main.errorbar(centers, scale * herwig_norm, yerr=[scale * hw_unc_down, scale * hw_unc_up],
                                 fmt='none', ecolor='#e42536', elinewidth=1.2, capsize=2)
        if vincia_truth is not None:
            vincia_norm, vincia_err = vincia_truth[i]
            vincia_norm, vincia_err = (
                vincia_norm[display_slice], vincia_err[display_slice]
            )
            ax_main.stairs(scale * vincia_norm, rho_edges, label=vincia_label,
                           color='#964a8b', ls='dashed', lw=2, baseline=None)
            ax_main.errorbar(centers, scale * vincia_norm, yerr=scale * vincia_err,
                             fmt='none', ecolor='#964a8b', elinewidth=1.2, capsize=2)
        if not stat_only:
            ax_main.stairs(y_syst_up, rho_edges, baseline=y_syst_down, fill=True, color=u.spec.band_color_total, label=total_label, alpha = 0.8)
            ax_main.stairs(y_stat_up, rho_edges, baseline=y_stat_down, fill=True, color=u.spec.band_color_stat, label=stat_label)
        # Approval comments (A. Meyer, SMP-24-010 style): the main legend
        # keeps only the series identity -- one "Data" row with a
        # representative marker -- while a separate compact key maps the
        # per-slice marker to its pT interval and 10^n offset factor,
        # mirroring SMP-24-010 Fig. 10 (marker key with the offsets
        # printed in the key, nothing written next to the curves).
        h_data = ax_main.errorbar(centers, scale * unfolded, yerr=scale * stat_unc,
                                  xerr=np.diff(rho_edges) / 2, fmt=markers[i],
                                  color='k', markersize=7, label='Data')
        pt_key_entries.append(
            (h_data, rf'{title_list[i]} ($\times 10^{{{exponent}}}$)'))

        ratio_inputs.append({
            "i": i, "edges": rho_edges, "centers": centers,
            "widths": np.diff(rho_edges), "unfolded": unfolded,
            "pythia": pythia, "syst_up": syst_up, "syst_down": syst_down,
            "stat_unc": stat_unc,
            "herwig": herwig_norm if has_herwig else None,
            "vincia": vincia_norm if vincia_truth is not None else None,
            "pythia_unc": (py_unc_up, py_unc_down),
            "herwig_unc": (hw_unc_up, hw_unc_down) if has_herwig else (None, None),
            "vincia_err": vincia_err if vincia_truth is not None else None,
        })

    ax_main.set_yscale('log')
    # Group duplicate legend entries: keep first occurrence, hide subsequent ones
    handles, labels = ax_main.get_legend_handles_labels()
    seen = set()
    for h, l in zip(handles, labels):
        if l == "" or l in seen:
            try:
                h.set_label('_nolegend_')  # matplotlib ignores labels starting with '_'
            except Exception:
                pass
        else:
            seen.add(l)
    # Legend headroom on the log scale (ARC round-2 "legend must not
    # touch the plots"): measure the legend box, then solve the decade
    # span so the tallest histogram sits just below it.
    # Approval comments (A. Meyer): Data at the top of the legend.
    handles = [h for h in handles if h.get_label() and not h.get_label().startswith('_')]
    handles.sort(key=lambda h: h.get_label() != 'Data')
    # Two columns for both legend blocks: four pT slices two decades apart
    # cannot clear a one-column, ten-row legend stack without a 1e20 axis.
    legend = ax_main.legend(handles=handles, loc="upper right", ncol=2,
                            fontsize=PUB_LEGEND_FONTSIZE, frameon=True,
                            facecolor="white", edgecolor="none", framealpha=0.95,
                            columnspacing=1.0)
    fig.canvas.draw()
    lbox = legend.get_window_extent().transformed(ax_main.transAxes.inverted())
    # SMP-24-010-style pT key: a second, frameless legend right below the
    # series legend, one row per pT slice with its marker and 10^n factor.
    ax_main.add_artist(legend)
    key_legend = ax_main.legend(
        [h for h, _ in pt_key_entries], [l for _, l in pt_key_entries],
        loc="upper right", bbox_to_anchor=(lbox.x1, lbox.y0 - 0.005),
        bbox_transform=ax_main.transAxes, fontsize=PUB_LEGEND_FONTSIZE - 2,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.95,
        handletextpad=0.4, labelspacing=0.35, ncol=2, columnspacing=1.0)
    fig.canvas.draw()
    kbox = key_legend.get_window_extent().transformed(ax_main.transAxes.inverted())
    lbox = kbox  # headroom solve clears the lower of the two blocks
    # Floor just below the lowest drawn band (the autoscale bottom can be
    # decades lower from a near-empty bin, bloating the axis range).
    ybot = getattr(u, "_summary_min", ax_main.get_ylim()[0]) / 3.0
    span = np.log10(u._summary_max / ybot)
    need = span / max(0.2, lbox.y0 - 0.05)
    # The two legend blocks can claim two thirds of the canvas; cap the
    # extra decades so the axis does not run to 1e20 (legends are opaque).
    need = min(need, span + 4.0)
    ax_main.set_ylim(ybot, ybot * 10 ** need)
    u._summary_max = u._summary_min = None
    del u._summary_max, u._summary_min
    ax_main.set_ylabel(u._normalized_ylabel(), fontsize=PUB_LABEL_FONTSIZE)

    ax_main.set_xlabel(u._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
    ax_main.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
    ax_main.set_xlim(*u._display_xlim())

    # In-frame CMS tag: the clearance solve keeps the top-left free.
    fig.canvas.draw()
    hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(), loc=2, ax=ax_main)
    if normalization_note is not None:
        ax_main.text(
            0.04, 0.78, normalization_note,
            transform=ax_main.transAxes, ha="left", va="top",
            fontsize=PUB_ANNOTATION_FONTSIZE,
        )

    save_path = f"./{u.spec.output_dir}groomed_summary.pdf" if u.groomed else f"./{u.spec.output_dir}ungroomed_summary.pdf"
    u._finalize_plot(save_path=save_path, show=show, fig=fig)

    # ARC round-2: separate RATIO figure -- a grid of panels, one per pT
    # slice (highest pT on top, like the summary). Ratio TO DATA: the
    # data uncertainty bands sit at 1, generator curves float around it.
    n = len(ratio_inputs)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 4 * n),
                             gridspec_kw={"hspace": 0.1})
    axes = np.atleast_1d(axes)
    for k, d in enumerate(ratio_inputs):
        ax = axes[n - 1 - k]
        edges, centers, widths = d["edges"], d["centers"], d["widths"]
        unfolded = d["unfolded"]

        def _r(arr):
            return np.divide(arr, unfolded, out=np.zeros_like(unfolded), where=unfolded != 0)

        # Data uncertainty bands hugging unity; Data/Theory curves on top.
        if not stat_only:
            ax.stairs(1.0 + _r(d["syst_up"]), edges,
                      baseline=1.0 - _r(d["syst_down"]), fill=True,
                      color=u.spec.band_color_total, alpha=0.8, label=total_label)
            ax.stairs(1.0 + _r(d["stat_unc"]), edges,
                      baseline=1.0 - _r(d["stat_unc"]), fill=True,
                      color=u.spec.band_color_stat, label=stat_label)
        ax.axhline(1.0, color='gray', ls='--', lw=1)

        def _data_over(arr):
            return np.divide(unfolded, arr, out=np.zeros_like(unfolded), where=arr != 0)

        def _theory_band(theory, unc_up, unc_down, color):
            # Theory uncertainty propagated onto Data/Theory, drawn as
            # error-bar "sticks" at the bin centers (band was too faint).
            if unc_up is None:
                return
            ratio = _data_over(theory)
            lo = np.divide(unfolded, theory + unc_up,
                           out=np.zeros_like(unfolded), where=(theory + unc_up) != 0)
            # a down shift that drives the theory to <= 0 has no finite
            # upper ratio: draw no upper stick there instead of a huge one
            hi_den = theory - unc_down
            hi = np.divide(unfolded, hi_den, out=np.array(ratio, dtype=float), where=hi_den > 0)
            ax.errorbar(centers, ratio,
                        yerr=[np.clip(ratio - lo, 0, None),
                              np.clip(hi - ratio, 0, None)],
                        fmt='none', ecolor=color, elinewidth=1.5, capsize=3)

        _theory_band(d["pythia"], *d["pythia_unc"], '#5790fc')
        ax.stairs(_data_over(d["pythia"]), edges, color='#5790fc',
                  ls='dotted', lw=3, label='Data / Pythia8', baseline=None)
        if d["herwig"] is not None:
            _theory_band(d["herwig"], *d["herwig_unc"], '#e42536')
            ax.stairs(_data_over(d["herwig"]), edges, color='#e42536',
                      ls='dashdot', lw=2, label='Data / Herwig7', baseline=None)
        if d["vincia"] is not None:
            if d["vincia_err"] is not None:
                _theory_band(d["vincia"], d["vincia_err"], d["vincia_err"], '#964a8b')
            ax.stairs(_data_over(d["vincia"]), edges, color='#964a8b',
                      ls='dashed', lw=2, label=f'Data / {vincia_label}', baseline=None)
        ax.set_ylim(0, 2)
        if not stat_only:
            _mark_offscale_ratio_band(
                ax, centers, _r(d["syst_up"]), _r(d["syst_down"]),
                top=2.0, bottom=0.0, xlim=(float(edges[0]), float(edges[-1])),
            )
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE)
        ax.text(0.97, 0.95, title_list[d["i"]], transform=ax.transAxes,
                ha="right", va="top", fontsize=PUB_ANNOTATION_FONTSIZE)
    # Legend outside the frame: one row spanning the top panel's width,
    # with the CMS label stamped above it (pad from the measured box).
    handles, leg_labels = axes[0].get_legend_handles_labels()
    # Two columns, sized to their content rather than stretched to the axes
    # width (mode="expand"): at this type size three columns pushed the last
    # entry past the right frame edge, and equal thirds were narrower than
    # "Syst. + Stat. Unc." so its text ran under the next column's handle.
    # The uncertainty bands go in the left column and the Data/Theory ratios
    # in the right one. matplotlib fills columns top-to-bottom, so padding
    # both groups to the same length puts each in a column of its own.
    bands = [(h, l) for h, l in zip(handles, leg_labels) if not l.startswith("Data /")]
    ratios = [(h, l) for h, l in zip(handles, leg_labels) if l.startswith("Data /")]
    if bands and ratios:
        rows = max(len(bands), len(ratios))
        spacer = plt.Line2D([], [], linestyle="none")
        bands += [(spacer, "")] * (rows - len(bands))
        ratios += [(spacer, "")] * (rows - len(ratios))
        handles = [h for h, _ in bands + ratios]
        leg_labels = [l for _, l in bands + ratios]
    leg = axes[0].legend(handles, leg_labels, loc="lower left",
                         bbox_to_anchor=(0, 1.005),
                         ncol=2, fontsize=PUB_LEGEND_FONTSIZE, frameon=False,
                         borderaxespad=0, columnspacing=1.6,
                         handletextpad=0.5, handlelength=1.8)
    axes[len(axes) // 2].set_ylabel("Data/Theory", fontsize=PUB_LABEL_FONTSIZE)
    axes[-1].set_xlabel(u._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
    axes[-1].set_xlim(*u._display_xlim())
    fig.canvas.draw()
    lbox = leg.get_window_extent().transformed(axes[0].transAxes.inverted())
    label_arts = hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(),
                               com=u._com_label(), ax=axes[0])
    # mplhep 1.2 has no pad kwarg: lift the label artists above the legend.
    dy = max(0.0, lbox.y1 - 1.0) + 0.015
    for art in label_arts:
        if art is not None:
            x, y = art.get_position()
            art.set_position((x, y + dy))
    save_path = (f"./{u.spec.output_dir}groomed_summary_ratio.pdf"
                 if u.groomed else
                 f"./{u.spec.output_dir}ungroomed_summary_ratio.pdf")
    u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_unfolded_summary_linear(u, show=True):
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
    npt = len(u.pt_edges) - 1
    stat_label = "Stat. Unc." if u.response_matrix_stat_available else "Input Stat. Unc."
    total_label = (
        r"Syst. $\oplus$ Stat. Unc."
        if u.response_matrix_stat_available and u.has_herwig
        else r"Partial Syst. $\oplus$ Stat. Unc."
    )
    title_list = []
    for i in range(npt):
        lo = int(u.pt_edges[i])
        if i + 1 < npt:
            hi = int(u.pt_edges[i + 1])
            title_list.append(rf"{lo}$<$$p_T$$<${hi} GeV")
        else:
            title_list.append(rf"{lo}$<$$p_T$$< \, \infty$  GeV")

    fig = plt.figure(figsize=(12, 10))
    # Every slice is normalized to unit area, so they overlay directly; the
    # 10^n offsets of the log summary make no sense on a linear axis.
    scale = 1.0
    for i in u._summary_pt_indices():
        display_slice, rho_edges = u._gen_display_slice(i)
        unfolded = np.array(
            u.normalized_results[i]['unfolded'], dtype=float
        )[display_slice]
        syst_up = np.array(
            u.normalized_results[i]['syst_unc']['up'], dtype=float
        )[display_slice]
        syst_down = np.array(
            u.normalized_results[i]['syst_unc']['down'], dtype=float
        )[display_slice]
        stat_unc = np.array(
            u.normalized_results[i]['stat_unc'], dtype=float
        )[display_slice]

        y_syst_up = scale * (unfolded + syst_up)
        y_syst_down = scale * (unfolded - syst_down)
        y_stat_up = scale * (unfolded + stat_unc)
        y_stat_down = scale * (unfolded - stat_unc)

        plt.stairs(scale * np.array(
            u.normalized_results[i]['true'], dtype=float
        )[display_slice], rho_edges, label='PYTHIA8', color='b', ls='dotted', lw=3, baseline=None)
        plt.stairs(y_syst_up, rho_edges, baseline=y_syst_down, fill=True, color=u.spec.band_color_total, label=total_label, alpha=0.8)
        plt.stairs(y_stat_up, rho_edges, baseline=y_stat_down, fill=True, color=u.spec.band_color_stat, label=stat_label)
        centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
        plt.plot(centers, scale * unfolded, label=title_list[i], color='k', lw=0, marker=markers[i])

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    for h, l in zip(handles, labels):
        if l == "" or l in seen:
            try:
                h.set_label('_nolegend_')
            except Exception:
                pass
        else:
            seen.add(l)
    plt.legend(fontsize=15, loc="upper right")
    plt.xlabel(u._observable_label())
    plt.ylabel(u._normalized_ylabel())
    hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(), fontsize=20)
    plt.xlim(*u._display_xlim())
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)   # headroom for the legend

    save_path = f"./{u.spec.output_dir}groomed_summary_linear.pdf" if u.groomed else f"./{u.spec.output_dir}ungroomed_summary_linear.pdf"
    u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_herwig_pythia_comparison(u, show=True):
    """
    One plot per pT bin with three panels:
      - Main panel  : normalized gen-level jet mass for PYTHIA8 and HERWIG7.
      - Middle panel: gen-level model uncertainty (HERWIG7 - PYTHIA8) / PYTHIA8.
      - Bottom panel: post-unfolding model uncertainty, i.e. envelope of
                      |unfolded_herwigUp - nominal| / nominal and
                      |unfolded_herwigDown - nominal| / nominal.
    """
    true_herwig_pt_binned = unflatten_gen_by_pt(u.y_true_herwig, u.gen_edges_by_pt)
    npt = len(u.pt_edges) - 1
    title_list = [
        "",
        r"$p_T$ 200 – 290 GeV",
        r"$p_T$ 290 – 400 GeV",
        r"$p_T$ 400 – $\infty$ GeV",
    ]

    ratio_ymax = 0.5

    def _annotate_overflow(ax, values, edges, ymax, color, fontsize=11):
        """Write the numeric value above the clip line for bins that exceed ymax."""
        centers = 0.5 * (edges[:-1] + edges[1:])
        for x, y in zip(centers, values):
            if np.isfinite(y) and y > ymax:
                ax.text(x, ymax - 0.02, f"{y:.2f}",
                        ha='center', va='top', fontsize=fontsize,
                        color=color, clip_on=True)

    for i in range(npt):
        bin_widths = np.diff(u.gen_edges_by_pt[i])
        edges = np.array(u.gen_edges_by_pt[i], dtype=float)

        pythia_norm = np.array(u.normalized_results[i]['true'], dtype=float)
        herwig_raw = true_herwig_pt_binned[i]
        herwig_norm = herwig_raw / bin_widths / u._shown_norm_total(herwig_raw, i)

        # Unsigned gen-level difference
        gen_model_diff = np.abs(np.divide(
            herwig_norm - pythia_norm, pythia_norm,
            out=np.full_like(herwig_norm, np.nan), where=pythia_norm != 0,
        ))

        # Post-unfolding model uncertainty: envelope of herwigUp / herwigDown
        syst_frac = u._build_syst_fraction_dict(i)
        herwig_up_frac   = syst_frac.get('herwigUp',   np.zeros_like(pythia_norm))
        herwig_down_frac = syst_frac.get('herwigDown', np.zeros_like(pythia_norm))
        model_unc_unfolded = np.maximum(herwig_up_frac, herwig_down_frac)

        # PYTHIA8 / HERWIG7 ratio (gen-level, normalized)
        pythia_over_herwig = np.divide(
            pythia_norm, herwig_norm,
            out=np.full_like(pythia_norm, np.nan), where=herwig_norm != 0,
        )

        fig, (ax_main, ax_ph, ax_ratio, ax_model) = plt.subplots(
            4, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1, 1]},
            figsize=(10, 14),
        )

        # Main panel
        hep.histplot(pythia_norm, edges, label='PYTHIA8', color='#5790fc', ls='dotted', lw=3, ax=ax_main)
        hep.histplot(herwig_norm, edges, label='HERWIG7', color='#e42536', ls='dashdot', lw=2, ax=ax_main)
        ax_main.set_ylabel(u._normalized_ylabel())
        pt_title = title_list[i] if i < len(title_list) else ""
        ax_main.legend(title=pt_title, fontsize=14, title_fontsize=15)
        hep.cms.label(u.cms_label, data=False, lumi=138, com=13, fontsize=20, ax=ax_main)

        # PYTHIA8 / HERWIG7 ratio panel
        ax_ph.axhline(1.0, color='gray', ls='--', lw=1)
        hep.histplot(pythia_over_herwig, edges, color='#5790fc', ls='dotted', lw=2, ax=ax_ph)
        ax_ph.set_ylabel('P8 / H7', fontsize=13)
        ax_ph.set_ylim(0.5, 1.5)
        ax_ph.set_yticks([0.5, 1.0, 1.5])

        # Gen-level model uncertainty panel: |HERWIG7 - PYTHIA8| / PYTHIA8
        ax_ratio.axhline(0.0, color='gray', ls='--', lw=1)
        hep.histplot(gen_model_diff, edges, color='#e42536', ls='dashdot', lw=2, ax=ax_ratio)
        ax_ratio.set_ylabel('Rel. Unc. GEN', fontsize=13)
        ax_ratio.set_ylim(0, ratio_ymax)
        ax_ratio.set_yticks([0.0, 0.25, 0.5])
        _annotate_overflow(ax_ratio, gen_model_diff, edges, ratio_ymax, color='#e42536')

        # Post-unfolding model uncertainty panel
        ax_model.axhline(0.0, color='gray', ls='--', lw=1)
        hep.histplot(model_unc_unfolded, edges, color='#7a21dd', ls='solid', lw=2, ax=ax_model)
        ax_model.set_ylabel('Rel. Unc. Unfolded', fontsize=13)
        ax_model.set_ylim(0, ratio_ymax)
        ax_model.set_yticks([0.0, 0.25, 0.5])
        ax_model.set_xlabel(u._observable_label())
        _annotate_overflow(ax_model, model_unc_unfolded, edges, ratio_ymax, color='#7a21dd')

        for ax in (ax_main, ax_ph, ax_ratio, ax_model):
            ax.set_xlim(*u._observable_xlim(i))

        ax_model.tick_params(axis='x', pad=8)
        ax_model.tick_params(axis='y', pad=6)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)
        suffix = "groomed" if u.groomed else "ungroomed"
        save_path = f"./{u.spec.output_dir}herwig_pythia_comparison_{suffix}_{i-1}.pdf"
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_unfolded(u, log=False, show=True):

    unfolded_pt_binned = unflatten_gen_by_pt(u.y_unf, u.gen_edges_by_pt)
    measured_pt_binned = unflatten_gen_by_pt(u.y_meas, u.reco_edges_by_pt)
    reco_mc_pt_binned = unflatten_gen_by_pt(u.mosaic.sum(axis = 1), u.reco_edges_by_pt)
    true_pt_binned = unflatten_gen_by_pt(u.y_true, u.gen_edges_by_pt)
    true_herwig_pt_binned = (
        unflatten_gen_by_pt(u.y_true_herwig, u.gen_edges_by_pt)
        if getattr(u, "has_herwig", True)
        else None
    )
    #error_pt_binned = unflatten_gen_by_pt(u.ye_unf, u.gen_edges_by_pt)
    u.normalized_herwig = []
    #print("Herwig pt Binned", true_herwig_pt_binned)
    u.herwig_closure_unc = []
    for i in u._reported_pt_indices():
        yerr = u.normalized_results[i]['syst_unc']['up']
        bin_widths = np.diff(u.gen_edges_by_pt[i])
        bin_widths_reco = np.diff(u.reco_edges_by_pt[i])
        #u.normalized_herwig.append(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum())
        if u.herwig_closure:
            hep.histplot(true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum(), u.gen_edges_by_pt[i], color='#964a8b', label = 'Herwig', alpha = 0.7, ls = 'dotted')
        else:
            hep.histplot(true_pt_binned[i]/bin_widths/true_pt_binned[i].sum(), u.gen_edges_by_pt[i], color='#5790fc', label = 'PYTHIA', alpha = 0.8, ls = 'dotted', lw = 3)
        hep.histplot(unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum(), u.gen_edges_by_pt[i], label = 'Unfolded Herwig' if u.herwig_closure else 'Unfolded', color = 'k', ls = '--' )

        

        #hep.histplot(measured_pt_binned[i]/bin_widths_reco/measured_pt_binned[i].sum(), u.reco_edges_by_pt[i], color = 'k', ls= '--', alpha= 0.5, label = 'Meas' )
        #dhep.histplot(reco_mc_pt_binned[i]/bin_widths_reco/reco_mc_pt_binned[i].sum(), u.reco_edges_by_pt[i], color = 'g', ls= '--', alpha= 0.5, label = 'Reco_MC' )
        title = f" {int(u.pt_edges[i])}-{int(u.pt_edges[i+1]) if i+1 < len(u.pt_edges)-1 else '∞'} GeV"
        plt.legend(title = title, fontsize = 18) 
        
        if u.groomed:
            #plt.xlim(0,250)
            plt.xlim(*u._observable_xlim(i))
            plt.xlabel(u._observable_label())
        #plt.ylim(0,0.02)
        if not u.groomed:
            plt.xlim(*u._observable_xlim(i))
            plt.xlabel(u._observable_label())
        save_path = f"./{u.spec.output_dir}unfold/unfolded_basic_groomed_{i-1}.pdf" if u.groomed else f"./{u.spec.output_dir}unfold/unfolded_basic_ungroomed_{i-1}.pdf"
        u._finalize_plot(save_path=save_path, show=show)
        # Plot relative difference: (true - unfolded) / true after normalization
        # true_norm = true_pt_binned[i] / np.diff(u.gen_edges_by_pt[i]) / true_pt_binned[i].sum()
        # true_norm = true_herwig_pt_binned[i] / np.diff(u.gen_edges_by_pt[i]) / true_herwig_pt_binned[i].sum()
        # unfolded_norm = unfolded_pt_binned[i] / np.diff(u.gen_edges_by_pt[i]) / unfolded_pt_binned[i].sum()
        
        # rel_diff = np.abs(true_norm - unfolded_norm) / true_norm
        # hep.histplot(rel_diff, u.gen_edges_by_pt[i], label="(Herwig - Unfolded) / Herwig", color="r")
        # title = f"pT bin: {int(u.pt_edges[i])}-{int(u.pt_edges[i+1]) if i+1 < len(u.pt_edges)-1 else '∞'} GeV"
        # plt.legend(title = title) 
        if u.herwig_closure:
            plt.figure(figsize=(10, 3))
            true_herwig = true_herwig_pt_binned[i]/bin_widths/true_herwig_pt_binned[i].sum()
            unfolded = unfolded_pt_binned[i]/bin_widths/unfolded_pt_binned[i].sum()
            herwig_closure_unc = np.abs(true_herwig - unfolded) / true_herwig
            u.herwig_closure_unc.append(herwig_closure_unc)
            plt.stairs(herwig_closure_unc, u.gen_edges_by_pt[i], label = 'Closure Unc (|Herwig - Unfolded| / Herwig)', color='#7a21dd', ls = 'dotted')
            if not u.groomed:
                plt.xlim(*u._observable_xlim(i))
            else:
                plt.xlim(*u._observable_xlim(i))
            plt.ylim(0, 1)
            plt.xlabel(u._observable_label())
            #plt.legend()
            groomed_tag = "groomed" if u.groomed else "ungroomed"
            save_path = f"./{u.spec.output_dir}unfold/herwig_closure_unc_{groomed_tag}_{i-1}.pdf"
            u._finalize_plot(save_path=save_path, show=show)
    # Save uncertainty in a file for later use
    if u.herwig_closure:
        groomed_tag = "groomed" if u.groomed else "ungroomed"
        np.save(f"{u.spec.input_dir}herwig_closure_unc_{u.spec.name}_{groomed_tag}.npy", u.herwig_closure_unc)
        # if u.groomed:
        #     plt.xlim(0,250)
        #     plt.xlabel("Groomed Jet Mass (GeV)" if u.groomed else "Ungroomed Jet Mass (GeV)")
        # #plt.ylim(0,0.02)
        # if not u.groomed:
        #     plt.xlim(20,250)
        #     plt.xlabel("Groomed Jet Mass (GeV)" if u.groomed else "Ungroomed Jet Mass (GeV)")
        # plt.show()


def plot_herwig_bias_test(u, show=True):
    """HERWIG bias (non-closure) test: PYTHIA matrix unfolds HERWIG reco.

    Unfolds the HERWIG reco spectrum with the nominal PYTHIA response and
    compares the result to HERWIG gen. Each panel shows:

      * Unfolded HERWIG as a dashed line.
      * HERWIG gen with its MC-stat uncertainty drawn as a band.
      * a ratio panel with the per-bin |HERWIG - Unfolded| / HERWIG and the
        HERWIG gen MC-stat uncertainty as a shaded envelope.

    The per-bin |HERWIG - Unfolded| / HERWIG is also saved as the
    model-dependence uncertainty input (``herwig_closure_unc_*.npy``).
    """
    if not getattr(u, "has_herwig", True) or u.y_true_herwig is None:
        return

    hep.style.use("CMS")
    u._ensure_herwig_bias_inputs()
    y_unf_herwig, _ = u._unfold_herwig_through_pythia()

    unfolded_pt_binned = unflatten_gen_by_pt(y_unf_herwig, u.gen_edges_by_pt)
    true_herwig_pt_binned = unflatten_gen_by_pt(
        u.y_true_herwig, u.gen_edges_by_pt
    )
    # HERWIG gen MC-stat error per flattened gen bin (sqrt of sumw2), or
    # None when the input pkls carry no variances.
    herwig_gen_err = (
        np.sqrt(np.clip(u.herwig_gen_var_flat, 0.0, None))
        if getattr(u, "herwig_gen_var_flat", None) is not None
        else None
    )
    herwig_gen_err_pt_binned = (
        unflatten_gen_by_pt(herwig_gen_err, u.gen_edges_by_pt)
        if herwig_gen_err is not None
        else None
    )

    def _rel(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(
                np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den != 0)
            )

    groomed_tag = "groomed" if u.groomed else "ungroomed"
    u.herwig_closure_unc = []
    for i in u._reported_pt_indices():
        edges = np.asarray(u.gen_edges_by_pt[i], dtype=float)
        bin_widths = np.diff(edges)

        herwig_sum = true_herwig_pt_binned[i].sum()
        unfolded_sum = unfolded_pt_binned[i].sum()
        herwig = true_herwig_pt_binned[i] / bin_widths / herwig_sum
        unfolded = unfolded_pt_binned[i] / bin_widths / unfolded_sum

        # HERWIG gen MC-stat (bin-width and per-slice normalization cancel
        # in the ratio, so apply the relative error to the density directly).
        # NB: TUnfold's propagated "unfolding uncertainty" on the HERWIG
        # pseudo-data is intentionally NOT drawn here -- closure mode feeds
        # no measured variances, so it is the sqrt(weighted-content) Poisson
        # error of weighted MC propagated through the unregularized inverse,
        # which is meaningless (and ~100%+) for a bias test.
        if herwig_gen_err_pt_binned is not None:
            herwig_rel = _rel(herwig_gen_err_pt_binned[i], true_herwig_pt_binned[i])
        else:
            herwig_rel = np.zeros_like(herwig)
        herwig_band = herwig * herwig_rel

        closure_unc = _rel(herwig - unfolded, herwig)
        u.herwig_closure_unc.append(closure_unc)

        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # HERWIG gen + MC-stat band
        hep.histplot(
            herwig, edges, ax=ax, color="#964a8b",
            label="Herwig (gen)", alpha=0.9, ls="dotted", lw=2,
        )
        if np.any(herwig_band > 0):
            ax.fill_between(
                edges, np.r_[herwig - herwig_band, (herwig - herwig_band)[-1]],
                np.r_[herwig + herwig_band, (herwig + herwig_band)[-1]],
                step="post", color="#964a8b", alpha=0.2, lw=0,
                label="Herwig stat. unc.",
            )

        # Unfolded HERWIG. The TUnfold uncertainty is intentionally omitted;
        # see the note above where the HERWIG MC-stat band is constructed.
        hep.histplot(
            unfolded, edges, ax=ax, color="k", ls="--", lw=2,
            label="Unfolded Herwig",
        )

        title = (
            f" {int(u.pt_edges[i])}-"
            f"{int(u.pt_edges[i + 1]) if i + 1 < len(u.pt_edges) - 1 else '∞'} GeV"
        )
        ax.legend(title=title, fontsize=15)
        ax.set_ylabel(u._normalized_ylabel())
        hep.cms.label(
            u.cms_label, data=False, lumi=u._lumi_label(),
            com=u._com_label(), fontsize=20, ax=ax,
        )

        # Ratio panel: compare the observed non-closure with HERWIG gen
        # MC statistics only.
        if np.any(herwig_rel > 0):
            ax_ratio.fill_between(
                edges, 0.0, np.r_[herwig_rel, herwig_rel[-1]],
                step="post", color="#964a8b", alpha=0.2, lw=0,
                label="Herwig stat. unc.",
            )
        ax_ratio.stairs(
            closure_unc, edges, color="#7a21dd", lw=2,
            label="|Herwig - Unfolded| / Herwig",
        )
        ax_ratio.set_ylim(0, 1)
        ax_ratio.set_ylabel("Rel. diff.")
        ax_ratio.set_xlim(*u._observable_xlim(i))
        ax_ratio.set_xlabel(u._observable_label())
        ax_ratio.legend(fontsize=11, loc="upper left")

        save_path = (
            f"./{u.spec.output_dir}unfold/"
            f"herwig_bias_test_{groomed_tag}_{i - 1}.pdf"
        )
        u._finalize_plot(save_path=save_path, show=show, fig=fig)

    # Persist the non-closure as the model-dependence systematic input.
    np.save(
        f"{u.spec.input_dir}herwig_closure_unc_{u.spec.name}_{groomed_tag}.npy",
        np.array(u.herwig_closure_unc, dtype=object),
    )


def plot_unfolded_unrolled_2d(u, show=True):
    """Full unrolled (mass x pT) unfolded data vs PYTHIA gen, absolute.

    Mirrors the reco-level input overlay, but at gen level: the unfolded
    result and the PYTHIA truth are shown at their original (un-normalized)
    absolute values on a linear y-axis, with a Data/MC ratio pad and dotted
    pT-slice dividers. The 2D-normalized arrays are still computed/saved by
    ``_compute_2d_normalized_result`` / ``save_2d_unfolded``.
    """
    hep.style.use("CMS")
    unf = np.asarray(u.unfolded_abs_flat, dtype=float)
    unf_err = np.asarray(u.unfolded_abs_err_flat, dtype=float)
    true = np.asarray(u.y_true, dtype=float)
    n = len(unf)
    x = np.arange(n)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.step(x, true, where="mid", color="blue", lw=1.4, label="PYTHIA8 (gen)")
    ax.errorbar(
        x, unf, yerr=unf_err, fmt="o", ms=3, color="black", lw=0.8,
        label="Unfolded data",
    )
    ax.set_ylabel("Unfolded events (absolute)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=13)
    hep.cms.label(
        u.cms_label, data=True, lumi=u._lumi_label(),
        com=u._com_label(), ax=ax, fontsize=18,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(unf, true, out=np.zeros_like(unf), where=true > 0)
        ratio_err = np.divide(unf_err, true, out=np.zeros_like(unf), where=true > 0)
    axr.axhline(1, color="gray", ls="--", lw=1)
    axr.errorbar(x, ratio, yerr=ratio_err, fmt="o", ms=3, color="black")
    axr.set_ylabel("Data / MC")
    axr.set_ylim(0.5, 1.5)
    axr.set_xlabel(
        "Unrolled bin index  (mass within each $p_T$ slice, slices concatenated)"
    )
    axr.set_xlim(-1, n)

    counts_per_slice = [len(edges) - 1 for edges in u.gen_edges_by_pt]
    boundaries = np.cumsum(counts_per_slice)
    pt_edges = np.asarray(u.pt_edges, dtype=float)
    start = 0
    for i, b in enumerate(boundaries):
        for a in (ax, axr):
            a.axvline(b - 0.5, color="steelblue", ls=":", lw=1, alpha=0.7)
        lo = pt_edges[i]
        hi = pt_edges[i + 1] if i + 1 < len(pt_edges) else np.inf
        lbl = f"{lo:g}-{'∞' if not np.isfinite(hi) or hi >= 13000 else f'{hi:g}'}"
        ax.text(
            (start + b - 1) / 2, ax.get_ylim()[1] * 0.5, lbl,
            ha="center", va="top", fontsize=9, color="steelblue", rotation=90,
        )
        start = b

    suffix = "groomed" if u.groomed else "ungroomed"
    save_path = f"./{u.spec.output_dir}unfold/unfolded_unrolled_2d_{suffix}.pdf"
    u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_jackknife_convergence(u, show=True):
    """Full sheet of jackknife stat convergence, one panel per (pT, rho) bin.

    Each panel shows the fractional statistical uncertainty (input/data,
    response-matrix/MC, and their quadrature sum) as a function of the
    number of jackknife replicas used (2..N). A flattening curve indicates
    the replica count is sufficient.
    """
    conv = u._jackknife_convergence_fractions()
    if conv is None:
        return
    ns, input_frac, matrix_frac, total_frac = conv

    counts = [len(edges) - 1 for edges in u.gen_edges_by_pt]
    n_pt = len(counts)
    n_col = max(counts)
    fig, axes = plt.subplots(
        n_pt, n_col, figsize=(2.4 * n_col, 2.3 * n_pt),
        squeeze=False, sharex=True,
    )
    offset = 0
    for pt_i, count in enumerate(counts):
        edges = np.asarray(u.gen_edges_by_pt[pt_i], dtype=float)
        lo = u.pt_edges[pt_i]
        hi = u.pt_edges[pt_i + 1] if pt_i + 1 < len(u.pt_edges) else np.inf
        pt_lbl = f"{lo:g}-{'∞' if not np.isfinite(hi) or hi >= 13000 else f'{hi:g}'} GeV"
        for r in range(n_col):
            ax = axes[pt_i][r]
            if r >= count:
                ax.axis("off")
                continue
            j = offset + r
            ax.plot(ns, input_frac[:, j], "o-", ms=3, lw=1, color="#1f77b4", label="input (data)")
            ax.plot(ns, matrix_frac[:, j], "s-", ms=3, lw=1, color="#ff7f0e", label="matrix (MC)")
            ax.plot(ns, total_frac[:, j], "^-", ms=3, lw=1.3, color="k", label="total")
            ax.axhline(total_frac[-1, j], color="gray", ls=":", lw=0.8)
            ax.set_title(rf"$\rho\in[{edges[r]:g},{edges[r+1]:g})$", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_ylabel(f"{pt_lbl}\nfrac. stat", fontsize=8)
            if pt_i == n_pt - 1:
                ax.set_xlabel("# replicas", fontsize=8)
        offset += count

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, ncol=3)
    mode = "groomed" if u.groomed else "ungroomed"
    fig.suptitle(
        f"Jackknife stat-uncertainty convergence vs # replicas ({mode})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = f"./{u.spec.output_dir}unfold/jackknife_convergence_{mode}.pdf"
    u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_jackknife_convergence_pt_avg(u, show=True, pt_min=None):
    """Jackknife stat convergence averaged over rho within each pT bin.

    Same diagnostic as plot_jackknife_convergence, but instead of one panel
    per (pT, rho) gen bin it collapses the rho dimension: each panel is one
    pT slice and shows the mean over that slice's rho bins of the fractional
    statistical uncertainty (input/data, response-matrix/MC, and their
    quadrature sum) as a function of the number of jackknife replicas used
    (2..N). A flattening curve indicates the replica count is sufficient.

    Drawn in the mplhep CMS style for the analysis note. ``pt_min`` drops
    pT slices whose upper edge is <= pt_min from the panels (the rho average
    within each kept slice is unaffected).
    """
    conv = u._jackknife_convergence_fractions()
    if conv is None:
        return
    ns, input_frac, matrix_frac, total_frac = conv

    counts = [len(edges) - 1 for edges in u.gen_edges_by_pt]
    # Per-slice rho-averaged curves, keeping the flat-array offsets aligned
    # even for slices we end up dropping from the panels.
    panels = []
    offset = 0
    for pt_i, count in enumerate(counts):
        sl = slice(offset, offset + count)
        offset += count
        lo = u.pt_edges[pt_i]
        hi = u.pt_edges[pt_i + 1] if pt_i + 1 < len(u.pt_edges) else np.inf
        if pt_min is not None and np.isfinite(hi) and hi <= pt_min:
            continue
        panels.append((
            lo, hi, count,
            input_frac[:, sl].mean(axis=1),
            matrix_frac[:, sl].mean(axis=1),
            total_frac[:, sl].mean(axis=1),
        ))

    if not panels:
        return

    n_panel = len(panels)
    n_col = min(n_panel, 4)
    n_row = int(np.ceil(n_panel / n_col))
    mode = "groomed" if u.groomed else "ungroomed"

    with plt.style.context(hep.style.CMS):
        fig, axes = plt.subplots(
            n_row, n_col, figsize=(5.6 * n_col, 5.4 * n_row),
            squeeze=False, sharex=True, layout="constrained",
        )
        for idx, (lo, hi, count, inp, mat, tot) in enumerate(panels):
            ax = axes[idx // n_col][idx % n_col]
            ax.plot(ns, inp, "o-", ms=8, lw=2.0, color="#5790fc", label="Input (data)")
            ax.plot(ns, mat, "s-", ms=8, lw=2.0, color="#f89c20", label="Matrix (MC)")
            ax.plot(ns, tot, "^-", ms=9, lw=2.6, color="k", label="Total")
            ax.axhline(tot[-1], color="gray", ls=":", lw=1.4)

            if np.isfinite(hi) and hi < 13000:
                pt_lbl = rf"${lo:g} < p_{{T}} < {hi:g}$ GeV"
            else:
                pt_lbl = rf"$p_{{T}} > {lo:g}$ GeV"
            ax.text(
                0.05, 0.95,
                pt_lbl + "\n" + rf"$\langle\,{count}\ \rho$ bins$\,\rangle$",
                transform=ax.transAxes, ha="left", va="top", fontsize=18,
            )

            vmax = max(float(inp.max()), float(mat.max()), float(tot.max()))
            ax.set_ylim(0.0, vmax * 1.45)  # headroom for the pT label
            ax.set_xlim(ns[0] - 0.4, ns[-1] + 0.4)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=17)
            ax.set_xlabel("Number of jackknife replicas", fontsize=19)

        for k in range(n_panel, n_row * n_col):
            axes[k // n_col][k % n_col].axis("off")

        # CMS multi-panel convention: "CMS Internal" over the first panel,
        # the luminosity over the last panel.
        first_ax = axes[0][0]
        last_ax = axes[(n_panel - 1) // n_col][(n_panel - 1) % n_col]
        first_ax.set_ylabel(
            "Mean fractional\nstatistical uncertainty", fontsize=19
        )
        hep.cms.label(u.cms_label, data=True, ax=first_ax, fontsize=20, rlabel="")
        last_ax.text(
            1.0, 1.01,
            rf"{u._lumi_label()} fb$^{{-1}}$ ({u._com_label()} TeV)",
            transform=last_ax.transAxes, ha="right", va="bottom", fontsize=18,
        )

        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside lower center", ncol=3, fontsize=17,
            frameon=False,
        )
        save_path = (
            f"./{u.spec.output_dir}unfold/"
            f"jackknife_convergence_pt_avg_{mode}.pdf"
        )
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_statistical_fraction(u, show=True):
    for i in u._reported_pt_indices():
        result = u.normalized_results[i]
        plt.figure()
        pt_bin = result['pt_bin']
        input_stat_fraction = result['input_stat_unc_frac']
        matrix_stat_fraction = result['matrix_stat_unc_frac']

        hep.histplot(
            input_stat_fraction[1:],
            u.gen_edges_by_pt[i][1:],
            label="Input statistical Uncertainty",
            ls="--",
        )
        # The matrix (response-stat) term only exists when jackknife
        # replicas were run; the covariance fallback leaves it identically
        # zero, and drawing that flat line just advertises a missing input.
        draw_matrix = (
            u.response_matrix_stat_available
            and bool(np.any(matrix_stat_fraction[1:]))
        )
        if draw_matrix:
            hep.histplot(
                matrix_stat_fraction[1:],
                u.gen_edges_by_pt[i][1:],
                label="Matrix uncertainty",
                ls="-.",
            )

        pt_bin_label = u._pt_bin_label(pt_bin)

        plt.legend(title=rf"$p_T$  {pt_bin_label} GeV")
        hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(), fontsize=20)
        plt.xlim(*u._display_xlim(i))
        # The y autoscale sees the buffer bins hidden outside the xlim
        # window (groomed panels blew up to 2-6 while the shown curve sits
        # at a few percent) -- scale to the shown bins only.
        edges = np.asarray(u.gen_edges_by_pt[i][1:], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xlo, xhi = u._observable_xlim(i)
        visible = (centers >= xlo) & (centers <= xhi)
        if np.any(visible):
            curves = [np.asarray(input_stat_fraction[1:], float)[visible]]
            if draw_matrix:
                curves.append(
                    np.asarray(matrix_stat_fraction[1:], float)[visible])
            vis_max = max(float(c.max()) for c in curves)
            if vis_max > 0:
                plt.ylim(0, 1.3 * vis_max)
        plt.xlabel(u._observable_label())
        plt.ylabel("Fractional Uncertainty")
        ax = plt.gca()
        ax.tick_params(axis='x', pad=8)
        ax.tick_params(axis='y', pad=8)
        plt.subplots_adjust(left=0.16, bottom=0.15)
        save_path = (
            f'./{u.spec.output_dir}uncertainties/stat_fraction_groomed_{u._output_panel_index(i)}.pdf'
            if u.groomed
            else f'./{u.spec.output_dir}uncertainties/stat_fraction_ungroomed_{u._output_panel_index(i)}.pdf'
        )
        u._finalize_plot(save_path=save_path, show=show)


def plot_systematic_fraction(u, syst_name='all', show=True, log=True):
    _plot_systematic_fraction_summary(u, grouped=False, show=show, log=log)


def _plot_systematic_fraction_summary(u, grouped=False, show=True, log=True):
    hep.style.use("CMS")
    u.syst_fraction_dicts = []
    grouped_legend_order = [
        "Jet Energy",
        "Jet Mass",
        "Parton Shower",
        "Lepton SFs",
        "Other Theory",
        "Model Uncertainty",
        "Shower Model",
        "Hadronization Model",
        "Stat Unc",
        "Total",
    ]
    # ARC round-2 casing for the legend; internal group keys unchanged.
    display_name = {
        "Jet Energy": "Jet energy",
        "Jet Mass": "Jet mass",
        "Parton Shower": "Parton shower",
        "Lepton SFs": "Lepton scale factors",
        "Other Theory": "Other theory",
        "Model Uncertainty": "Model uncertainty",
        "Shower Model": "Shower model",
        "Hadronization Model": "Hadronization model",
        "Stat Unc": "Stat unc.",
    }
    # Grouped entries on the CVD-friendly Petroff palette (ARC round-2
    # color check); ungrouped per-systematic colors on the 10-color set.
    summary_style_map = {
        "JES": {"color": "#3f90da", "ls": "-"},
        "JER": {"color": "#ffa90e", "ls": "-"},
        "Pileup": {"color": "#bd1f01", "ls": "-"},
        "Electron RECO": {"color": "#832db6", "ls": "-"},
        "Electron ID": {"color": "#a96b59", "ls": "-"},
        "Electron Trigger": {"color": "#e76300", "ls": "-"},
        "Muon RECO": {"color": "#94a4a2", "ls": "-"},
        "Muon ID": {"color": "#b9ac70", "ls": "-"},
        "Muon Trigger": {"color": "#92dadd", "ls": "-"},
        "Muon ISO": {"color": "#717581", "ls": "-"},
        "PDF": {"color": "#f89c20", "ls": "--"},
        "Q2 Scale": {"color": "#5790fc", "ls": "--"},
        "L1 Prefiring": {"color": "#964a8b", "ls": "--"},
        "ISR": {"color": "#e42536", "ls": "--"},
        "FSR": {"color": "#7a21dd", "ls": "--"},
        "JMR": {"color": "#9c9ca1", "ls": "--"},
        "JMS": {"color": "#bd1f01", "ls": ":"},
        "Jet Energy": {"color": "#5790fc", "ls": "-", "lw": 2.5},
        "Jet Mass": {"color": "#f89c20", "ls": "-", "lw": 2.5},
        "Parton Shower": {"color": "#e42536", "ls": "-.", "lw": 2.5},
        "Lepton SFs": {"color": "#964a8b", "ls": "-", "lw": 2.5},
        "Other Theory": {"color": "#9c9ca1", "ls": "-", "lw": 2.5},
        "Model Uncertainty": {"color": "#7a21dd", "ls": "-.", "lw": 2.5},
        "Shower Model": {"color": "#e42536", "ls": "-.", "lw": 2.5},
        "Hadronization Model": {"color": "#7a21dd", "ls": "--", "lw": 2.5},
        "Stat Unc": {"color": "k", "ls": ":", "lw": 2.5},
    }

    for i in u._reported_pt_indices():
        result = u.normalized_results[i]
        # CMS-default canvas (a forced figsize shrinks every font in the
        # scaled-down PDF; ARC round-2 "make all labels bigger").
        fig = plt.figure(layout="constrained")
        pt_bin = result["pt_bin"]
        syst_fraction_dict = u._build_syst_fraction_dict(i)
        result["syst_fraction_dict"] = syst_fraction_dict
        u.syst_fraction_dicts.append(syst_fraction_dict)

        plot_fraction_dict = u._group_syst_fraction_dict(syst_fraction_dict, grouped=grouped)
        rho_edges = np.asarray(u.gen_edges_by_pt[i], dtype=float)
        rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
        visible_mask = u._complete_bin_mask(rho_edges, u._display_xlim(i))
        visible_total = np.asarray(plot_fraction_dict["Total_Up"], dtype=float)[
            visible_mask
        ]
        visible_max = float(np.max(visible_total)) if visible_total.size else 0.0
        # Reserve the upper part of the canvas for the in-frame legend.
        # The uncertainty itself remains unmodified; only the axis gets
        # panel-specific headroom based on bins that are actually shown.
        linear_ymax = max(
            0.05,
            0.025 * np.ceil(max(visible_max * 1.9, 0.05) / 0.025),
        )

        # ARC round-2 s5ff: the total drawn as a filled silhouette BEHIND
        # the components, so the line hugging its top edge in each bin is
        # visibly the dominant contribution.
        hep.histplot(
            plot_fraction_dict["Total_Up"],
            u.gen_edges_by_pt[i],
            histtype="fill",
            facecolor="0.88",
            edgecolor="black",
            linewidth=2,
            label="Total",
        )

        plotted_labels = set()
        for syst_name, syst_fraction in plot_fraction_dict.items():
            label, variation = u._split_systematic_variation(syst_name)
            if syst_name in {"Stat Unc", "Total_Up", "Total_Down"} or variation == "Down":
                continue

            if label in plotted_labels:
                continue

            style = summary_style_map.get(label, {"ls": "-"})
            hep.histplot(
                syst_fraction,
                u.gen_edges_by_pt[i],
                label=display_name.get(label, label),
                **style,
            )
            plotted_labels.add(label)

        hep.histplot(
            plot_fraction_dict["Stat Unc"],
            u.gen_edges_by_pt[i],
            label=display_name["Stat Unc"],
            **summary_style_map["Stat Unc"],
        )

        if not log:
            ax = plt.gca()
            for x_pos, y_val in zip(rho_centers, np.asarray(plot_fraction_dict["Total_Up"], dtype=float)):
                if y_val > linear_ymax:
                    ax.text(
                        x_pos,
                        linear_ymax - 0.015,
                        f"{y_val:.2f}",
                        ha="center",
                        va="top",
                        fontsize=PUB_LEGEND_FONTSIZE - 4,
                        clip_on=True,
                    )

        if log:
            plt.yscale("log")
        pt_bin_label = u._pt_bin_label(pt_bin)

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # ARC round-2: legend in-canvas
        if grouped:
            display_order = [display_name.get(l, l) for l in grouped_legend_order]
            label_to_handle = dict(zip(labels, handles))
            ordered_labels = [label for label in display_order if label in label_to_handle]
            ordered_handles = [label_to_handle[label] for label in ordered_labels]
            ax.legend(
                ordered_handles,
                ordered_labels,
                title=rf"$p_{{\mathrm{{T}}}}$  {pt_bin_label} GeV",
                loc="upper right",
                ncol=2,
                fontsize=PUB_LEGEND_FONTSIZE,
                title_fontsize=PUB_LEGEND_FONTSIZE,
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=1.0,
                columnspacing=1.0,
                handlelength=2.2,
            )
        else:
            ax.legend(
                title=rf"$p_{{\mathrm{{T}}}}$  {pt_bin_label} GeV",
                loc="upper right",
                fontsize=13,
                title_fontsize=15,
                ncol=2,
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=1.0,
            )
        if log:
            # the 18-entry per-source legend needs a decade of headroom
            plt.ylim(10e-5, 1 if grouped else 10)
        else:
            plt.ylim(0, linear_ymax)
        plt.xlim(*u._display_xlim(i))
        #plt.xlim(0,200)
        ax.set_xlabel(u._observable_label(), fontsize=PUB_LABEL_FONTSIZE)
        ax.set_ylabel("Fractional uncertainty", fontsize=PUB_LABEL_FONTSIZE)
        ax.tick_params(axis="x", pad=8, labelsize=PUB_TICK_FONTSIZE)
        ax.tick_params(axis="y", pad=8, labelsize=PUB_TICK_FONTSIZE)
        # Stamp the CMS label LAST, after the axis labels: under
        # constrained layout a later set_ylabel shrinks the axes width and
        # the (already frozen) CMS<->suffix gap collapses. The draw lets
        # the layout settle before mplhep measures the text. Above-frame
        # tag here: the grey Total silhouette can fill the top-left
        # (ungroomed first bin), so in-frame loc=2 collides with it.
        fig.canvas.draw()
        hep.cms.label(u._cms_extra_label(), data=True, lumi=u._lumi_label(), com=u._com_label(), ax=ax)

        panel_index = u._output_panel_index(i)
        if grouped:
            if log:
                save_path = (
                    f"./{u.spec.output_dir}uncertainties/summary_grouped_groomed_{panel_index}.pdf"
                    if u.groomed
                    else f"./{u.spec.output_dir}uncertainties/summary_grouped_ungroomed_{panel_index}.pdf"
                )
            else:
                save_path = (
                    f"./{u.spec.output_dir}uncertainties/summary_grouped_linear_groomed_{panel_index}.pdf"
                    if u.groomed
                    else f"./{u.spec.output_dir}uncertainties/summary_grouped_linear_ungroomed_{panel_index}.pdf"
                )
        else:
            if log:
                save_path = (
                    f"./{u.spec.output_dir}uncertainties/summary_groomed_{panel_index}.pdf"
                    if u.groomed
                    else f"./{u.spec.output_dir}uncertainties/summary_ungroomed_{panel_index}.pdf"
                )
            else:
                save_path = (
                    f"./{u.spec.output_dir}uncertainties/summary_linear_groomed_{panel_index}.pdf"
                    if u.groomed
                    else f"./{u.spec.output_dir}uncertainties/summary_linear_ungroomed_{panel_index}.pdf"
                )
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_systematic_fraction_grouped(u, show=True, log=True):
    _plot_systematic_fraction_summary(u, grouped=True, show=show, log=log)


def plot_nominal_minus_variation(u, syst_names=["pu"], show=True):
    for i in u._reported_pt_indices():
        result = u.normalized_results[i]
        fig, ax = plt.subplots(figsize=(12, 8))
        nominal = result["unfolded"]
        pt_bin = result["pt_bin"]

        for syst in syst_names:
            up_key, down_key = u._resolve_raw_systematic_pair(syst)
            label = u._get_systematic_label(syst)

            if up_key in u.normalized_systematics[i]["unfolded"]:
                up_shift = nominal - u.normalized_systematics[i]["unfolded"][up_key]
                hep.histplot(
                    up_shift,
                    u.gen_edges_by_pt[i],
                    label=f"{label}: nominal - Up",
                    ls="-",
                )

            if down_key in u.normalized_systematics[i]["unfolded"]:
                down_shift = nominal - u.normalized_systematics[i]["unfolded"][down_key]
                hep.histplot(
                    down_shift,
                    u.gen_edges_by_pt[i],
                    label=f"{label}: nominal - Down",
                    ls="--",
                )

        pt_bin_label = u._pt_bin_label(pt_bin)

        ax.axhline(0.0, color="k", lw=1, alpha=0.5)
        ax.legend(title=rf"$p_T$  {pt_bin_label} GeV", fontsize=15)
        hep.cms.label(u._cms_extra_label(), data=True, lumi=138, com=13, fontsize=20, ax=ax)
        #ax.set_xlim(*u._observable_xlim(i))
        ax.set_xlim(0,200)
        ax.set_xlabel(u._observable_label())
        ax.set_ylabel("Nominal - Variation")
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=8)

        save_stub = "_".join(str(name) for name in syst_names)
        save_path = (
            f"./{u.spec.output_dir}uncertainties/nominal_minus_{save_stub}_groomed_{i-1}.pdf"
            if u.groomed
            else f"./{u.spec.output_dir}uncertainties/nominal_minus_{save_stub}_ungroomed_{i-1}.pdf"
        )
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_systematic_frac_indiv(u, syst_names=['JES', 'JER'], ylim=None, show=True):
    def build_plot_fraction_dict(raw_syst_fraction_dict):
        plot_fraction_dict = u._group_syst_fraction_dict(raw_syst_fraction_dict, grouped=False)

        electron_keys = [key for key in plot_fraction_dict if key.startswith("Electron ")]
        if electron_keys:
            electron_up = [plot_fraction_dict[key] for key in electron_keys if key.endswith("Up")]
            electron_down = [plot_fraction_dict[key] for key in electron_keys if key.endswith("Down")]
            if electron_up:
                plot_fraction_dict["ElectronSFUp"] = np.sqrt(np.sum([value**2 for value in electron_up], axis=0))
            if electron_down:
                plot_fraction_dict["ElectronSFDown"] = np.sqrt(np.sum([value**2 for value in electron_down], axis=0))

        muon_keys = [key for key in plot_fraction_dict if key.startswith("Muon ")]
        if muon_keys:
            muon_up = [plot_fraction_dict[key] for key in muon_keys if key.endswith("Up")]
            muon_down = [plot_fraction_dict[key] for key in muon_keys if key.endswith("Down")]
            if muon_up:
                plot_fraction_dict["MuonSFUp"] = np.sqrt(np.sum([value**2 for value in muon_up], axis=0))
            if muon_down:
                plot_fraction_dict["MuonSFDown"] = np.sqrt(np.sum([value**2 for value in muon_down], axis=0))

        return plot_fraction_dict

    def resolve_syst_keys(syst_fraction_dict, syst):
        alias_bases = [syst]
        summary_name = u._get_systematic_summary_name(syst, grouped=False)
        if summary_name not in alias_bases:
            alias_bases.append(summary_name)
        label_name = u._get_systematic_label(syst)
        if label_name not in alias_bases:
            alias_bases.append(label_name)

        for base_name in alias_bases:
            up_key = f"{base_name}Up"
            down_key = f"{base_name}Down"
            if up_key in syst_fraction_dict or down_key in syst_fraction_dict:
                return up_key, down_key

        return f"{syst}Up", f"{syst}Down"

    # First, collect all values to determine global y-range
    all_values = []
    for i in u._reported_pt_indices():
        result = u.normalized_results[i]
        raw_syst_fraction_dict = result.get('syst_fraction_dict', {})
        syst_fraction_dict = build_plot_fraction_dict(raw_syst_fraction_dict)
        for syst in syst_names:
            up_key, down_key = resolve_syst_keys(syst_fraction_dict, syst)
            if up_key in syst_fraction_dict:
                all_values.append(np.abs(syst_fraction_dict[up_key]))
            if down_key in syst_fraction_dict:
                all_values.append(np.abs(syst_fraction_dict[down_key]))


    # Now plot with fixed y-range
    for i in u._reported_pt_indices():
        result = u.normalized_results[i]
        raw_syst_fraction_dict = result.get('syst_fraction_dict', {})
        syst_fraction_dict = build_plot_fraction_dict(raw_syst_fraction_dict)
        #plt.figure(figsize=(12, 8))
        pt_bin = result['pt_bin']
        panel_values = []
        for syst in syst_names:
            up_key, down_key = resolve_syst_keys(syst_fraction_dict, syst)
            color_map = ['#e42536', '#5790fc', '#964a8b']
            color = color_map[syst_names.index(syst)] if syst in syst_names and syst_names.index(syst) < len(color_map) else None
            if up_key not in syst_fraction_dict and down_key not in syst_fraction_dict:
                print(
                    f"[DEBUG] Neither '{up_key}' nor '{down_key}' found in syst_fraction_dict "
                    f"for pt bin {pt_bin}. Available keys: {list(syst_fraction_dict.keys())}"
                )

            # Plot Up uncertainty (solid)
            label_dic = {'pu':'Pileup', 'l1prefiring': 'L1 Prefiring', 'q2': r'Q$^2$ Scale', 'pdf': 'PDF', 'herwig': 'Model Unc.', 'isr': 'ISR', 'fsr': 'FSR', 'jms': 'JMS', 'jmr': 'JMR'}
            if up_key in syst_fraction_dict:
                hep.histplot(syst_fraction_dict[up_key][1:], u.gen_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Up", color=color, ls='-')
                panel_values.append(np.asarray(syst_fraction_dict[up_key][1:], float))
            # Plot Down uncertainty (dashed)
            if down_key in syst_fraction_dict:
                hep.histplot(-syst_fraction_dict[down_key][1:], u.gen_edges_by_pt[i][1:], label=f"{label_dic.get(syst, syst)} Down", color=color, ls='--')
                panel_values.append(np.asarray(syst_fraction_dict[down_key][1:], float))


        # None of the requested systematics exist for this spec (e.g.
        # isr/fsr under model_envelope, where they live in the envelope
        # instead) -- don't save an empty frame.
        if not panel_values:
            plt.close()
            continue

        pt_bin_label = u._pt_bin_label(pt_bin)

        # if ylim is not None:
        #     plt.ylim(ylim)
        # Opaque legend frame (as in plot_bottom_line): with the tightened
        # symmetric y-range a flat curve can run through the legend text.
        plt.legend(title=rf"$p_T$  {pt_bin_label} GeV", fontsize = 15,
                   frameon=True, framealpha=0.9, facecolor='white',
                   edgecolor='0.7')
        hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(), com=u._com_label(), fontsize=20)
        # Symmetric y-range from the bins inside the shown window only --
        # the hidden buffer bins otherwise inflate the autoscale and
        # flatten the visible curves (same trap as the stat/model plots).
        edges = np.asarray(u.gen_edges_by_pt[i][1:], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        xlo, xhi = u._observable_xlim(i)
        visible = (centers >= xlo) & (centers <= xhi)
        if np.any(visible):
            vmax = max(float(np.abs(v[visible]).max()) for v in panel_values)
            if vmax > 0:
                plt.ylim(-1.3 * vmax, 1.3 * vmax)

        if u.groomed:
            plt.xlim(*u._observable_xlim(i))
            plt.xlabel(u._observable_label())
            save_path = f'./{u.spec.output_dir}uncertainties/{syst_names[0]}_groomed_{i-1}.pdf'
        else:
            plt.xlim(*u._observable_xlim(i))
            plt.xlabel(u._observable_label())
            save_path = f'./{u.spec.output_dir}uncertainties/{syst_names[0]}_ungroomed_{i-1}.pdf'
        u._finalize_plot(save_path=save_path, show=show)


def plot_herwig_systematic(u, show=True):
    flat_uncertainty = np.sqrt(np.diag(u.cov_data_herwig_np))/np.abs(u.y_unf_dict['herwigUp'])
    uncertainty_pt_binned = unflatten_gen_by_pt(flat_uncertainty, u.gen_edges_by_pt)
    unfolded_pt_binned = unflatten_gen_by_pt(u.y_unf, u.gen_edges_by_pt)
    
    for i, result in enumerate(u.normalized_results):
        syst_fraction_dict = result.get('syst_fraction_dict', {})
        # syst_fraction_dict is only populated for the reported pt bins
        # (first_reported_pt_bin..), so the unreported sink bin(s) carry no
        # 'herwigUp' -- skip them instead of raising.
        if 'herwigUp' not in syst_fraction_dict:
            continue
        error_in_syst = uncertainty_pt_binned[i]*syst_fraction_dict['herwigUp']  # Uncertainty on relative uncertainty
        pt_bin = result['pt_bin']
        if 'herwigUp' in syst_fraction_dict:
            hep.histplot(syst_fraction_dict['herwigUp'], u.gen_edges_by_pt[i], yerr = error_in_syst, label=f"Model Unc.", color='#964a8b', ls='-')


        # Fit a polynomial to the herwigUp systematic fraction
        if 'herwigUp' in syst_fraction_dict:
            edges = np.array(u.gen_edges_by_pt[i], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            #centers[0] = -100000000  # Set the first center to a very large negative value to exclude it from the fit
            y = syst_fraction_dict['herwigUp']
            mask = np.isfinite(y) & (y > 0)
            if mask.sum() > 3:
                degree = 2
                coeffs = np.polyfit(centers[mask], y[mask], degree, w=1.0/np.where(error_in_syst[mask] > 0, error_in_syst[mask], 1e-10))
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(centers[mask][1], centers[mask][-1], 200)
                plt.plot(x_fit, poly(x_fit), color='#5790fc', ls='--', lw=2, label=f"Poly fit (deg {degree})")

        pt_bin_label = u._pt_bin_label(pt_bin)
        plt.legend(title=rf"$p_T$  {pt_bin_label} GeV")
        hep.cms.label(u.cms_label, data=True, lumi = 138, com = 13, fontsize = 20)
        plt.ylim(0,0.5)
        plt.xlim(*u._observable_xlim(i))
        plt.xlabel(u._observable_label())
        plt.ylabel("Relative Uncertainty")
        save_path = f'./{u.spec.output_dir}uncertainties/herwig_groomed_{i-1}.pdf' if u.groomed else f'./{u.spec.output_dir}uncertainties/herwig_ungroomed_{i-1}.pdf'
        u._finalize_plot(save_path=save_path, show=show)


def plot_purity_stability_herwig(u, show=True):
    """Overlay Pythia8 vs Herwig7 purity & stability to diagnose generator dependence."""
    hep.style.use("CMS")
    suffix = "groomed" if u.groomed else "ungroomed"
    len_underflow = len(u.gen_edges_by_pt[0]) - 1

    def _purity_stability(mosaic):
        diagonal = np.diag(mosaic)
        purity_denom = mosaic[len_underflow:, :].sum(axis=0)
        stability_denom = mosaic[:, len_underflow:].sum(axis=1)
        purity = np.divide(diagonal, purity_denom,
                           out=np.zeros_like(diagonal, dtype=float), where=purity_denom != 0)
        stability = np.divide(diagonal, stability_denom,
                              out=np.zeros_like(diagonal, dtype=float), where=stability_denom != 0)
        return (
            unflatten_gen_by_pt(purity, u.gen_edges_by_pt),
            unflatten_gen_by_pt(stability, u.gen_edges_by_pt),
        )

    purity_py, stability_py = _purity_stability(u.mosaic_gen)
    purity_hw, stability_hw = _purity_stability(u.mosaic_gen_herwig)

    title_list = [
        "",
        r"200 $<$ $p_T$ $<$ 290 GeV",
        r"290 $<$ $p_T$ $<$ 400 GeV",
        r"400 $<$ $p_T$ $< \, \infty$ GeV",
    ]

    for i in range(len(u.pt_edges) - 1):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        fig.subplots_adjust(wspace=0.05)

        for ax, (pur_py, stab_py, pur_hw, stab_hw, metric) in zip(
            axes,
            [
                (purity_py[i], None, purity_hw[i], None, "Purity"),
                (None, stability_py[i], None, stability_hw[i], "Stability"),
            ],
        ):
            vals_py = pur_py if metric == "Purity" else stab_py
            vals_hw = pur_hw if metric == "Purity" else stab_hw
            edges = u.gen_edges_by_pt[i]
            hep.histplot(vals_py, edges, label="Pythia8", ax=ax, color="steelblue")
            hep.histplot(vals_hw, edges, label="Herwig7", ax=ax, color="darkorange", linestyle="--")
            ax.axhline(0.5, color="k", linestyle=":", linewidth=1.2, label="0.5 threshold")
            ax.set_xlabel(u._observable_short_label(), fontsize=14)
            ax.set_xlim(*u._observable_xlim(i))
            ax.set_ylim(0.0, 1.05)
            ax.set_title(metric, fontsize=14)
            ax.legend(title=title_list[i], fontsize=12)

        axes[0].set_ylabel("Purity / Stability", fontsize=14)
        hep.cms.label(u.cms_label, data=False, lumi=138, com=13, fontsize=18, ax=axes[0])

        save_path = f"./{u.spec.output_dir}unfold/purity_stability_herwig_{suffix}_{i-1}.pdf"
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_lcurve(u, show=True):
    """L-curve of the nominal-data tau scan (regularized runs only)."""
    scan = getattr(u, "lcurve_scan", None)
    if scan is None:
        return
    plt.figure(figsize=(8, 7))
    plt.plot(scan["x"], scan["y"], "-o", ms=3, lw=1.5, color="#5790fc",
             label="L-curve scan")
    plt.plot(
        [scan["best_x"]], [scan["best_y"]], "*", ms=18, color="#e42536",
        label=rf"chosen: $\tau$ = {scan['tau']:.3g}",
    )
    plt.xlabel(r"$\log_{10}\,\chi^2_{\rm data}$")
    plt.ylabel(r"$\log_{10}\,(Lx)^2$")
    suffix = "groomed" if u.groomed else "ungroomed"
    plt.legend(title=f"{u.regularization}, {suffix}")
    save_path = f"./{u.spec.output_dir}unfold/lcurve_{suffix}.pdf"
    u._finalize_plot(save_path=save_path, show=show)


def plot_correlation(u, show=True, shown_only=False, covariance="stat"):
    """Plot the normalized-result correlation matrix.

    ``covariance="stat"`` is the historical default.  Prepared pair-split
    result plots can opt into ``"total"`` to include the rank-one
    systematic covariance contributions in the displayed correlation.
    """
    # ``shown_only`` crops each pT slice to the reported rho window (the
    # per-pT shown floors) before forming the correlation, matching the
    # approval-talk figure; the full-space version stays the default.
    cov_matrix = u._correlation_covariance(covariance)
    # The explicit first reported slice controls the correlation view.  Some
    # legacy inputs have a 185--200 GeV sink at index zero; pair-split
    # inputs instead begin their physical measurement at that index.
    first_pt_bin = getattr(u, "first_reported_pt_bin", 0)
    gen_offset = sum(
        len(edges) - 1 for edges in u.gen_edges_by_pt[:first_pt_bin]
    )
    cov_matrix = cov_matrix[gen_offset:, gen_offset:]
    reported_gen_edges = u.gen_edges_by_pt[first_pt_bin:]
    if shown_only:
        keep = []
        ncols_by_gp = []
        offset = 0
        for i, edges in enumerate(reported_gen_edges, start=first_pt_bin):
            edges = np.asarray(edges, float)
            local = np.flatnonzero(u._shown_gen_mask(i))
            keep.extend((offset + local).tolist())
            ncols_by_gp.append(len(local))
            offset += len(edges) - 1
        cov_matrix = cov_matrix[np.ix_(keep, keep)]
    else:
        ncols_by_gp = [len(e) - 1 for e in reported_gen_edges]
    std_devs = np.sqrt(np.diag(cov_matrix))

    # Avoid division by zero by replacing zeros with a small number
    std_devs[std_devs == 0] = 1e-10

    # Compute the outer product of standard deviations
    std_matrix = np.outer(std_devs, std_devs)

    # Compute correlation matrix
    corr_matrix = cov_matrix / std_matrix

    u.corr_matrix = corr_matrix
    ## colormap
    num_bins = 20
    bounds = np.linspace(-1, 1, num_bins + 1)  # 21 boundaries for 20 bins

    # Get the 'seismic' colormap
    base_cmap = plt.get_cmap("seismic", num_bins)
    colors = base_cmap(np.linspace(0, 1, num_bins))  # Extract colors
    
    # Force the bins for -0.1 to 0.1 to be white
    for i in range(len(bounds) - 1):
        if -0.1 <= bounds[i] <= 0.1:
            colors[i] = [1, 1, 1, 1]  # Set to white (RGBA)


        
    #import matplotlib.colors as mcolors
    # Create colormap and normalizer
    cmap =  mcolors.ListedColormap(colors)  # Discrete 'seismic' colormap
    norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalize for discrete bins
    

    fig, ax = plt.subplots(layout="constrained")
    img = ax.imshow(corr_matrix, cmap=cmap, norm=norm, origin='lower')

    # ---- Add grid lines and labels for pt bins ----
    # Block structure per reported pT bin (shown-cropped when shown_only)
    x_bounds = np.r_[0, np.cumsum(ncols_by_gp)]
    # Draw dashed lines at pt bin boundaries
    for x in x_bounds[1:-1]:
        ax.axvline(x-0.5, color="r", ls="--", lw=2, alpha=0.6)
        ax.axhline(x-0.5, color="r", ls="--", lw=2, alpha=0.6)
    # Optional: thin grid inside each block at every integer cell
    ax.set_xticks(np.arange(-0.5, corr_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, corr_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", alpha=0.15, lw=0.5)
    # Tick labels at block centers
    x_centers = (x_bounds[:-1] + x_bounds[1:] - 1) / 2.0
    pt_edges = u.pt_edges[first_pt_bin:]
    x_labels = [f"{int(pt_edges[i])}–{int(pt_edges[i+1]) if i+1 < len(pt_edges)-1 else '∞'}" for i in range(len(pt_edges)-1)]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(x_centers)
    ax.set_yticklabels(x_labels, rotation=90, va="center")

    # The aspect-equal matrix plus its colorbar leaves a much narrower axes
    # than a normal 10x10 panel, so the full PUB_* sizes do not fit here:
    # the pT-range tick strings run into each other and the 21 colorbar
    # boundaries overprint. Keep the axis labels at PUB and step the rest
    # down to what the axes can actually hold.
    ax.set_xlabel(r"GEN $p_T$ (GeV)", fontsize=PUB_LABEL_FONTSIZE)
    ax.set_ylabel(r"GEN $p_T$ (GeV)", fontsize=PUB_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=PUB_TICK_FONTSIZE - 8)

    cbar = fig.colorbar(img, ax=ax, ticks=bounds[::2], boundaries=bounds,
                        fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (Groomed)" if u.groomed else "Correlation (Ungroomed)",
                   fontsize=PUB_LABEL_FONTSIZE - 8)
    cbar.ax.tick_params(labelsize=PUB_TICK_FONTSIZE - 12)
    # The colorbar renarrows the aspect-equal axes under constrained
    # layout, which would squeeze the frozen CMS-label geometry; settle
    # and freeze the layout, then stamp the label on the final axes
    # (same fix as plot_correlation_before_after).
    fig.canvas.draw()
    fig.set_layout_engine("none")
    # 20pt ran "CMS Internal" straight into the lumi text on this narrow axes.
    hep.cms.label(u.cms_label, data=True, lumi=u._lumi_label(),
                  com=u._com_label(), ax=ax, fontsize=15)
    mode = "groomed" if u.groomed else "ungroomed"
    tag = "_shown" if shown_only else ""
    if covariance == "total":
        tag += "_total"
    save_path = f'{u.spec.output_dir}unfold/correlation_{mode}{tag}.pdf'
    u._finalize_plot(save_path=save_path, show=show)


def plot_response_matrix(u, probability=True, log=False, show=True):
    (
        reported_mosaic,
        reported_reco_edges,
        reported_gen_edges,
        reported_pt_edges,
    ) = u._reported_matrix_view(u.mosaic)
    # Display "cheat" (ARC round-2): show the response matrix only over the
    # reported window (rho >= the display floor), hiding the low-rho buffer
    # bins that the unfold uses but that we never plot.
    floor = (u.spec.xlim_lower_groomed if u.groomed
             else u.spec.xlim_lower_ungroomed)
    reported_mosaic, reported_reco_edges, reported_gen_edges = (
        u._crop_matrix_view_to_floor(
            reported_mosaic, reported_reco_edges, reported_gen_edges, floor)
    )
    fig, ax = _plot_response_mosaic_cms(u, 
        reported_mosaic,
        reco_mass_edges_by_pt=reported_reco_edges,
        gen_mass_edges_by_pt=reported_gen_edges,
        reco_pt_edges=reported_pt_edges,
        gen_pt_edges=reported_pt_edges,
        probability = probability,
        mask_zeros=True,
        log=log,                              # set False for linear
        rlabel="Groomed" if u.groomed else "Ungroomed",
    )
    u._finalize_plot(show=show, fig=fig)


def plot_uncertainty_heatmap(u, show=True):
    """
    2D heatmap of fractional uncertainties: rows = systematic groups,
    columns = mass bins, color = uncertainty magnitude in %.
    Gives an at-a-glance budget matrix showing which source dominates
    in which part of the spectrum. One figure per pT bin.
    """
    group_order = [
        "Jet Energy",
        "Jet Mass",
        "Parton Shower",
        "Lepton SFs",
        "Other Theory",
        "Model Uncertainty",
        "Stat Unc",
        "Total",
    ]

    for i in u._reported_pt_indices():
        result = u.normalized_results[i]
        syst_fraction_dict = u._build_syst_fraction_dict(i)
        grouped_dict = u._group_syst_fraction_dict(syst_fraction_dict, grouped=True)

        # Compute per-group envelope: max(Up, Down) for each group
        envelopes = {}
        seen_bases = set()
        for key in grouped_dict:
            if key in {"Stat Unc", "Total_Up", "Total_Down"}:
                continue
            base, variation = u._split_systematic_variation(key)
            if base not in seen_bases:
                seen_bases.add(base)
                up = grouped_dict.get(f"{base}Up", np.zeros_like(grouped_dict[key]))
                down = grouped_dict.get(f"{base}Down", np.zeros_like(grouped_dict[key]))
                envelopes[base] = np.maximum(up, down)

        envelopes["Stat Unc"] = grouped_dict["Stat Unc"]
        envelopes["Total"] = np.maximum(
            grouped_dict.get("Total_Up", np.zeros(1)),
            grouped_dict.get("Total_Down", np.zeros(1)),
        )

        # Build matrix in the predefined group order, skipping absent groups
        rows = [g for g in group_order if g in envelopes]
        matrix = np.array([envelopes[row] for row in rows]) * 100.0  # → percent

        edges = np.array(u.gen_edges_by_pt[i], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Trim columns below the observable lower limit:
        # ungroomed starts at 20 GeV (20-30 bin), groomed at 10 GeV (10-20 bin)
        x_lo, _ = u._observable_xlim(i)
        start_col = int(np.searchsorted(edges[:-1], x_lo, side='left'))
        centers = centers[start_col:]
        matrix = matrix[:, start_col:]

        n_rows, n_cols = matrix.shape

        fig_w = max(10, n_cols * 0.9 + 3)
        fig_h = n_rows * 0.8 + 2.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        vmax = 50.0

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                       vmin=0.0, vmax=vmax, origin='upper')

        # Annotate every cell with its numeric value
        for r in range(n_rows):
            for c in range(n_cols):
                val = matrix[r, c]
                text_color = 'white' if val > vmax * 0.65 else 'black'
                ax.text(c, r, f"{val:.1f}", ha='center', va='center',
                        fontsize=13, color=text_color, fontweight='bold')

        # This is a wide, densely annotated budget matrix rather than a
        # half-textwidth panel, so the cell/tick type is boosted well above
        # the old 8-9pt but kept below the full PUB_* sizes, which would
        # collide at this column density.
        # One decimal: log10(rho^2) bin centres are ~0.5 apart, so "%.0f"
        # collapsed distinct columns onto the same label ("-2", "-2").
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([f"{c:.1f}" for c in centers],
                           rotation=45, ha='right', fontsize=16)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(rows, fontsize=18)

        # Visual separators: dashed blue before stat unc, solid black before total
        if "Stat Unc" in rows:
            ax.axhline(rows.index("Stat Unc") - 0.5, color='steelblue', lw=1.5, ls='--')
        if "Total" in rows:
            ax.axhline(rows.index("Total") - 0.5, color='black', lw=2.0)

        # Aligned colorbar: make_axes_locatable ensures it matches the heatmap height exactly
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Uncertainty (%)", fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        pt_bin = result['pt_bin']
        if pt_bin[1] == float('inf') or pt_bin[1] > 100000:
            pt_bin_label = f"{int(pt_bin[0])}–∞ GeV"
        else:
            pt_bin_label = f"{int(pt_bin[0])}–{int(pt_bin[1])} GeV"

        ax.set_xlabel(u._observable_label(), fontsize=PUB_LABEL_FONTSIZE)

        # mplhep freezes the CMS<->lumi gap as an axes fraction when the
        # label is drawn, so with tight_layout the label must be stamped
        # AFTER the axes have been resized (skill rule 9). The pT bin moves
        # into rlabel now that the title is gone.
        plt.tight_layout()
        hep.cms.label(u._cms_extra_label(), data=True, ax=ax, fontsize=18,
                      rlabel=rf"$p_T$: {pt_bin_label}, "
                             rf"{u._lumi_label()} fb$^{{-1}}$ "
                             rf"({u._com_label()} TeV)")
        suffix = "groomed" if u.groomed else "ungroomed"
        save_path = f"./{u.spec.output_dir}uncertainties/heatmap_{suffix}_{u._output_panel_index(i)}.pdf"
        u._finalize_plot(save_path=save_path, show=show, fig=fig)


def plot_model_envelope(u, show=False):
    """Per-pT model-uncertainty composition: shower / CR / frag re-unfold
    shifts, the FSR PSWeight shift, and their per-bin envelope (the term
    added to the total band when ``spec.model_envelope`` is on)."""
    if not hasattr(u, "model_shift_components"):
        return
    groomed_tag = "groomed" if u.groomed else "ungroomed"
    colors = {"Vincia": "#e42536", "CR": "#5790fc",
              "frag": "#f89c20", "FSR": "#7a21dd"}
    vincia_label = (
        "Vincia"
        if getattr(u.spec, "model_envelope_source", "zjet_offline")
        == "prepared_systematics"
        else "Vincia"
    )
    npz_payload = {}
    for i in u._reported_pt_indices():
        result = u.normalized_results[i]
        edges = np.asarray(u.gen_edges_by_pt[i], float)
        fig, ax = plt.subplots(layout="constrained")
        for name in ("Vincia", "CR", "frag"):
            frac = np.asarray(u.model_shift_components[name][i], float)
            label = vincia_label if name == "Vincia" else name
            ax.stairs(frac, edges, color=colors[name], lw=2, label=label)
            npz_payload[f"{name}_{i}"] = frac
        fsr = np.asarray(u.model_fsr_frac[i], float)
        ax.stairs(fsr, edges, color=colors["FSR"], lw=2, label="FSR")
        npz_payload[f"FSR_{i}"] = fsr
        envelope = np.asarray(result["model_unc_frac"], float)
        ax.stairs(envelope, edges, color="black", lw=3, label="Total model")
        npz_payload[f"envelope_{i}"] = envelope
        npz_payload[f"edges_{i}"] = edges
        ax.set_xlim(*u._observable_xlim(i))
        # Scale to the shown bins only: the buffer bins hidden outside the
        # xlim window carry large fractions that would otherwise squash
        # the visible curves (groomed panels blew up to 0.9-1.6).
        xlo, xhi = u._observable_xlim(i)
        centers = 0.5 * (edges[:-1] + edges[1:])
        visible = (centers >= xlo) & (centers <= xhi)
        curves = [
            np.asarray(u.model_shift_components[name][i], float)
            for name in ("Vincia", "CR", "frag")
        ] + [fsr, envelope]
        vis_max = (
            max(float(c[visible].max()) for c in curves)
            if np.any(visible) else 0.0
        )
        ax.set_ylim(0, max(0.1, 1.3 * vis_max))
        ax.set_xlabel(u._observable_label())
        ax.set_ylabel("Model uncertainty fraction")
        pt_low, pt_high = result["pt_bin"]
        pt_high_label = (
            "∞" if not np.isfinite(pt_high) or pt_high > 100000
            else u._as_int_when_whole(pt_high)
        )
        ax.legend(
            title=rf"$p_T$: {u._as_int_when_whole(pt_low)}–{pt_high_label} GeV",
            fontsize=15,
        )
        hep.cms.label(
            u.cms_label, data=True, lumi=u._lumi_label(),
            com=u._com_label(), fontsize=20, ax=ax,
        )
        output_index = u._output_panel_index(i)
        save_path = (
            f"./{u.spec.output_dir}unfold/"
            f"model_envelope_{groomed_tag}_{output_index}.pdf"
        )
        u._finalize_plot(save_path=save_path, show=show, fig=fig)
    npz_path = Path(f"./{u.spec.output_dir}unfold")
    npz_path.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path / f"model_envelope_{groomed_tag}.npz", **npz_payload)


def run_all_plots(u, show=False):
    # Initialize the shared style before the first figure. Otherwise the
    # first mode can use Matplotlib defaults until a later plot sets CMS.
    hep.style.use("CMS")
    plot_unfolded_fancy(u, show=show)
    plot_unfolded_summary_linear(u, show=show)
    plot_statistical_fraction(u, show=show)
    plot_systematic_fraction(u, show=show)
    plot_systematic_fraction_grouped(u, show=show)
    plot_systematic_fraction_grouped(u, show=show, log=False)
    if getattr(u.spec, "model_envelope", False):
        plot_model_envelope(u, show=show)
    if u.has_herwig and "herwigUp" in u.y_unf_dict:
        plot_herwig_systematic(u, show=show)
    q2_group = (
        ["q2"]
        if any(name in u.systematics for name in ("q2Up", "q2Down"))
        else ["q2muR", "q2muF"]
    )
    # The lepton entries are aggregates built inside
    # plot_systematic_frac_indiv from the per-leg systematics, whose raw
    # names (elereco/eleid/...) don't start with the aggregate name -- map
    # them to the prefixes that actually appear in u.systematics.
    group_prefix_aliases = {
        "ElectronSF": ("elereco", "eleid", "eletrig"),
        "MuonSF": ("mureco", "muid", "mutrig", "muiso"),
    }
    for systematic_group in (
        ["JES", "JER"],
        ["JMS", "JMR"],
        [*q2_group, "pdf"],
        ["pu", "l1prefiring", "lumi"],
        ["ElectronSF", "MuonSF"],
        ["isr", "fsr"],
    ):
        available_group = [
            name for name in systematic_group
            if u._has_systematic(*group_prefix_aliases.get(name, (name,)))
        ]
        if available_group:
            plot_systematic_frac_indiv(u, available_group, show=show)
    if u.has_herwig and u._has_systematic("herwig"):
        plot_systematic_frac_indiv(u, ["herwig"], show=show)
    plot_correlation(u, show=show)
    if getattr(u.spec, "normalize_over_shown", False):
        # Reported-region companion (approval-talk convention): each pT
        # slice cropped to its shown rho window.
        plot_correlation(u, show=show, shown_only=True)
    plot_lcurve(u, show=show)
    u.save_normalized_covariance()
    u.save_2d_unfolded()
    u.save_2d_uncertainty_summary()
    plot_unfolded_unrolled_2d(u, show=show)
    plot_jackknife_convergence(u, show=show)
    plot_uncertainty_heatmap(u, show=show)
    plot_unfolded(u, show=show)
    if u.has_herwig:
        plot_herwig_bias_test(u, show=show)
    plot_response_matrix(u, probability=True, show=show)
    plot_folded(u, show=show)
    plot_bottom_line(u, show=show)
    # ARC round-2 (WS3.3): detector level rebinned to the gen binning, so
    # K is square and chi2_smeared is directly comparable to chi2_unfold.
    # These are the panels shown in the AN: no chi2 stamp, the numbers come
    # from the chi2 summary bar charts below.
    plot_bottom_line(u, show=show, rebin_reco_to_gen=True,
                          annotate_chi2=False)
    # PRIMARY (ARC round-2): chi2/ndof over the SHOWN space only -- each
    # slice cut at the rho value it is displayed to, with one global number.
    plot_bottom_line_chi2_summary(u, show=show, normalized=True, min_edge="shown")
    plot_bottom_line_chi2_summary(u, show=show, min_edge="shown")
    # Reference: the full-range and raw-chi2 versions.
    plot_bottom_line_chi2_summary(u, show=show)
    plot_bottom_line_chi2_summary(u, show=show, normalized=True)
    plot_fakes_misses(u, show=show)
    plot_purity_stability(u, show=show)
    if u.has_validation_inputs:
        plot_input_data_mc(u, show=show)


def _plot_response_mosaic_cms(
    u,
    mosaic,
    reco_mass_edges_by_pt,   # list: per reco-pT slice, mass edges used (rows per block)
    gen_mass_edges_by_pt,    # list: per gen-pT slice,  mass edges used (cols per block)
    reco_pt_edges,           # e.g. [200, 290, 400, 13000]
    gen_pt_edges,            # e.g. [200, 290, 400, 13000]
    *,
    mask_zeros=True,
    probability=True,  # if True, normalize each block to sum to 1
    log=False,
    cmap="viridis",
    rlabel=None,             # e.g. "Ungroomed"
    show_cond=False,         # append the condition number to rlabel (ARC r2: leaning "don't show")
    vmin=None, vmax=None,
    ax=None
):
    """
    Draw a CMS-style gridded 'flattened' response plot.
    - `mosaic` is your unpadded 2D array (blocks concatenated).
    - Mass bin counts per (reco pT, gen pT) block come from the edge lists.
    - Dashed grid lines drawn at pT-block boundaries.
    """
    if probability:
        # Normalize each column to sum to 1
        mosaic = mosaic / np.sum(mosaic, axis=0, keepdims=True)
        # ensure no NaNs
        mosaic = np.nan_to_num(mosaic, nan=0.0)
    
    # Compute all the singular values of the mosaic matrix
    # and condition number (ratio of largest to smallest singular value)
    # to check for numerical stability.

    singular_values = np.linalg.svd(mosaic, compute_uv=False)
    #print("Singular values of the response mosaic:", singular_values)

    hep.style.use("CMS")
    if ax is None:
        # Keep the CMS-style ~10x10 per-axes footprint (a larger canvas
        # shrinks every font relative to it in the scaled-down PDF; ARC
        # round-2 "make all labels much bigger") -- the extra width pays
        # for the colorbar so the matrix axes stay square. Constrained
        # layout keeps the CMS-label spacing and the colorbar height.
        fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    else:
        fig = ax.figure

    img = mosaic.copy()
    if mask_zeros:
        img = np.ma.masked_where(img == 0, img)

    # Choose norm
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else None

    # ---- Physical bin coordinates: stack the per-pT-block rho edges so
    # every cell is drawn with its true log10(rho^2) width. The matching
    # (i,i) block diagonal is then the real y=x line -- with index-space
    # imshow the unequal rho bin widths made it a fake diagonal.
    def _stacked_edges(edges_by_pt):
        coords = [0.0]
        bounds = [0.0]
        for edges in edges_by_pt:
            e = np.asarray(edges, dtype=float)
            coords.extend(bounds[-1] + (e[1:] - e[0]))
            bounds.append(bounds[-1] + (e[-1] - e[0]))
        return np.asarray(coords), np.asarray(bounds)

    x_coords, x_bounds = _stacked_edges(gen_mass_edges_by_pt)
    y_coords, y_bounds = _stacked_edges(reco_mass_edges_by_pt)

    # rasterized=True: the mesh is embedded as one high-res raster image
    # (see the dpi in savefig below) instead of per-cell vector rectangles,
    # whose shared edges every PDF rasterizer renders as hairline seams.
    im = ax.pcolormesh(x_coords, y_coords, img, cmap=cmap, norm=norm,
                       vmin=None if log else vmin, vmax=None if log else vmax,
                       antialiased=False, rasterized=True)
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(y_coords[0], y_coords[-1])

    # Dashed lines at pT-block boundaries
    for y in y_bounds[1:-1]:
        ax.axhline(y, color="r", ls="--", lw=2, alpha=0.6)
    for x in x_bounds[1:-1]:
        ax.axvline(x, color="r", ls="--", lw=2, alpha=0.6)

    # ---- Tick labels at pT bin centers ----
    x_centers = (x_bounds[:-1] + x_bounds[1:]) / 2.0
    y_centers = (y_bounds[:-1] + y_bounds[1:]) / 2.0
    # Label with pT edges (lower edges are fine; pick what you prefer)
    x_labels = [f"{int(gen_pt_edges[i])}–{int(gen_pt_edges[i+1]) if i+1 < len(gen_pt_edges)-1 else '∞'}" for i in range(len(gen_pt_edges)-1)]
    y_labels = [f"{int(reco_pt_edges[i])}–{int(reco_pt_edges[i+1]) if i+1 < len(reco_pt_edges)-1 else '∞'}" for i in range(len(reco_pt_edges)-1)]

    # This canvas is 12 in wide against the 10 in of the other published
    # figures, so it scales down further at the same \textwidth: give its
    # type the matching 1.2x so on-page sizes agree.
    mosaic_label_fs = round(PUB_LABEL_FONTSIZE * 1.2)
    mosaic_tick_fs = round(PUB_TICK_FONTSIZE * 1.2)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(y_labels, rotation=90, va="center")
    ax.tick_params(which="both", top=False, right=False,
                   labelsize=mosaic_tick_fs)

    ax.set_xlabel(r"Gen. $\log_{10}(\rho^{2}) \otimes p_{\mathrm{T}}$ (GeV)",
                  fontsize=mosaic_label_fs)
    ax.set_ylabel(r"Reco. $\log_{10}(\rho^{2}) \otimes p_{\mathrm{T}}$ (GeV)",
                  fontsize=mosaic_label_fs)

    # True y=x diagonal within the matching (i,i) blocks (gen and reco
    # cover the same rho range per block, so corner-to-corner is exact).
    for i in range(min(len(x_bounds)-1, len(y_bounds)-1)):
        ax.plot([x_bounds[i], x_bounds[i+1]], [y_bounds[i], y_bounds[i+1]],
                color="r", lw=1, alpha=0.7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Migration probability" if probability else "Counts",
                   fontsize=mosaic_label_fs)
    cbar.ax.tick_params(labelsize=mosaic_tick_fs)

    # CMS label
    rparts = [rlabel] if rlabel else []
    if show_cond:
        rparts.append(f"Cond. = {np.linalg.cond(mosaic):.2f}")
    hep.cms.label(u.cms_label, data=False, rlabel=", ".join(rparts))

    groomed_tag = "groomed" if u.groomed else "ungroomed"
    resp_path = u._relocate_output(f"response_{groomed_tag}.pdf")
    resp_path.parent.mkdir(parents=True, exist_ok=True)
    u._save_label_flavors(fig, resp_path, dpi=300)
    return fig, ax

