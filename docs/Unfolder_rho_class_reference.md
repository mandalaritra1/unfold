# `Unfolder` class reference for `Unfolder_rho.py`

Source: `src/unfold/tools/Unfolder_rho.py`

This document explains every method defined inside the `Unfolder` class and the main function calls each method makes. The class does four main jobs:

1. Build binning and response-matrix inputs.
2. Run TUnfold for nominal and systematic variations.
3. Derive statistical and systematic uncertainties.
4. Produce validation and publication-style plots.

## High-level call flow

When an `Unfolder` object is created, `__init__` runs this sequence:

1. `_setup_binning()`
2. `_make_inputs_numpy()`
3. `_configure_systematics()`
4. `_load_data()`
5. `_perform_unfold()` for nominal
6. `_perform_unfold()` for each requested systematic
7. `_compute_stat_unc()`
8. `_normalize_result()`
9. `_compute_total_systematic()`

## Constructor and setup helpers

| Method | What it does | Main calls / effects |
| --- | --- | --- |
| `__init__(groomed, closure=False, herwig_closure=False, do_syst=False, cms_label="Internal")` | Main entry point. Stores configuration, prepares inputs, runs unfolding, and computes uncertainties. | Calls `_setup_binning`, `_make_inputs_numpy`, `_configure_systematics`, `_load_data`, `_perform_unfold`, `_compute_stat_unc`, `_normalize_result`, `_compute_total_systematic`. Fills most analysis attributes. |
| `_setup_binning()` | Loads the rho and pT bin definitions for groomed or ungroomed mode. | Calls `binning.bin_edges(self.groomed)`. Stores `self.bins`, `self.edges`, `self.edges_gen`, `self.pt_edges`. |
| `_configure_systematics(do_syst)` | Chooses which systematics to run. | Reads `self.sys_matrix_dic.keys()`. Uses all available variations if `do_syst=True`, otherwise only `["nominal"]`. |
| `_load_pickle(filename)` | Small utility to read a pickle file. | Calls `open(..., "rb")` and `pkl.load`. |
| `_finalize_plot(save_path=None, show=True, fig=None)` | Standard plot finalizer used by all plotting methods. Saves the figure, then either shows it or closes it. | Calls `Path(...).parent.mkdir`, `fig.savefig`/`plt.gcf().savefig`, `plt.show`, `plt.close`. |
| `_cms_extra_label()` | Formats the extra CMS text label. | Returns a label string with a leading space when needed. |
| `_histogram_keys()` | Selects histogram names based on groomed vs ungroomed mode. | Returns the response, reco, and gen keys used to read pickle contents. |

## Input preparation helpers

| Method | What it does | Main calls / effects |
| --- | --- | --- |
| `_prepare_jackknife_inputs(data2d_jk, mass_edges_reco, pt_edges, reco_mass_edges_by_pt)` | Converts jackknife reco histograms into flattened reco vectors. | Calls histogram `project`, `.values()`, `reorder_to_expected_2d`, `merge_mass_flat`. Returns a list of 10 jackknife vectors. |
| `_prepare_truth_spectrum(gen2d, mass_edges_gen, pt_edges, gen_mass_edges_by_pt)` | Builds the flattened truth spectrum used for comparisons. | Calls histogram `project`, `.values()`, `reorder_to_expected_2d`, `merge_mass_flat`. |
| `_prepare_jackknife_response_matrices(...)` | Builds jackknife response mosaics from the jackknife 4D response histogram. | Calls histogram slicing, `project`, `.values(flow=False)`, `reorder_to_expected`, `mosaic_no_padding`. Returns 10 response matrices. |
| `_prepare_nominal_inputs(...)` | Prepares nominal reco, fake, miss, and response objects. | Calls histogram `project`, `.values()`, `reorder_to_expected_2d`, `reorder_to_expected`, `mosaic_no_padding`, `merge_mass_flat`. Stores `self.h2d`, `self.h2d_fakes`, `self.h2d_misses`, `self.M_np_2d_gen`, `self.mosaic_gen`, `self.fakes_2d`, `self.misses_2d`. |
| `_prepare_herwig_inputs(...)` | Prepares Herwig reco, fake, and miss inputs used by model systematics or closure studies. | Calls histogram slicing, `project`, `.values(flow=False)`, `reorder_to_expected_2d`, `merge_mass_flat`. Stores Herwig-specific flattened arrays. |
| `_response_matrix_for_systematic(syst, resp_matrix_4d_herwig, sys_matrix_dic)` | Returns the response matrix for a given systematic. | For Herwig variations it also accesses histogram variances; otherwise it pulls the numpy matrix directly from `sys_matrix_dic`. |
| `_compute_fake_fraction(fakes_flat, matched_flat)` | Computes reco-bin fake fractions with safe divide behavior. | Calls `np.divide` inside `np.errstate`, then `np.clip`. |
| `_finalize_reco_views(mass_edges_reco, reco_mass_edges_by_pt)` | Finalizes nominal reco-side arrays after all systematics are loaded. | Calls `merge_mass_flat` and `_compute_fake_fraction`. Stores `self.M_np_2d`, `self.mosaic`, `self.mosaic_2d`, `self.fake_fraction_2d`, and Herwig equivalents when available. |

## Data loading and unfolding core

| Method | What it does | Main calls / effects |
| --- | --- | --- |
| `_load_data(...)` | Central input loader. Merges eras, loads data/MC/Herwig/jackknife pickles, builds fake and miss histograms, and prepares all response mosaics. | Calls `_merge_eras`, `_merge_eras_jk`, `_load_pickle`, `_histogram_keys`, `rebin_hist`, `_prepare_truth_spectrum`, `_prepare_jackknife_inputs`, `_prepare_jackknife_response_matrices`, `_prepare_nominal_inputs`, `_prepare_herwig_inputs`, `_response_matrix_for_systematic`, `reorder_to_expected`, `mosaic_no_padding`, `_finalize_reco_views`. |
| `plot_fakes_misses(show=True)` | Plots fake rate and efficiency versus rho for each pT bin. | Uses `unflatten_gen_by_pt`, `plt.stairs`, `hep.cms.label`, `_finalize_plot`. |
| `_compute_stat_unc()` | Computes jackknife statistical uncertainties from input fluctuations and response-matrix fluctuations. | Repeatedly calls `_perform_unfold` with `do_jk=True`, then uses `np.std`, `np.divide`, `np.sqrt`, `unflatten_gen_by_pt`. Stores fractional and pT-binned statistical uncertainties. |
| `_select_measured_spectrum(closure, herwig_closure, meas_flat)` | Chooses which reco spectrum should be unfolded. | Returns user-provided `meas_flat`, nominal reco data, closure reco MC, or Herwig reco input depending on flags. |
| `_apply_fake_correction(meas_flat, systematic, closure, herwig_closure)` | Applies fake subtraction unless running a closure mode. | Uses nominal or Herwig fake fractions and calls `np.asarray`, `np.clip`. |
| `_build_root_binning()` | Builds ROOT `TUnfoldBinning` trees for truth and reco axes. | Calls `ROOT.TUnfoldBinning`, `AddBinning`, `AddAxis`, and uses `array("d", edges)`. |
| `_fill_root_histogram(hist, values)` | Fills a 1D ROOT histogram from a flat numpy vector. | Calls `hist.SetBinContent`. |
| `_fill_response_histogram(h_resp, resp_np, misses)` | Fills the migration matrix and miss row in the ROOT response histogram. | Calls `h_resp.SetBinContent` for reco/true bin pairs and miss entries. |
| `_store_covariances(unfold, systematic)` | Extracts covariance matrices from TUnfold outputs. | Calls `GetEmatrixTotal`, `GetEmatrixSysUncorr`, `GetEmatrixInput`, `GetBinContent`. Stores numpy covariance arrays for nominal and Herwig cases. |
| `_store_unfold_result(systematic, do_jk, jk_target, unfold, h_meas, h_true)` | Converts unfolded ROOT histograms to numpy arrays and stores the results. | Calls `unfold.GetOutput`, `unfold.GetFoldedOutput`, `_th1_to_arrays`, `unfold.GetL`. Appends jackknife outputs or stores nominal/systematic results. |
| `_perform_unfold(systematic="nominal", closure=False, herwig_closure=False, meas_flat=None, do_jk=False, resp_np=None, jk_target="input")` | Core unfolding routine. Creates ROOT histograms, runs `TUnfoldDensity`, and stores outputs. | Calls `_select_measured_spectrum`, `_apply_fake_correction`, `_build_root_binning`, `ROOT.TUnfoldBinning.CreateHistogramOfMigrations`, `_fill_response_histogram`, `_fill_root_histogram`, `ROOT.TUnfoldDensity`, `SetInput`, `DoUnfold`, `_store_covariances`, `_store_unfold_result`. |

## Plotting and conversion helpers

| Method | What it does | Main calls / effects |
| --- | --- | --- |
| `plot_L(show=True)` | Saves the regularization matrix `L` as both ROOT and matplotlib images. | Calls `ROOT.TCanvas`, `Draw("colz")`, `SaveAs`, `GetNbinsX/Y`, `GetBinContent`, `plt.imshow`, `hep.cms.label`, `_finalize_plot`. |
| `_th1_to_arrays(h)` | Converts a ROOT `TH1` into numpy value and error arrays. | Calls `GetNbinsX`, `GetBinContent`, `GetBinError`. |
| `plot_folded(show=True)` | Compares measured reco data with the folded unfolded result. | Calls `unflatten_gen_by_pt`, `plt.subplots`, `hep.histplot`, `np.divide`, `ax.errorbar`, `hep.cms.label`, `_finalize_plot`. |
| `plot_jk(show=True)` | Plots jackknife unfolded outputs and jackknife input spectra per pT bin. | Calls `unflatten_gen_by_pt`, `plt.subplots`, `ax.stairs`, `hep.cms.label`, `_finalize_plot`. |
| `plot_bottom_line(show=True)` | Draws ratio-only validation plots: unfolded/true and measured/reco-MC. | Calls `unflatten_gen_by_pt`, `np.divide`, `hep.histplot`, `hep.cms.label`, `_finalize_plot`. |
| `plot_unfolded_fancy(log=False, show=True)` | Produces the main publication-style unfolded spectra and ratio panels, plus a summary overlay plot. | Calls `unflatten_gen_by_pt`, `plt.subplots`, `plt.stairs`, `plt.plot`, `np.divide`, `hep.cms.label`, `_finalize_plot`. Reads `self.normalized_results`. |
| `plot_unfolded_summary_linear(show=True)` | Draws a linear-scale summary overlay of unfolded spectra across pT bins. | Uses `plt.stairs`, `plt.plot`, `hep.cms.label`, `_finalize_plot`. |
| `plot_unfolded(log=False, show=True)` | Simpler unfolded-vs-truth comparison, with optional Herwig closure uncertainty extraction. | Calls `unflatten_gen_by_pt`, `hep.histplot`, `plt.stairs`, `np.save`, `_finalize_plot`. |

## Normalization and uncertainty bookkeeping

| Method | What it does | Main calls / effects |
| --- | --- | --- |
| `_normalize_result()` | Converts flat unfolded, truth, measured, and reco-MC spectra into normalized per-pT distributions. Also normalizes all systematic variations. | Calls `unflatten_gen_by_pt`, `np.diff`, array division. Fills `self.normalized_results` and `self.normalized_systematics`. |
| `_compute_total_systematic()` | Combines up/down systematics and statistical uncertainties in quadrature. | Loops over `self.systematics`, uses `np.abs`, `np.sqrt`, and stores total, input-stat, matrix-stat, and combined stat uncertainties into `self.normalized_results`. |
| `plot_statistical_fraction(show=True)` | Plots input-stat and matrix-stat fractional uncertainties for each pT bin. | Calls `hep.histplot`, `hep.cms.label`, `_finalize_plot`. |
| `plot_systematic_fraction(syst_name='all', show=True, log=True)` | Thin wrapper for the full systematic summary plot. | Calls `_plot_systematic_fraction_summary(grouped=False, ...)`. |
| `_get_systematic_group_name(syst_name)` | Maps individual systematics into broad categories. | Returns labels such as `Jet Energy`, `Jet Mass`, `Lepton SFs`, `Parton Shower`, `Other Theory`. |
| `_get_systematic_label(syst_name)` | Produces a human-readable label for a single systematic name. | Uses an internal lookup map and strips `Up`/`Down` suffixes. |
| `_get_systematic_summary_name(syst_name, grouped=False)` | Produces the legend name used in systematic summary plots. | Calls `_get_systematic_group_name` and `_get_systematic_label` when needed. |
| `_build_syst_fraction_dict(pt_index)` | Builds per-bin fractional effects of all unfolded systematics relative to nominal. | Uses `np.divide` and stored normalized results. Adds `Stat Unc`, `Total_Up`, and `Total_Down`. |
| `_group_syst_fraction_dict(syst_fraction_dict, grouped=True)` | Combines related systematics in quadrature into grouped plot categories. | Calls `_get_systematic_summary_name`, uses `np.sqrt`, preserves total and stat entries. |
| `_plot_systematic_fraction_summary(grouped=False, show=True, log=True)` | Main systematic-summary plotter for either detailed or grouped views. | Calls `_build_syst_fraction_dict`, `_group_syst_fraction_dict`, `hep.histplot`, `hep.cms.label`, `_cms_extra_label`, `_finalize_plot`. |
| `plot_systematic_fraction_grouped(show=True, log=True)` | Thin wrapper for grouped systematic summary plots. | Calls `_plot_systematic_fraction_summary(grouped=True, ...)`. |
| `plot_systematic_frac_indiv(syst_names=['JES', 'JER'], ylim=None, show=True)` | Plots selected Up/Down systematics one by one. | Uses stored `syst_fraction_dict`, `hep.histplot`, `hep.cms.label`, `_finalize_plot`. |
| `plot_herwig_systematic(show=True)` | Plots the model uncertainty and fits a polynomial trend to the Herwig variation. | Calls `np.sqrt`, `np.diag`, `unflatten_gen_by_pt`, `hep.histplot`, `np.polyfit`, `np.poly1d`, `plt.plot`, `hep.cms.label`, `_finalize_plot`. |

## Matrix-quality plots and orchestration

| Method | What it does | Main calls / effects |
| --- | --- | --- |
| `plot_purity_stability(show=True)` | Computes and plots purity and stability from the generator-level response matrix. | Calls `np.diag`, `unflatten_gen_by_pt`, `hep.histplot`, `plt.stairs`, `hep.cms.label`, `_finalize_plot`. |
| `plot_correlation(show=True)` | Builds and plots the unfolded-bin correlation matrix. | Uses covariance arrays, `np.sqrt`, `np.outer`, `plt.imshow`, `plt.colorbar`, `hep.cms.label`, `_finalize_plot`. |
| `plot_response_matrix(probability=True, log=False, show=True)` | Wrapper around the generic response-mosaic plotter using the nominal matrix. | Calls `_plot_response_mosaic_cms` and `_finalize_plot`. |
| `run_all_plots(show=False)` | Convenience driver that runs nearly every plotting method in sequence. | Calls `plot_unfolded_fancy`, `plot_unfolded_summary_linear`, `plot_statistical_fraction`, `plot_systematic_fraction`, `plot_systematic_fraction_grouped`, `plot_herwig_systematic`, `plot_systematic_frac_indiv`, `plot_correlation`, `plot_unfolded`, `plot_response_matrix`, `plot_folded`, `plot_bottom_line`, `plot_fakes_misses`. |
| `_plot_response_mosaic_cms(...)` | Low-level renderer for the response matrix mosaic with pT block boundaries and optional probability normalization. | Uses `np.sum`, `np.nan_to_num`, `np.linalg.svd`, `np.linalg.cond`, `plt.subplots`, `np.ma.masked_where`, `LogNorm`, `ax.imshow`, `ax.axhline`, `ax.axvline`, `fig.colorbar`, `hep.cms.label`. Returns `(fig, ax)`. |

## Input-building utilities

| Method | What it does | Main calls / effects |
| --- | --- | --- |
| `_make_inputs_numpy(filenames=[...])` | Precomputes and assembles the systematic response-matrix dictionary from per-era pickle inputs. This is the largest setup method in the class. | Calls `pkl.load`, histogram `project`, `.values()`, `.variances()`, computes correlated and uncorrelated JES combinations, loads the Herwig-model substitute file, and finally fills `self.sys_matrix_dic`. |
| `_merge_eras(filenames=[...])` | Merges reco, gen, and response histograms across all standard eras into a single dictionary and pickle. | Calls `pkl.load`, histogram `project`, histogram addition, `pkl.dump`. Stores `self.pythia_hists`. |
| `_merge_eras_jk(filenames=[...])` | Merges jackknife response histograms across eras. | Calls `pkl.load`, histogram `project`, histogram addition. Stores `self.pythia_hists_jk` and returns the merged dictionary. |

## Practical notes

- The class is stateful. Most methods do not return analysis objects; they write results into `self`.
- `_make_inputs_numpy()` must run before `_configure_systematics()` because `self.sys_matrix_dic` is created there.
- `_load_data()` prepares both nominal and systematic response matrices, but actual unfolding is done later in `_perform_unfold()`.
- Plotting methods assume the full constructor workflow has already completed.
