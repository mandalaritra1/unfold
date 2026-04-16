# `Unfolder` class reference (core mass + rho implementation)

Source: `src/unfold/tools/unfolder_core.py`

This is the current reference for the shared `Unfolder` class used for both mass and rho workflows, implemented in `unfolder_core.py` and parameterized by `ObservableSpec`.

## Quick start

```python
from unfold.tools.unfolder_core import Unfolder, MASS_SPEC, RHO_SPEC

# Mass unfolding
u_mass = Unfolder(spec=MASS_SPEC, groomed=True, do_syst=True, cms_label="Internal")

# Rho unfolding
u_rho = Unfolder(spec=RHO_SPEC, groomed=False, do_syst=True, cms_label="Internal")
```

## Architecture update

- `ObservableSpec`: declarative config for observable-dependent behavior:
  - axis names (`reco_axis`, `gen_axis`)
  - input/output paths and filenames
  - histogram key names
  - bin-edge attribute names
  - labels, axis limits, normalized y-axis text
- `MASS_SPEC` and `RHO_SPEC` are concrete instances passed into `Unfolder`.
- One `Unfolder` implementation now supports both observables.

## High-level constructor flow

When an object is created:

1. `_setup_binning()`
2. `_make_inputs_numpy()`
3. `_configure_systematics()`
4. `_load_data()`
5. `_perform_unfold()` (nominal + requested systematics)
6. `_compute_stat_unc()`
7. `_normalize_result()`
8. `_compute_total_systematic()`

## Method reference

### Constructor and config helpers

| Method | Purpose |
| --- | --- |
| `__init__(spec, groomed, closure=False, herwig_closure=False, do_syst=False, cms_label="Internal")` | Main entry point. Runs full setup, unfolding, and uncertainty pipeline. |
| `_setup_binning()` | Loads binning from `binning.bin_edges` using `spec` attribute names. |
| `_configure_systematics(do_syst)` | Uses all `self.sys_matrix_dic` keys when enabled, else `["nominal"]`. |
| `_load_pickle(filename)` | Reads pickle file. |
| `_resolve_input_path(filename, *fallbacks)` | Returns first existing candidate path; raises `FileNotFoundError` if none exist. |
| `_finalize_plot(save_path=None, show=True, fig=None)` | Save/show/close wrapper used by plotters. |
| `_cms_extra_label()` | Ensures CMS extra label spacing is consistent. |
| `_observable_label()` | Axis label for current observable and groomed mode. |
| `_observable_short_label()` | Short observable label variant. |
| `_observable_xlim(i_pt=None)` | Lower bound from `spec`, upper bound from active pt-bin edges. |
| `_normalized_ylabel()` | Returns normalized y-axis label from `spec`. |
| `_histogram_keys()` | Returns groomed/ungroomed key map (`response`, `reco`, `gen`). |

### Input and response preparation

| Method | Purpose |
| --- | --- |
| `_prepare_jackknife_inputs(...)` | Builds flattened reco vectors for 10 JK inputs. |
| `_prepare_truth_spectrum(...)` | Builds flattened gen-truth spectrum (nominal if systematic axis exists). |
| `_prepare_jackknife_response_matrices(...)` | Builds 10 JK response mosaics. |
| `_prepare_nominal_inputs(...)` | Prepares nominal reco/fake/miss arrays and nominal gen-response mosaic. |
| `_prepare_herwig_inputs(...)` | Prepares Herwig reco/fake/miss arrays and Herwig gen-response mosaic. |
| `_response_matrix_for_systematic(...)` | Returns matrix for a given systematic (with Herwig branch handling). |
| `_compute_fake_fraction(fakes_flat, matched_flat)` | Safe fake-fraction computation with clipping to `[0, 1]`. |
| `_finalize_reco_views(...)` | Finalizes nominal reco-side views and fake fractions. |
| `_load_data(...)` | Merges eras, loads pkl inputs, prepares JK objects, and fills response dictionaries. |

### Unfold core

| Method | Purpose |
| --- | --- |
| `_compute_stat_unc()` | JK-based input and matrix statistical uncertainty propagation. |
| `_select_measured_spectrum(closure, herwig_closure, meas_flat)` | Chooses measured spectrum source for unfolding. |
| `_apply_fake_correction(...)` | Applies fake correction unless in closure mode. |
| `_build_root_binning()` | Builds `TUnfoldBinning` trees for truth/reco. |
| `_fill_root_histogram(hist, values)` | Fills TH1 from flat array. |
| `_fill_response_histogram(h_resp, resp_np, misses)` | Fills migration matrix + miss row. |
| `_store_covariances(unfold, systematic)` | Extracts covariance matrices from TUnfold. |
| `_store_unfold_result(...)` | Stores nominal/systematic or JK unfolding outputs. |
| `_perform_unfold(...)` | Runs `ROOT.TUnfoldDensity` and stores outputs. |

### Plotting and array conversion

| Method | Purpose |
| --- | --- |
| `plot_L(show=True)` | Saves ROOT and matplotlib views of regularization matrix `L`. |
| `_th1_to_arrays(h)` | Converts TH1 content/errors to numpy arrays. |
| `plot_folded(show=True)` | Measured vs folded comparison with ratio panel. |
| `plot_jk(show=True)` | JK outputs and JK input spectra per pt bin. |
| `plot_bottom_line(show=True)` | Ratio-only validation: unfolded/true and measured/reco-MC. |
| `plot_fakes_misses(show=True)` | Fake-rate and efficiency visualizations per pt bin. |
| `plot_input_data_mc(show=True)` | Reco-level Data/PYTHIA/HERWIG overlay + theory/data ratio panel. |
| `plot_unfolded_fancy(log=False, show=True)` | Main publication-style plot + summary overlay. |
| `plot_unfolded_summary_linear(show=True)` | Linear summary overlay across pt bins. |
| `plot_herwig_pythia_comparison(show=True)` | 4-panel comparison: spectra, P8/H7 ratio, gen-model unc, unfolded-model unc. |
| `plot_unfolded(log=False, show=True)` | Basic unfolded-vs-prior/Herwig comparison and optional closure-unc save. |

### Normalization and uncertainty helpers

| Method | Purpose |
| --- | --- |
| `_normalize_result()` | Builds normalized per-pt dictionaries for nominal and each systematic. |
| `_compute_total_systematic()` | Combines systematics and stat components in quadrature per bin. |
| `plot_statistical_fraction(show=True)` | Input-stat and matrix-stat fractional uncertainty plots. |
| `plot_systematic_fraction(syst_name='all', show=True, log=True)` | Wrapper for detailed systematic summary. |
| `_get_systematic_group_name(syst_name)` | Maps names to broad groups (Jet Energy, Lepton SFs, etc.). |
| `_get_systematic_label(syst_name)` | Human-friendly label normalization. |
| `_split_systematic_variation(syst_name)` | Splits base name and `Up`/`Down` tag. |
| `_get_systematic_summary_name(syst_name, grouped=False)` | Summary legend name resolution. |
| `_build_syst_fraction_dict(pt_index)` | Builds per-pt fractional effects map. |
| `_group_syst_fraction_dict(syst_fraction_dict, grouped=True)` | Quadrature-combines related contributions by group/label. |
| `_plot_systematic_fraction_summary(grouped=False, show=True, log=True)` | Main summary plot logic (grouped or detailed). |
| `plot_systematic_fraction_grouped(show=True, log=True)` | Grouped systematic summary wrapper. |
| `_resolve_raw_systematic_pair(syst_name)` | Resolves matching Up/Down keys for a systematic base. |
| `plot_nominal_minus_variation(syst_names=["pu"], show=True)` | Plots nominal minus variation shifts. |
| `plot_systematic_frac_indiv(syst_names=['JES', 'JER'], ylim=None, show=True)` | Selected individual uncertainty components. |
| `plot_herwig_systematic(show=True)` | Model uncertainty plot + polynomial fit. |

### Matrix diagnostics and orchestration

| Method | Purpose |
| --- | --- |
| `plot_purity_stability(show=True)` | Nominal purity/stability per pt + global-bin view. |
| `plot_purity_stability_herwig(show=True)` | Side-by-side Pythia vs Herwig purity/stability. |
| `plot_correlation(show=True)` | Correlation matrix from covariance components. |
| `plot_response_matrix(probability=True, log=False, show=True)` | Wrapper around response mosaic renderer. |
| `plot_uncertainty_heatmap(show=True)` | Heatmap of grouped fractional uncertainties per mass bin. |
| `run_all_plots(show=False)` | Batch runner for full plotting suite (includes `plot_input_data_mc`). |
| `_plot_response_mosaic_cms(...)` | Low-level response mosaic plotter with pt-block grid and condition number. |

### Input assembly utilities

| Method | Purpose |
| --- | --- |
| `_make_inputs_numpy(filenames=None)` | Builds `self.sys_matrix_dic` from per-era response pickles and systematic rules. |
| `_merge_eras(filenames=None)` | Merges reco/gen/response histograms across eras into `self.pythia_hists` and writes merged MC pkl. |
| `_merge_eras_jk(filenames=None)` | Merges JK response histograms across eras into `self.pythia_hists_jk`. |

## Notes

- The class is stateful: most methods write to `self` instead of returning objects.
- `spec` drives all observable-specific behavior (labels, keys, paths, binning attributes).
- `_make_inputs_numpy()` must run before `_configure_systematics()`.
- `plot_input_data_mc()` now tolerates inputs both with and without an explicit `systematic` axis.
