# unfold

TUnfold-based unfolding of the jet-mass observable rho = m / (pT R), stored as
log10(rho^2), for the three CMS Run-2 channels **zjet**, **dijet** and
**trijet**.  The second observable, the jet mass, has the code path but no
inputs yet.

```bash
unfold run --channel zjet            # = --observable rho --tag original
unfold run --channel dijet
unfold run --channel trijet
unfold tags                          # every registered channel / observable / tag
```

A tag names one complete configuration (inputs, binning, systematics, knobs)
in `config.TAGS`; `original` is the production configuration of every
channel.  All channels run the same engine (`unfold/engine.py`) and the same
figure suite (`unfold/plots.py`); they differ only in how the inputs are read:

| channel | `original` inputs | loader |
|---|---|---|
| zjet | merged-era coffea pickles, `inputs/zjet/rho/jmsjmr_unity/` | `zjet_inputs.py` |
| dijet, trijet | full Run-2 hadronic pickles on CERNBox (`paths.py`) | `hadronic/` |
| dijet, trijet, tag `2018` | single-year `inputs/<channel>/rho/minimal_rho_*.pkl` | `channel_inputs.py` |

## Setup

One command, from the repository root, in every new shell:

```bash
source env.sh
```

It creates `.venv` on first use (Python 3.11, the version the local ROOT
build was compiled against), installs this package into it, activates the
venv and puts ROOT on the path.  After that `unfold`, `python` and `pytest`
refer to this environment.  ROOT with TUnfold comes from an external build
(`UNFOLD_ROOTSYS` in `setup_root.sh`), not from pip; RooUnfold is picked up
when it has been built (`setup_roounfold.sh`, only for `--method roounfold_bayes`).
Everything that lives outside the repository (CERNBox skims, the generator
campaigns of the hadronic modelling uncertainty) is located through the
environment variables listed in `src/unfold/paths.py`.

## Running

```bash
unfold run --channel zjet                        # all systematics, ~10 min per grooming mode
unfold run --channel zjet --no-syst --grooming-mode groomed    # quick nominal-only run
unfold run --channel dijet --tag 2018            # the older single-year inputs

# Run-2 dijet/trijet: fixed-fake jackknife by default, analytic if files are absent
unfold run --channel dijet --stat-method analytic  # outputs/dijet/rho/original_stat_analytic/
unfold run --channel trijet --jackknife-input-root ~/cernbox/hadronic_jackknife_run2_20260908

# option runs never overwrite the tag's outputs: the directory gets a suffix
unfold run --channel zjet --jacobian --regularization ratio_curvature   # outputs/zjet/rho/original_jacobian_reg/

# publication figures: no provenance stamp (date | git revision | inputs)
unfold run --channel trijet --no-stamp --cms-label Preliminary

unfold gallery --root outputs/zjet/rho/original  # rebuild the HTML gallery
```

Outputs go to `outputs/<channel>/<observable>/<tag>/` (git-ignored) in a
categorized layout (`summary/`, `unfolded/`, `uncertainties/`,
`bottom_line/`, `response/`, `validation/`, `data/`); the hadronic channels
add one level, `groomed/` and `ungroomed/`.  Every run writes
`run_manifest.json` with the resolved configuration, the command and the git
revision.  The numeric products are:

* zjet: `data/normalized_covariance_<mode>.npz`, `data/unfolded_2d_<mode>.pkl`,
  `data/uncertainty_summary_2d_<mode>.pkl`
* dijet, trijet: `<mode>/artifacts/<mode>_results.npz` (`artifacts/` for tag `2018`)

Run-2 dijet/trijet statistics use ten data replicas and ten MC replicas, with
the nominal fake fraction fixed. The MC response and GEN/misses vary together.
Each replica is normalized before its covariance is calculated; the independent
data and MC covariances are then added. The selected covariances feed the error
bands, correlations, result chi-squared calculations, and saved arrays. The full
nominal measured covariance remains the fit weight, including dijet event
correlations. The bottom-line test retains its data-statistics-plus-model scope.

The default replica directory is set in `paths.py`; override it with
`UNFOLD_HADRONIC_JACKKNIFE_INPUTS` or `--jackknife-input-root`. Missing any required
era/data/MC file selects the analytical method for that run. Existing malformed
files cause an error. The manifest records the requested and actual method,
fallback reason, replica hashes, fixed-fake policy, and covariance ranks.
`artifacts/jackknife_statistics.npz` retains the replicas and analytical
covariances for comparison. Z+jet and single-year tags keep their existing method.

The September 8 replica campaign carries about 10.5% more weighted data than
the aligned nominal inputs. They are treated as the same sample: the replica
covariance is applied to the nominal central values and the relative
difference is recorded in the manifest (`statistics.data_sample_relative_l1_difference`).
The data statistical uncertainty is therefore that of the slightly larger
sample, about 5% smaller in relative terms. See
[the jackknife study](docs/hadronic_jackknife_comparison.md).

The provenance stamp on every figure is one switch: `--no-stamp` on the
command line or `UNFOLD_NO_STAMP=1` in the environment (`cms_plot.set_stamp`),
and the `scripts/` figure producers honour the same variable.

## Layout

```
src/unfold/
  binning.py        analysis binnings as data (Binning dataclass, ZJET_BINNINGS)
  config.py         the tag registry TAGS[(channel, observable)] and ObservableSpec
  systematics.py    JES year correlations, luminosities, systematic-name helpers
  histmath.py       flatten / merge / mosaic helpers on numpy arrays
  inputs.py         UnfoldInputs (the engine's input contract) + prepared-input builder
  zjet_inputs.py    merged-era Z+jet pickles -> UnfoldInputs
  channel_inputs.py dijet/trijet minimal_rho pickles -> adapted hists
  hadronic/        hadronic inputs, Vincia/CR/frag modelling, diagnostics, run glue
  engine.py         Unfolder: TUnfold / RooUnfold, stat and syst propagation, bottom line
  model.py          model envelope, enclosing-template covariance, prediction statistics
  plots.py          every figure, as functions of a run Unfolder
  cli.py            the `unfold` command
  gallery.py, hepdata.py, cms_plot.py, roounfold.py, theory_*.py
scripts/            plot book, slide deck, data/MC validation figures, HEPData packaging
tests/              unit tests on the pure functions + the golden regression
legacy/             the pre-restructure tree, kept for reference (see legacy/README.md)
docs/               method notes
```

## Adding a tag

Add an entry to `config.TAGS[(channel, observable)]`.  For zjet that is an
`ObservableSpec` built with `dataclasses.replace(...)` from `RHO_BASE`,
`ZJET_RHO_ORIGINAL` or `ZJET_RHO_ARC_R2` (pick the binning by name from
`binning.ZJET_BINNINGS`); for dijet / trijet a `HadronicTag` (normalization
window, binning variant, systematics request, model covariance) or a
`ChannelTag` (year).  Give it `output_dir("<channel>", "<observable>", "<tag>")`.
`unfold tags` lists the registry and `config.describe(tag)` prints one entry.

## Adding a channel

Write a loader that returns an `UnfoldInputs` (see the field comments in
`inputs.py`).  If the producer histograms carry a `systematic` axis, adapt
them and call `inputs.prepared_inputs(...)` as `channel_inputs.py` and
`hadronic/run.py` do.  Then `Unfolder(inputs, spec, groomed).run()` and
`plots.run_all_plots(u)`.

## Checking that nothing changed

`outputs/_golden_legacy/` (local only) holds the data products produced by
the pre-restructure code at tag `snapshot-2026-09-09` for the dijet and trijet
`original` runs (both grooming modes) and zjet `original`.
After a change, rerun the same commands into another directory and compare:

```bash
unfold run --channel zjet  --era-split linear --output-dir outputs/_golden_new/zjet/rho/original  --no-gallery --no-stamp
unfold run --channel dijet --stat-method analytic --output-dir outputs/_golden_new/dijet/rho/original --no-gallery --no-stamp
python tests/compare_golden.py outputs/_golden_legacy/zjet/rho/original outputs/_golden_new/zjet/rho/original
```

The zjet `original` tag is the configuration that was called
`jmsjmr_unity_groomed400_floor3` before 2026-09-09; the golden tree carries
that name's outputs.

`tests/test_golden.py` does the same from pytest when both trees exist.

## Physics changes and caveats

* The Z+jet JES year-correlation split now uses the JetMET prescription
  (amplitudes sqrt(rho), sqrt(1-rho)), as the hadronic channels always did.
  Before 2026-09-09 Z+jet used linear coefficients (rho, 1-rho), which
  under-covers the rho = 0.5 sources by a factor sqrt(2) in amplitude.  The
  central values are unchanged; only the JES legs of the band move.  The old
  behaviour is available as `unfold zjet --era-split linear`, and that is what
  the Z+jet golden regression uses (its reference tree predates the fix).
  Measured effect: central values identical, JES leg +4-8% per bin (max
  +19%), total band +0.2-0.8% per bin (max +4.6%).  See
  `systematics.era_split_coefficients`.
* The `herwig` systematic of the legacy Z+jet tags projects the fallback
  response without selecting a systematic category (see
  `zjet_inputs._reweighted_response`).
