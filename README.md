# unfold

Jet substructure unfolding for the CMS Z+jet / dijet / trijet analyses. The
repository unfolds two observables — **rho** (`log10(rho^2)`) and **jet mass** —
each for three channels: **zjet**, **dijet**, **trijet**.

A single shared implementation, `Unfolder` in
[`src/unfold/tools/unfolder_core.py`](src/unfold/tools/unfolder_core.py), drives
every cell. An `ObservableSpec` parameterizes it per observable, and a
`(channel, observable, tag)` registry (`CHANNEL_OBSERVABLES`, `get_spec`) makes
the full matrix first-class.

## The matrix

| Channel \ Observable | rho | mass |
|---|---|---|
| **zjet** | ✅ `RHO_SPECS["original"]` (default) + `["fixed_jec"]` | ⚙️ `MASS_SPEC` — code ready, **inputs must be regenerated** |
| **dijet** | ✅ 2018, prepared channel inputs | — not available |
| **trijet** | ✅ 2018, prepared channel inputs | — not available |

- **zjet** rho/mass run from merged-era pickles (`inputs/zjet/...`) via the spec
  path (`Unfolder(spec, groomed).run_all_plots()`), interactively from notebooks
  or from the CLI.
- **dijet/trijet** rho run from prepared per-channel inputs
  (`inputs/<channel>/rho/`) and omit jackknife response statistics (no jackknife
  inputs). The HERWIG/model uncertainty is included **when** a
  `minimal_rho_<channel>_herwig_<year>.pkl` is present (auto-discovered; dijet
  2018 has one, so it gets the alternate-generator model uncertainty + bias
  test); otherwise it is skipped. Detector-level validation is deferred until
  dedicated validation inputs exist.

## Repository layout

```
src/unfold/tools/
  unfolder_core.py     # Unfolder class, ObservableSpec, specs + registry
  binning.py           # binning helpers
  rho_channel_inputs.py# dijet/trijet input discovery + adaptation
  hepdata_export.py    # HEPData export
  merge_data.py        # per-era pickle merger
src/unfold/utils/      # integrate_and_rebin, merge_helpers
scripts/               # run_unfolding.py (unified), run_rho_unfolding.py, hepdata,
                       # purity, and study_*/plot_* cross-checks (see "Studies")
notebooks/             # interactive runners (unfolder_v4_{rho,mass}, data_mc_rho_fancy)
inputs/                # gitignored data; see inputs/README.md for the layout
  zjet/{rho/{original,fixed_jec},mass,validation}/
  {dijet,trijet}/rho/
  _archive/
outputs/               # plots + artifacts, mirrors inputs by channel
  zjet/{rho/{original,fixed_jec},mass,validation}/
  {dijet,trijet}/<year>/rho/
  _archive/
docs/                  # reference docs (see Documentation below)
```

Input pickles are gitignored — [`inputs/README.md`](inputs/README.md) is the
tracked record of the expected files and their provenance.

## Environment

Activate the project venv from the repository root; the activation hook sources
`scripts/setup_root.sh` (ROOT defaults to `/Users/aritra/opt/root-6.40.00-rc1`):

```bash
source .venv/bin/activate
# or, in an existing shell:
source scripts/setup_root.sh
```

Install the package so `unfold` is importable without `PYTHONPATH=src`:

```bash
pip install -e .            # deps come from requirements.txt; PyROOT stays external
```

`import ROOT` is required for an actual unfolding run and is provided by the
external ROOT build, not pip. The scripts also fall back to adding `src/` to
`sys.path`, so they work even without the editable install.

## Running

### Unified runner

[`scripts/run_unfolding.py`](scripts/run_unfolding.py) is the single entrypoint:

```bash
source scripts/setup_root.sh

# zjet rho (default tag = original); use --tag fixed_jec for the JEC-fixed set
python scripts/run_unfolding.py --channel zjet  --observable rho
python scripts/run_unfolding.py --channel zjet  --observable rho --tag fixed_jec

# Every tag has a "<tag>_jacobian" twin (same inputs) whose normalized-result
# statistics are propagated through the normalization Jacobian: error bars and
# the correlation matrix describe the normalized spectrum, and a
# normalized_covariance_{groomed,ungroomed}.npz (stat + rank-1 systematic
# covariances) is written under unfold/. Outputs land in a sibling dir
# (e.g. outputs/zjet/rho/original_jacobian/) for comparison-app pairing.
python scripts/run_unfolding.py --channel zjet  --observable rho --tag original_jacobian

# "<tag>_jacobian_reg" additionally enables ratio-curvature regularization:
# custom L rows penalize the curvature of x/x_MC per pT slice (zero penalty
# for spectra proportional to the MC prior), tau from an L-curve scan on the
# nominal data unfold and frozen for systematic/jackknife re-unfolds.
# Validation: scripts/study_regularization_rho.py (exact self-closure, <1%
# added HERWIG-closure bias, roughly halved input-stat uncertainties).
python scripts/run_unfolding.py --channel zjet  --observable rho --tag original_jacobian_reg

# The same options are available as flags for ANY channel/observable/tag;
# flag runs auto-suffix the output dir (_jacobian/_reg) so the baseline
# outputs are never overwritten. --tau fixes the strength (skips the scan).
python scripts/run_unfolding.py --channel zjet  --observable mass --jacobian --regularization ratio_curvature
python scripts/run_unfolding.py --channel dijet --observable rho --year 2018 --jacobian --regularization ratio_curvature
# -> outputs/dijet/2018/rho/unfolding_jacobian_reg/ (settings + per-mode tau in run_manifest.json)

# Backend choice: iterative Bayes / D'Agostini via RooUnfold (CMS-recommended)
# instead of TUnfold. The jackknife replicas re-unfold through it unchanged, so
# the statistical uncertainty stays jackknife-based. Needs a built libRooUnfold
# (source scripts/setup_roounfold.sh); --n-iter sets the iterations (default 4).
# The output dir gets a '_bayes' suffix.
source scripts/setup_roounfold.sh
python scripts/run_unfolding.py --channel zjet --observable rho --method roounfold_bayes --n-iter 4
# -> outputs/zjet/rho/original_bayes/  (full plot suite + 2D summaries through Bayes)
# Works for dijet/trijet too; those have no jackknife inputs, so the Bayes stat
# uncertainty falls back to RooUnfold's propagated covariance.
python scripts/run_unfolding.py --channel dijet --observable rho --year 2018 --method roounfold_bayes
# -> outputs/dijet/2018/rho/unfolding_bayes/

# dijet / trijet rho (delegates to run_rho_unfolding.py)
python scripts/run_unfolding.py --channel dijet  --observable rho --year 2018
python scripts/run_unfolding.py --channel trijet --observable rho --year 2018

# zjet mass (requires regenerated inputs/zjet/mass/ pickles)
python scripts/run_unfolding.py --channel zjet --observable mass
```

`run_unfolding.py --help` works without ROOT. Unavailable combinations
(dijet/mass, trijet/mass) exit with a clear message.

### Notebooks (interactive)

- [`notebooks/unfolder_v4_rho.ipynb`](notebooks/unfolder_v4_rho.ipynb) — produces
  tagged zjet rho outputs under `outputs/zjet/rho/original/` and
  `outputs/zjet/rho/fixed_jec/`.
- [`notebooks/unfolder_v4_mass.ipynb`](notebooks/unfolder_v4_mass.ipynb) — zjet
  mass workflow.

### Dijet/trijet direct runner

The unified runner delegates to the producer-compatible path, which can also be
called directly:

```bash
python scripts/run_rho_unfolding.py --channel dijet  --year 2018
python scripts/run_rho_unfolding.py --channel trijet --year 2018
```

Outputs go to `outputs/<channel>/<year>/rho/unfolding/` (plots, NPZ artifacts,
a run manifest, and a text summary). The dijet groomed result uses a studied
variable low-rho binning in the 400–570 and 570–760 GeV intervals; trijet and
legacy zjet binning are unchanged
([docs/dijet_groomed_rho_binning_study.md](docs/dijet_groomed_rho_binning_study.md)).

## Galleries

Build static scrollable HTML galleries of the saved plots:

```bash
python outputs/build_rho_gallery.py  --root outputs/zjet/rho/original
python outputs/build_rho_gallery.py  --root outputs/zjet/rho/fixed_jec
python outputs/build_mass_gallery.py --root outputs/zjet/mass
```

Then open the generated `index.html` in the chosen `--root`.

> The external `unfold-rho-compare` app syncs committed PNG previews from
> `outputs/zjet/rho/{original,fixed_jec,original_jacobian}/_previews/`; its
> sync script handles the channel-reorganized layout and takes
> `--unfold-root` / `--versions` arguments.

To combine selected PNG/JPEG plots into a configurable slide-ready grid:

```bash
python scripts/serve_image_grid.py
```

The local webpage supports folder selection, thumbnail filtering and
selection, drag reordering, adjustable columns/spacing/padding, and copying or
downloading the rendered grid as a PNG.

## Run 2 detector-level rho validation

```bash
python notebooks/data_mc_rho_fancy.py --input-production-tag validation
```

CMS Internal PDFs are written to `outputs/zjet/rho/data_mc/`, CMS Preliminary
versions to `outputs/zjet/rho/data_mc/Preliminary/`. A `run2_plot_config.json`
records the command, phase-space configuration, input production tag, and a
SHA-256 hash of every input pickle. Inputs come from `inputs/zjet/validation/`.
The plotting command also refreshes `outputs/zjet/rho/data_mc/index.html` and
its cached PNG previews; pass `--no-gallery` to skip that step.

## Studies and cross-checks

Standalone scripts that probe the robustness of the zjet rho unfolding. Each
writes a self-contained folder under `outputs/zjet/rho/` with its own
`README.md` summarizing method and results. The heavy producer `--*-mc` pickles
are **not** committed, but the small `.npz` artifacts are, so the plot-only
helpers can redraw figures without ROOT.

```bash
source scripts/setup_root.sh

# Regularization L-curve / closure study (tau scan, self-closure, HERWIG bias)
python scripts/study_regularization_rho.py

# Response reweight-to-data: rebuild the response from a data-matched gen prior
python scripts/study_response_reweight.py            # -> outputs/zjet/rho/reweight_test/

# Data-prior response test: unfold through a response whose gen prior is
# reweighted toward the unfolded data (needs the weighted producer pickle)
python scripts/study_data_prior.py --weighted-mc /path/to/weighted_pythia_all.pkl
python scripts/plot_data_prior_unfolded_comparison.py  # redraw from committed npz, no ROOT
#   -> outputs/zjet/rho/original_data_prior_test/

# Rho-averaged-per-pT jackknife stat-uncertainty convergence sheet
python scripts/plot_jk_convergence_pt_avg.py --tag original
#   -> outputs/zjet/rho/original/unfold/jackknife_convergence_pt_avg_{mode}.pdf

# D'Agostini (iterative Bayes) via RooUnfold vs TUnfold, reusing the same
# jackknife replicas for the stat uncertainty (needs a built libRooUnfold)
source scripts/setup_roounfold.sh
python scripts/study_roounfold_bayes.py --tag original --n-iter 4
#   -> outputs/zjet/rho/original/roounfold_bayes/
```

## HEPData export

```bash
python scripts/run_hepdata_export.py --spec fixed_jec        # -> outputs/zjet/rho/hepdata
python scripts/build_hepdata_submission.py --root outputs/zjet/rho/hepdata
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite uses the stdlib `unittest` (no extra dependency).

## Documentation

- [docs/Unfolder_core_class_reference.md](docs/Unfolder_core_class_reference.md)
  — `Unfolder` / `ObservableSpec` reference.
- [docs/rho_channel_unfolding.md](docs/rho_channel_unfolding.md) — dijet/trijet
  rho runner details.
- [docs/dijet_groomed_rho_binning_study.md](docs/dijet_groomed_rho_binning_study.md)
  — the dijet groomed low-rho binning study.
- [docs/ratio_curvature_regularization.md](docs/ratio_curvature_regularization.md)
  — how the regularization L-matrix is modified (curvature of `x/x_MC`), with a
  figure of its block-diagonal structure and the validation summary.
- [inputs/README.md](inputs/README.md) — input file layout and provenance.
