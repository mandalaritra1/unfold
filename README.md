Use the latest shared unfolder implementation in `src/unfold/tools/unfolder_core.py`.

## Documentation

- Core class reference: [docs/Unfolder_core_class_reference.md](docs/Unfolder_core_class_reference.md)
- Dijet/trijet rho runner: [docs/rho_channel_unfolding.md](docs/rho_channel_unfolding.md)

For the rho workflow:

- Run `notebooks/unfolder_v4_rho.ipynb` to produce tagged rho outputs under `outputs/rho/original/` and `outputs/rho/fixed_jec/`.
- Review saved plots in `notebooks/rho_review.ipynb`.
- Build static scrollable galleries with `python3 outputs/build_rho_gallery.py --root outputs/rho/original` and `python3 outputs/build_rho_gallery.py --root outputs/rho/fixed_jec`.
- Open `outputs/rho/original/index.html` or `outputs/rho/fixed_jec/index.html` in a browser when you want a fast overview of many plots.

## Environment

Activate the project venv from the repository root:

```bash
source .venv/bin/activate
```

The activation hook sources `scripts/setup_root.sh`, which defaults ROOT to
`/Users/aritra/opt/root-6.40.00-rc1`.  To source it manually in an existing
terminal:

```bash
source scripts/setup_root.sh
```

## Dijet/trijet rho unfolding

Run the producer-compatible 2018 workflows with:

```bash
source scripts/setup_root.sh

PYTHONPATH=src .venv/bin/python scripts/run_rho_unfolding.py \
    --channel dijet --year 2018

PYTHONPATH=src .venv/bin/python scripts/run_rho_unfolding.py \
    --channel trijet --year 2018
```

Outputs are separated under
`outputs/<channel>/<year>/rho/unfolding/`. These runs intentionally omit
jackknife response statistics and HERWIG/model uncertainty. Dijet/trijet
detector-level validation is deferred until dedicated validation inputs are
available. Plot definitions and formatting come from the same
`Unfolder.run_all_plots()` methods used by the Z+jet workflow.

The dijet groomed result uses a studied variable low-rho binning in the
400-570 and 570-760 GeV intervals. Trijet and legacy Z+jet binning are
unchanged; details are in
[docs/dijet_groomed_rho_binning_study.md](docs/dijet_groomed_rho_binning_study.md).

## Run 2 detector-level rho validation

Create the combined 2016--2018 data/MC plots with:

```bash
.venv/bin/python notebooks/data_mc_rho_fancy.py \
    --input-production-tag validation
```

The CMS Internal PDFs are written to `outputs/rho/data_mc/`, with matching
CMS Preliminary versions in `outputs/rho/data_mc/Preliminary/`. The generated
`run2_plot_config.json` records the command, phase-space configuration, input
production tag, and SHA-256 hash of every input pickle.
