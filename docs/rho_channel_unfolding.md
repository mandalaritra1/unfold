# Dijet/Trijet Rho Unfolding

The generic rho runner consumes the minimal producer outputs in:

```text
inputs/<channel>/
  minimal_rho_<channel>_data_<year>.pkl
  minimal_rho_<channel>_mg_pythia8_<year>.pkl
```

Supported channels are `dijet` and `trijet`.

## Run

ROOT must be sourced before starting the venv Python:

```bash
source scripts/setup_root.sh

PYTHONPATH=src .venv/bin/python scripts/run_rho_unfolding.py \
  --channel dijet \
  --year 2018

PYTHONPATH=src .venv/bin/python scripts/run_rho_unfolding.py \
  --channel trijet \
  --year 2018
```

The runner bootstraps both `src/` and `$ROOTSYS/lib`, so the commands remain
valid even though the explicit `PYTHONPATH=src` replaces the value set by
ROOT's `thisroot.sh`.

The runner does not define a separate plotting implementation. After adapting
the channel inputs and installing the channel binning, it calls the shared
`Unfolder.run_all_plots(show=False)` suite used by the Z+jet rho workflow.
HERWIG-, jackknife-, and validation-only methods are skipped because those
inputs are unavailable.

## Physics Configuration

- pT edges: `[0, 200, 290, 400, 570, 760, 13000]` GeV.
- The `0–200` GeV bin remains in the response and unfolding calculation to
  retain migrations, but report plots begin at 200 GeV.
- Reco/gen rho edges are exact subsets of the producer axes and preserve the
  established main-workflow rho binning.
- Dijet groomed rho uses the studied variable low-rho binning documented in
  [dijet_groomed_rho_binning_study.md](dijet_groomed_rho_binning_study.md):
  the 400-570 and 570-760 GeV truth bins are split at `-5`, while all other
  pT intervals retain the established `[-10,-4.5]` merged tail.
- Trijet and all legacy Z+jet bin definitions remain unchanged.
- Data contributes reco histograms only. MC must provide reco, gen, and
  response histograms.
- Every response systematic present in the MG+PYTHIA8 file is unfolded.
  Missing categories are never copied from nominal.
- Fake and miss corrections are calculated from the nominal MG+PYTHIA8
  inclusive reco/gen spectra and matched response.
- Weighted data variances receive the same squared nominal fake-correction
  factor as the measured spectrum, then are passed to TUnfold and propagated
  as the input statistical covariance.

The uncertainty result is **partial**. It includes the input-data statistical
covariance and available response variations, but excludes:

- response-matrix statistical uncertainty, because no jackknife inputs exist;
- HERWIG/alternate-generator model uncertainty, which is intentionally skipped
  for a consistent dijet/trijet definition.

## Outputs

The default output is:

```text
outputs/<channel>/<year>/rho/unfolding/
  unfold/
  uncertainties/
  diagnostics/
  artifacts/
  _previews/
  run_manifest.json
  run_summary.txt
```

`run_manifest.json` records input SHA-256 hashes, ROOT version, command,
systematics, exact binning, luminosity, unavailable uncertainty components,
output paths, and numerical sanity checks.

The future detector-level validation location is reserved as:

```text
outputs/<channel>/<year>/rho/validation/
```

No dijet/trijet validation command or plots are produced until dedicated
validation inputs are available.
