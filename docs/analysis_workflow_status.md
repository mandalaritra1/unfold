# Rho unfolding workflow status and provenance

**Status date:** 2026-08-02
**Scope:** current Z+jet rho work, with the related 2018 dijet/trijet and
exploratory Combine/theory studies.

This page records the working analysis decisions and validation boundaries. It
does not promote an exploratory cross-check to a production result.

## Confirmed unfolding behavior to retain

The following changes are intentional and must be retained when organizing the
worktree:

- Non-DY background is subtracted before the DY fake correction. The ARC
  background must be produced with the current ARC binning; it must not be
  borrowed from a differently binned response.
- The ARC/JMS/JMR response inputs, pT-sink treatment, groomed floors, and
  normalization over the shown window are part of the intended configuration.
- When response-matrix jackknife inputs are absent, response-matrix statistical
  covariance is included as the fallback uncertainty treatment.
- The model envelope source choice and the requested covariance-correlation
  scope are part of the result definition.
- Full-covariance chi-square with the corresponding effective degrees of
  freedom, data-prior axis rebinning, and exported processor lookup weights are
  intended features.
- The JMS/JMR-unity response shifts and their uncertainty interpretation are
  retained as a result variant, not discarded as a plotting-only change.

These items affect response matrices, corrections, covariance construction,
normalization, or uncertainty interpretation. Any future refactor must retain
them explicitly and rerun the appropriate closure, purity/stability, and
regularization checks before interpreting changed results.

## Input and provenance state

### ARC background input

`RHO_ARC_R2_SPEC` requests:

```text
inputs/zjet/rho/arc_r2/bkg_all.pkl
```

The confirmed current-binning artifact is
`inputs/zjet/rho/jmsjmr_unity/bkg_all.pkl`. ARC resolves that pkl explicitly
through `../jmsjmr_unity/bkg_all.pkl`; this avoids maintaining a duplicate or
silently falling back to an unsubtracted input. The input is still gitignored,
so a run must verify that it is present locally.

### Latest retained candidate

The current candidate retained in the checkout is:

```text
outputs/zjet/rho/jmsjmr_unity_groomed400_floor3/
```

It is intentionally ignored by Git. It currently has result arrays, plots, and
derived summaries but no complete `run_manifest.json` capturing the exact
command, Git revision, input hashes, and environment. It must therefore be
treated as a **candidate with incomplete replay provenance**, not as a
bit-for-bit reproducible release.

For a future rerun, the manifest should record at least:

- channel, observable, tag, regularization/backend settings, and
  model-covariance scope;
- resolved input paths and SHA-256 hashes, including the background input;
- Git commit plus whether the tree was dirty;
- ROOT/Python environment identifiers; and
- the exact command and completion time.

### Output retention policy

Generated outputs are excluded from Git. The current archive is:

```text
/Users/aritra/cernbox (2)/unfold_outputs/2026-08-02/outputs/
```

It contains the moved historical rho, dijet, trijet, Combine, validation, and
other generated outputs. The latest `jmsjmr_unity_groomed400_floor3` result
remains in the checkout by request. The two gallery-builder scripts under
`outputs/` remain trackable source files.

Existing historical outputs that were tracked in Git still appear as deletions
until a later, reviewable cleanup commit records their removal from the index.
Ignoring new generated outputs does not itself rewrite that history.

## Exploratory Combine and theory work

The Combine and theory/NLO scripts remain in the repository because they are
useful cross-check and development paths. They are not production unfolding
results at this point.

- The Z+jet full profiled Combine run completed, but its covariance quality was
  `covQual=2`; use it for comparison with TUnfold, not as an uncertainty result
  without further validation.
- The Z+jet self-closure reached a maximum absolute residual of about 3.2% in
  the archived session. The independent 50:50 closure had percent-level median
  residuals but larger sparse/edge-bin excursions.
- Dijet/trijet nominal Combine cards are diagnostic. The full profiled groomed
  nuisance fits failed, and dijet ungroomed has a zero fake-survival bin, so
  none of these channel results should be labelled production-ready.
- The NLO workflow remains a handoff/staging study. It expects external NLO
  producer pickles and has not established a registered, validated NLO result
  tag in this repository.

The archived session note with detailed commands and fit diagnostics is at:

```text
/Users/aritra/cernbox (2)/unfold_outputs/2026-08-02/outputs/session_summary/unfold_combine_session_summary.md
```

## Nonportable paths requiring an explicit policy

These paths are intentionally not changed in this organization pass because
changing an output location or input source can alter a review handoff. They
should be converted to explicit command-line options or environment variables
once their desired destinations are confirmed.

| Location | Current behavior | Safe replacement direction |
|---|---|---|
| `notebooks/plot_channel_split_rho.py` | Takes explicit `--input` and `--output-dir` arguments; defaults preserve the local AN handoff. | Override destinations on another machine or for an archive-only run. |
| `scripts/diagnostics/study_bottom_line_ratio.py`, `scripts/diagnostics/study_bottom_line_merge30.py`, `scripts/diagnostics/study_purity_stability_figs.py` | Take `--slide-dir` and `--no-slide-copy`; defaults preserve the local review handoff. | Use `--no-slide-copy` for a generated-output-only run. |
| `scripts/studies/derive_vincia_reweight.py` | Takes `--vincia-dir`, `--pythia-input`, and `--output-dir`; the local CERNBox directory remains the default and `VINCIA_DIR` is still honoured. | Supply explicit paths and retain the source hash for a result. |
| `scripts/studies/study_nlo_response_unfold.py` and `docs/lo_vs_nlo_handoff.md` | Take explicit NLO input and output paths; `~/Downloads` remains a local development default. | Record the staged NLO source and hash in a run manifest. |
| `inputs/zjet/nlo_skims` | Local symlink to a CERNBox sync directory. | Ignore it in Git and document the canonical external source before it is used in a run. |

`scripts/setup_root.sh` also supplies a macOS-local default ROOT location, but
it is deliberately overridable with `UNFOLD_ROOTSYS`; this is environment
configuration rather than an analysis-input provenance issue.
