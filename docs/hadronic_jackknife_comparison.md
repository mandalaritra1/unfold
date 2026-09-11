# Hadronic Run-2 jackknife comparison, 2026-09-10

The September 10 comparison covered dijet and trijet, groomed and ungroomed,
using ten leave-one-out replicas from all four Run-2 eras. That diagnostic did
not change production outputs. On September 11, Aritra selected fixed-fake
jackknife as the default Run-2 CLI statistical method, with an analytical
fallback when replica files are absent.

## CLI integration, 2026-09-11

`unfold run --channel dijet` and `--channel trijet` now request jackknife.
`--stat-method analytic` selects the old method explicitly and uses a
`_stat_analytic` output suffix unless `--output-dir` is supplied.
`--jackknife-input-root` overrides the campaign location. Z+jet and the
single-year tags retain their previous behavior.

`src/unfold/hadronic/jackknife.py` validates the input files and selects groups
by category label. It runs ten data and ten response/GEN/misses fits with the
nominal fake fraction, full measured fit covariance, and selected tau fixed.
Data replicas are multiplied by `10/9` for absolute-yield covariance; that
factor cancels for normalized spectra. Per-pT and globally normalized
covariances use explicitly normalized replicas. Data and response covariances
replace the corresponding analytical components in error bars, totals,
correlations, chi-squared calculations, and exports. The bottom-line test
retains its existing data-statistics-plus-model scope. The analytical matrices
and replica spectra remain available in `artifacts/jackknife_statistics.npz`.

Any missing required era/data/MC file selects the analytical method for the
whole run, with the reason in the manifest. Existing corrupt files and
incompatible histogram axes or labels fail explicitly. The manifest records
the actual method, file hashes, fixed-fake policy, group count, and ranks.

The CLI compares the nominal data with the full sample reconstructed from the
replicas and records the relative L1 difference in the manifest (10.51% for
groomed dijet with the current inputs). On 2026-09-11 Aritra decided to treat
the two as the same campaign, so the difference is reported, not fatal; the
replica covariance is applied to the nominal central values.

The four covariance calculations reproduce the accepted September 10 study
using its exact normalization windows and matching reconstructed sample.
The check also verifies the loader against the label-aligned audit arrays and
retains the full input fit covariance. Evidence is in
`outputs/jackknife_cli_validation_20260911/same_sample_validation.json`.
All 45 tests passed. Fresh analytical dijet, trijet, and Z+jet runs match the
pre-restructure golden products using `tests/compare_golden.py`; comparisons
with the cached current references have zero numerical difference. Four
selected jackknife correlation renders were inspected. The validation receipt
and logs are in the same directory. The full run also found and fixed a MESS
audit path check affected by the CERNBox rename; manifest and ntuple hash
validation is preserved when the campaign's parent directory moves.

## Inputs and preparation

Source: `/Users/aritra/cernbox/hadronic_jackknife_run2_20260908/`.
All 16 pickle sizes and SHA256 hashes agree with the supplied manifest. Required
histograms have finite values, finite nonnegative sumw2, nominal systematics,
the expected fine axes, and all ten replica labels. Labels are not in a common
storage order across histograms; selection uses `hist.loc(label)` before merging.

The producer assigns groups with chunk-local event index modulo ten. Both jets
from one event remain grouped, but partitioning is not invariant under
rechunking. MC replicas use the full-sample generator-weight denominator, as
documented in `HadronicProcessorBase._accumulate_sumw`. There is no full-sample
`jk=-1` category. Each event contributes to nine replicas, so their sums divided
by nine reconstruct full-coverage average values **and** sumw2. This relies on
replica-independent event processing; these aggregate outputs do not verify
that stochastic jet corrections are identical across replicas.

The three accepted data shortfalls remain in the audit metadata. No production
rerun was requested or performed.

## A different data sample from the saved reference

The reference is `outputs/_golden_new/{channel}/rho/original/`, produced by the
current engine during the repository restructure. Its input paths identify the
older `hadronic_minimal_rho_pairsplit_aligned` campaign. Those data files have
fewer initial events than the new jackknife files. For dijet 2018, the counts
are 624,120,640 and 676,743,923, respectively. New weighted reco yields exceed
the older yields by roughly 8-15% depending on era and channel. The four
rebinned Run-2 comparisons differ by about 10.5% in summed absolute raw yield.
This establishes different coverage; it does not establish why the earlier
campaign processed fewer events.

The central MC comparisons are much closer: summed absolute bin differences
are below 0.053% for GEN and below 0.149% for the response, relative to the older
totals. After unfolding and normalization, the new central distributions differ
from the old ones by 0.62-0.98% in summed absolute difference relative to the
sum of the old normalized bin values over the comparison window. These are
aggregate metrics, not maximum per-bin differences.

Production dijet unfolding requires the data-derived event-level reco covariance
and supplies it to TUnfold, including correlations between jets from the same
event. The saved dijet manifest confirms `full_reco_covariance`; production is
not using a diagonal-only input covariance.

The new jackknife data pickles omit the event-level reco covariance. Accordingly, the
fit keeps the saved nominal raw data covariance as its weight matrix. The
same-sample analytic data comparator propagates the **new diagonal sumw2**
through that fixed fit and the normalization Jacobian. For dijet this omits
correlations between jets from the same event. It is not a validation of the
production full-data covariance. The older analytic input covariance is saved
separately and is not used as the main same-sample comparator.

## Three comparisons

Each replica uses the unchanged TUnfold engine, the reference binning and
normalization windows, tau=0 and the area constraint. Each unfolded replica is
normalized separately. Covariance is `(g-1)/g * sum(delta_i delta_i.T)`, with
`g=10` and deviations from the replica mean.

- Data: vary the measured spectrum with nominal MC. Compare against the new
  diagonal sumw2 propagated through the same fit.
- Response MC: vary response and misses together, with nominal fake survival.
  Compare against TUnfold `GetEmatrixSysUncorr` with weighted response/miss sumw2.
  Jackknife includes event-level MC correlations that the binwise analytic
  approximation cannot reconstruct.
- Full MC diagnostic: additionally vary fake survival consistently with each
  MC replica. This tests a broader statistical contribution than the analytic
  response-only term, so the difference is not itself an analytic-method failure.

The median ratios of jackknife to analytic error over the selected bins are:

| Channel | Grooming | Data / new diagonal input | Response / TUnfold response | Full MC / TUnfold response |
|---|---|---:|---:|---:|
| Dijet | Groomed | 0.914 | 1.116 | 4.589 |
| Dijet | Ungroomed | 0.847 | 1.125 | 2.211 |
| Trijet | Groomed | 1.029 | 0.875 | 1.271 |
| Trijet | Ungroomed | 0.935 | 0.910 | 1.266 |

The response-only errors are of similar scale, with appreciable binwise scatter.
The full MC diagnostic is larger in some dijet bins, reaching a factor 14.88
relative to the small response-only analytic term. On September 10, Aritra
decided to set aside fake-correction statistics. Subsequent matrix comparisons
keep nominal fake survival fixed and exclude this additional contribution.
Ten replicas limit every component covariance to rank nine. They cannot establish
partition-count stability or provide a replacement invertible production covariance.

## Covariance structure with fixed fake correction

The total statistical matrices are the sums of the independent data and
response-MC covariance estimates. Matching replica labels across these two
samples do not define joint fluctuations, so the replica vectors are not added.
The plots use the selected normalized GEN bins, ordered by increasing rho within
each pT block. Covariance panels share a symmetric logarithmic color scale;
correlation panels share the range [-1, 1].

| Channel | Grooming | Median error JK / analytic | Off-diagonal correlation RMS difference | Total rank analytic / JK |
|---|---|---:|---:|---:|
| Dijet | Groomed | 0.911 | 0.359 | 15 / 15 |
| Dijet | Ungroomed | 0.867 | 0.307 | 20 / 18 |
| Trijet | Groomed | 1.023 | 0.324 | 12 / 12 |
| Trijet | Ungroomed | 0.937 | 0.341 | 12 / 12 |

The diagonal error scales are similar, but the jackknife correlation matrices
are patchier, with larger typical correlations between pT blocks. Across these
four cases, median absolute cross-pT correlations are 0.106-0.188 analytically
and 0.266-0.340 with jackknife. Ten replicas provide only nine independent
contrasts per component; adding the independent data and response matrices
allows total rank up to eighteen. The table gives ranks in the selected window,
including the expected normalization null directions. The ungroomed dijet sum
still loses two additional directions relative to the analytic matrix.

This supports using the jackknife as an uncertainty-scale diagnostic, but does
not establish stable off-diagonal entries or justify replacing the analytic
covariance in chi-square calculations. Finite-replica noise and event-level
correlations absent from the analytic comparator cannot be separated by these
four matrices alone. The same-sample analytic data term still uses diagonal
input sumw2, so this is not a direct comparison with the production full-data
covariance.

## Verification and outputs

For all four configurations: exact MC self-closure, independent area-constrained
GLS reproduction of TUnfold central values and input covariance, agreement of
each data replica with the fixed linear map, positive-semidefinite replica
covariances, and the normalization null directions were checked. The covariance
from exact replica normalization differs from its linearized version by
0.07-0.47% in relative Frobenius norm. All 32 PNG panels were visually inspected;
the PDF has 32 pages. No engine, production loader or binning was edited.
The subsequent covariance comparison has four pages; all four matrix PNGs and
the standalone groomed-dijet correlation comparison were visually checked.

Final outputs are under `outputs/jackknife_run2_20260910/`:

- `input_audit/audit.json`: hash/axis/replica checks and label-aligned NPZs.
- `data_nominal_crosscheck.json`: per-era coverage and raw histogram comparisons.
- `comparison_v3/comparison.json`: final methodology and numerical summaries.
- `comparison_v3/*.npz`: full covariance matrices and normalized replicas.
- `comparison_v3/plots/comparison.pdf`: all data and MC comparisons.
- `comparison_v3/covariance_comparison/covariance_comparison.pdf`: four pages
  comparing total statistical covariance and correlation matrices with fixed fakes.
- `comparison_v3/covariance_comparison/matrix_metrics.json`: full and selected
  ranks, diagonal ratios and correlation differences, also split by component.
- `verification_receipt.json`: code, input-extraction and output hashes.

The initial `comparison/` stopped on an overstrict elementwise check of nonlinear
normalization. `comparison_v2/` used the old sample's data covariance as the
analytic comparator. Use **comparison_v3** for the corrected same-sample study.

Reproduce from the repository root after `source env.sh`, using fresh output paths:

```bash
python scripts/diagnostics/audit_hadronic_jackknife.py --input-root '/Users/aritra/cernbox/hadronic_jackknife_run2_20260908' --output outputs/jk_new/input_audit
python scripts/diagnostics/compare_hadronic_jackknife.py --audit outputs/jk_new/input_audit --reference-root outputs/_golden_new --output outputs/jk_new/comparison
python scripts/diagnostics/plot_hadronic_jackknife.py --comparison outputs/jk_new/comparison
python scripts/diagnostics/plot_hadronic_jackknife_covariance.py --comparison outputs/jk_new/comparison
```
