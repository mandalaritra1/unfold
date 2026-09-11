# Dijet/trijet review tracker

Started 2026-09-05. Z+jet is frozen. Findings and evidence are in
`scratchpad/review_2026-09-05/REVIEW.md` in this checkout.

Status distinguishes a code fix from validation of existing production. Closing
a code defect does not certify old inputs or authorize a production rerun.

| ID | Issue | Status | Next action |
|---|---|---|---|
| R1 | Chunk-dependent gen/reco marginals and skipped systematic variations | Closed by user on 2026-09-08; fix merged into local dev at d7c53c8 | User decided no production rerun is needed; no further impact study requested |
| R2 | PDF members collapsed before histogramming | Deferred by user on 2026-09-05; diagnostic saved | Revisit coherent Hessian propagation later; no implementation or production now |
| R3 | 2017 trigger threshold: code 310 vs AN 359 GeV | User regards AN as outdated; moved on | Keep code at 310; efficiency provenance remains unverified |
| R4 | AN systematic-closure success claim contradicted by tables | User accepts model-case exception; leave unchanged | No edit; model reco limitations cited by user, not independently verified |
| R5 | Large dijet refolding residuals and statistical interpretation | Closed by user review on 2026-09-05 | No further investigation or analysis change requested |
| R6 | Displayed model envelope differs from covariance | Closed; prescription accepted by user on 2026-09-05 | Two enclosing-template group covariances; see hadronic_model_covariance.md |
| R7 | Dijet model/producer fiducials differ | Closed by user review on 2026-09-05; accepted limitation | No further investigation or selection change; numerical effect not quantified |
| R8 | JES exclusions and model fake treatment | Fake assumption accepted; JES restoration conditional on valid inputs | Retain defective categories exclusion; restore only verified genuine JES variations |
| R9 | Prediction statistics omit normalization correlations | Fixed and validated on 2026-09-08 | Jacobian propagation; see hadronic_prediction_statistics.md |
| R10 | Additional AN physics definitions and claims | Closed on 2026-09-08 | Mass formula, recoil interpretation and bottom-line wording corrected; PDF rebuilt and inspected |

## R1: Chunk-dependent histogram filling

- Authorization: user requested the fix in `smp_jetmass_run2` and a separate
  GPT-6 Astra task with medium reasoning on 2026-09-05.
- Scope: dijet/trijet only. Preserve existing selections, weights, axes and
  unrelated changes. Restore contributions lost by premature chunk returns.
- Task: `01a0704e-398f-75b2-adad-64985fe66f33`, GPT-6 Astra, medium reasoning.
- Checkout: `/Users/aritra/.codex/worktrees/9b20/smp_jetmass_run2`, branch
  `codex/fix-hadronic-chunk-partition`, based on reviewed commit `82d5df2`.
  The initial worktree was at older `64d44e1`; it was clean and was moved to
  the reviewed committed base. Saved-project uncommitted work is untouched.
- Implementation: complete and validated. User authorized integration on
  2026-09-05; committed as `d7c53c8` and fast-forward merged into the saved
  checkout on local `dev`. Not pushed. Changed files:
  `smp_jetmass_run2/dijet_processor.py`, `smp_jetmass_run2/trijet_processor.py`,
  `smp_jetmass_run2/hadronic_base.py`,
  `tests/test_hadronic_chunk_partition.py`.
  Removed premature MET/gen/reco/matching/final-selection returns. Inclusive
  gen and reco retain their independent masks; response arrays retain the
  original matched `final_seq`. Empty arrays contribute no entries and later
  jet/weight variations run. This restores miss/fake marginal contributions
  formerly lost in sparse chunks, plus previously skipped cutflow/skim entries.
  A hadronic-only helper supplies typed empty JER/JES records because Coffea's
  correction factory cannot build an entirely empty jet collection. Populated
  jets retain the existing correction path. Dijet singleton axes replace
  `ak.unflatten(array, 1)`, which crashes on zero-length selections. No cuts,
  weights, matching definitions, axes, or Z+jet code changed.
- Validation: GluonJetMass venv, Coffea 2026.5.0, from this worktree.
  `PYTHONPATH=. /Users/aritra/Projects/GluonJetMass/.venv/bin/python tests/test_hadronic_chunk_partition.py`
  reports `Ran 6 tests in 9.437s`, `OK`. Covers both channels, mass/rho,
  gen-only/reco-only/matched/neither-selected events, physically empty jet
  collections, and all-MET-fail chunks. Combined vs singleton partitions agree
  binwise in contents and sumw2 including flows for nominal, puUp, and later
  JMSUp; independently expected yields/variances and nonzero JMS bin migration
  are checked. The reviewed original source fails with 96 assertion failures
  and 6 missing-category errors. These deterministic-provider tests validate
  control flow, not calibration correctness or stochastic JER reproducibility.
- Required processor checks: repository verify-processors cutflow script
  copied to `/tmp/r1-validation/cutflow_regression.py` with only its hardcoded
  SMP import path changed to this worktree. Run with the same venv and
  `--proc dijet`, `--proc trijet`, and each with `--data`, default 2 x 50k.
  Dijet MC: `PASS: all 36 common cutflow steps match.` Dijet data:
  `PASS: all 49 common cutflow steps match.` Trijet data:
  `PASS: all 31 common cutflow steps match.` Trijet MC exits 1 with exactly
  the runbook's two permitted JMR exceptions: `recoRap2p5` and `recoRap_seq`,
  each GluonJetMass 456 vs smp 458. Every other common step matches.
  `PYTHONPATH=. /Users/aritra/Projects/GluonJetMass/.venv/bin/python tests/test_hist_parity.py`
  reports `OK: histogram registration is in parity across channels (7 mode comparisons, do_gen both ways).`
- Additional validation: `/tmp/r1-validation/populated_parity.py` compares
  reviewed original and patched process() on the same 5,000 real NanoAOD
  events, nominal/JMSUp/puUp, minimal and minimal_rho. Exact histogram contents,
  sumw2, axes, cutflows and sumw agree: dijet 16/23 histograms, trijet 11/15.
  `/tmp/r1-validation/real_empty.py` removes FatJet/SubJet collections from
  5,000 real NanoAOD events and passes for both channels with nominal, JERUp,
  JES_AbsoluteScaleDown, JMSUp, JMRDown and puUp. Gen truth is retained in all
  six categories; reco/response are zero. Log files and scripts are under
  `/tmp/r1-validation/`; original-source failure log is
  `/tmp/hadronic-partition-baseline.log`. `git diff --check` passes.
- Blockers: none for the code fix. Historical yield/uncertainty impact cannot
  be measured from the available aggregate ledger; per-chunk evidence or an
  explicitly scoped reproduction is needed before choosing a rerun scope.
- Existing production impact: local campaign ledger records deployed tree
  `7372447c66c125638247b4cd57cd43ffa55ab5ec`, which contains R1 returns.
  Its 2018 dijet/trijet all-systematic configs use 50,000-event chunks.
  Historical source exposure is supported; frequency and lost contributions
  remain unquantified. Aggregate 77/77 populated systematic categories cannot
  determine per-chunk loss. No evidence yet of material bias. Typical user
  chunks are 25k–400k; rare/remainder chunks still need assessment.
- Rerun decision, 2026-09-08: user decided a production rerun is not needed
  and requested moving on. R1 is closed for this review; no further historical
  impact study is requested. The numerical impact remains unquantified.

The fix task owns this R1 section and its table status. Update them with actual
checkout/branch, changed files, validation commands/results, and any blocker.
Keep production impact distinct from implementation completion. Other issues
remain owned by the review discussion task. Read before editing and preserve
concurrent updates; do not replace the whole tracker.

## Activity

- 2026-09-05: Initial review discussed. R1 fix authorized; R2 is next for discussion.

## R2: PDF member propagation, source verification

Status: deferred by user on 2026-09-05. Prescription and bounded diagnostic
are recorded; no PDF implementation or production change authorized.

- Current `GetPDFweights` reduces member weights per event. This loses coherent
  inter-event member information. It is not necessarily a pure rate variation,
  but can erase shape effects and cannot reproduce general PDF covariance.
- Correct accumulation order: keep each member coherent across events and
  merge chunks/samples with matched PDF member IDs before reduction. For the
  measurement uncertainty, vary MC response and corresponding gen/reco/fake/miss
  inputs, unfold the same observed data, normalize every member result using
  the established per-pT window, then construct the set-appropriate covariance.
  Keep PDF effects on theory predictions distinct from response dependence.
- Replica sets use ensemble spread/covariance; symmetric Hessian sets use
  sums of signed-direction outer products without division by member count.
  Paired Hessian sets have a different direction construction. Central members,
  confidence levels and alpha_s members must be identified from metadata.
- Public CMS DAS 2026 tutorial says most UL samples use Hessian NNPDF3.1.
  This is not verification of the actual QCD sample members. Do not assume
  replica format from the NNPDF name or hardcode a universal index mapping.
- Next concrete check: inspect `LHEPdfWeight` branch documentation and LHE/run
  member IDs for representative actual campaign files, including all eras and
  dataset families; verify central-weight normalization and error-set metadata.
  Existing pdfUp/pdfDown-only histograms cannot recover lost member information.

Sources inspected 2026-09-05:

- [CMS DAS generator guidance](https://fnallpc.github.io/generators/aio/index.html#estimating-systematic-uncertainties)
- [PDF4LHC Run II, section 6.2, equations 19–28](https://arxiv.org/pdf/1510.03865#page=49)
- [LHAPDF PDFSet uncertainty, correlation and ErrorType documentation](https://www.lhapdf.org/classLHAPDF_1_1PDFSet.html)
- [CMS HWW VBF 2018 member configuration](https://github.com/latinos/PlotsConfigurationsRun3/blob/1c1f0406e05435452117c3dbcee666df0b181a7e/VBF_differential/Full2018_v9/nuisances.py#L488-L510)
- [mkShapesRDF member weighting](https://github.com/latinos/mkShapesRDF/blob/3cebbe4397981e406789d79ed44cda72fc5fbd6a/mkShapesRDF/shapeAnalysis/runner.py#L597-L670)
- [Histogram-level reduction](https://github.com/latinos/mkShapesRDF/blob/3cebbe4397981e406789d79ed44cda72fc5fbd6a/mkShapesRDF/shapeAnalysis/histo_utils.py#L281-L319)
- [CMS WW Run3 separate Hessian-direction nuisances](https://github.com/latinos/PlotsConfigurationsRun3/blob/1c1f0406e05435452117c3dbcee666df0b181a7e/WW_Run3/nuisances_ALL.py#L289-L313)

Code-example caveats: the VBF framework RMS is centered on nominal with 1/N
and compresses to Up/Down, so it demonstrates correct member accumulation,
not a complete prescription for normalized unfolding. The Run3 example retains
separate directions; its process decorrelation and fallback choices are not
general CMS rules. Code was inspected, not independently validated against its
production datasets or publication configuration.

### Local NanoAOD inspection, 2026-09-05

Inspected 25 local NanoAOD files under Projects/rootfiles, GluonJetMass/test_files,
and the CERNBox root directory. The additional Research-Data download was also
checked and is data. Searches covered project, download, document and CERNBox
ROOT filenames; this does not establish that no other local cache exists.
Detailed file paths, event counts, branch titles and sampled weight checks are
saved in `scratchpad/review_2026-09-05/local_pdf_metadata.json`.

- The staged Run-2 MC is UL18 `QCD_Pt_170to300_TuneCP5_13TeV_pythia8`,
  198000 events, with no Events/LHEPdfWeight or LHEScaleWeight branches.
  It is not the MadGraph-MLM HT-binned sample used by the unfolding.
- The local Run3 QCD PT-binned files also lack LHEPdfWeight.
- Nine local Run3 ttbar/Zprime files carry LHA IDs 325300–325402. In the
  first 1000 events per file, every event has 103 finite weights and member
  zero equals 1. These are useful implementation fixtures but cannot establish
  the Run-2 QCD campaign's PDF prescription.
- The authoritative LHAPDF info file identifies 325300 as
  `NNPDF31_nnlo_as_0118_mc_hessian_pdfas`, ErrorType `symmhessian+as`:
  member 0 central; 1–100 eigenvectors; 101/102 alpha_s=0.116/0.120.
  Source: https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_as_0118_mc_hessian_pdfas/NNPDF31_nnlo_as_0118_mc_hessian_pdfas.info
- No matching Run-2 QCD_HT MadGraph-MLM NanoAOD was found in the searched
  local locations. The campaign fileset is
  `/Users/aritra/Projects/smp_jetmass_run2/samples/hadronic/fileset_MG_pythia8.json`.
  Actual campaign member mapping remains unresolved until a matching file is
  located or read remotely. No PDF code, production or uncertainty changed.
- CERNBox recheck, following symlinks: 6391 ROOT files enumerated; no basename
  matches the MadGraph QCD campaign fileset. The two top-level UUID-named files
  are data without generator/PDF weights; other ROOT groups are standalone
  generator products and analysis outputs. Hadronic skim directories contain
  derived pickle files. Next step is authenticated read-only inspection of
  original campaign NanoAOD after the user enables the VOMS proxy.

### Authenticated campaign inspection, 2026-09-05

The user enabled the VOMS proxy and authorized reading actual analysis files.
Read one QCD_HT300to500 MadGraph-MLM file from each of 2016APV, 2016,
2017 and 2018 using the repository fileset. Remote byte-range reads only;
no full-file downloads or production runs. Proxy validity was confirmed.

All four Events/LHEPdfWeight branch titles identify LHA IDs 325300–325402.
Each of the first 100 events per file contains 103 finite weights with member
zero exactly 1. This confirms the same symmetric-Hessian-plus-alpha_s set
identified in the authoritative LHAPDF metadata above, now for actual Run-2
QCD campaign files rather than unrelated local Run3 fixtures.

Member mapping for the checked files:

- 0: central.
- 1–100: symmetric Hessian directions, combined by quadrature of observable
  shifts after coherent member accumulation and normalization, without a 1/N
  factor.
- 101: alpha_s=0.116; 102: alpha_s=0.120. Treat separately from PDF directions;
  document any rescaling to the chosen alpha_s uncertainty interval.

The current helper includes all 103 members in an event-level std/mean, so
three problems are confirmed for these files: lost member coherence, the
wrong Hessian reduction, and mixing alpha_s members into the PDF spread.
No change to PDF processing or AN text was made. Checks cover one HT bin
in four eras, not every file/HT bin; future preparation should validate each
input's metadata rather than assume a universal mapping.

Evidence: `scratchpad/review_2026-09-05/campaign_pdf_metadata_ht300to500.json`.
Reproducer: `scratchpad/review_2026-09-05/inspect_campaign_pdf.py`, run using
the GluonJetMass venv with uproot/XRootD and `--output <review-json-path>`.

### Bounded gen-level size and memory diagnostic, 2026-09-05

- User asked whether PDF uncertainty is small relative to roughly 10% total
  uncertainty and whether 100 member histograms would cost too much memory.
- Read only gen/PDF branches from one 2018 QCD MadGraph file per HT bin,
  at most 50k events/file, totaling 358720. All eight files confirm 103 finite
  weights with LHA IDs 325300-325402 and central member one. No production
  code, unfolding or uncertainty prescription changed.
- Coherent symmetric-Hessian members, genWeight, cross-section/sample-sumw HT
  combination, hadronic truth cuts, and canonical per-pT normalization give
  reported-bin PDF ranges: dijet groomed 0.089-0.599%, dijet ungroomed
  0.030-1.041%, trijet groomed 0.276-1.931%, trijet ungroomed 0.190-4.562%.
  These are preliminary gen-shape estimates. Half-sample differences are
  appreciable, especially trijet; the 4.56% bin has about 19 effective entries.
  This does not establish negligible response dependence or a Run-2 result.
- Dense weighted histogram payload per dataset/channel for 103 members:
  gen alone, both modes, 0.605 MiB; fine response+reco+gen, both modes,
  213.92 MiB. Ten-member blocks reduce the latter to 20.77 MiB. Counts exclude
  temporary arrays, metadata and worker/dataset multiplication. Dedicated PDF
  accumulation need not duplicate detector systematic or diagnostic histograms.
- Reproducer and detailed findings:
  `scratchpad/review_2026-09-05/gen_pdf_diagnostic.py`,
  `gen_pdf_diagnostic.json`, `gen_pdf_diagnostic_arrays.npz`,
  `gen_pdf_diagnostic.log`, and `gen_pdf_diagnostic.md` in that directory.
- R2 remains open. No full PDF-production implementation authorized yet.

### Integration and review decisions, 2026-09-05

- User authorized merging R1 into dev and deferred R2. Local dev now contains
  `d7c53c8` via fast-forward. Focused partition tests immediately before commit:
  `Ran 6 tests in 10.031s`, `OK`; prior cutflow/parity and real-event validation
  above remain applicable. All 23 pre-existing changed files retain identical
  SHA256 contents after integration. Fix paths are clean. No push or rerun.
- R3 remains a confirmed code/note discrepancy: 2017 AK8PFJet200 stitching
  starts above 310 GeV of corrected leading-jet pT, but AN-25-039
  EventReconstruction.tex:135 lists 359 GeV for full efficiency. Recover the
  underlying efficiency measurement before deciding which value needs correction.

- R3 GluonJetMass cross-check: local `coffea-2025-migrate` at `edb9f68`
  also uses 310 GeV for 2017 AK8PFJet200 (`python/utils.py:38`), with the
  next threshold 450 GeV. Default `applyPrescales` at line 499 uses this table;
  active dijet/trijet data calls at lines 374/415 do not override it.
  `git blame` attributes the table to `00ddc578`, 2025-03-28. The separate
  `turnOnPts_JetHT_old` has 340 GeV, not 359, and is not the default.
  Thus both current local implementations agree at 310. Need colleague's
  efficiency measurement/provenance to resolve the AN discrepancy. No contact
  or threshold changes made.

- Review decision: user regards the R3 AN table as outdated because both local
  codebases agree and requested moving on to R4. No trigger or AN edits made;
  agreement between implementations does not independently verify efficiency.

- User decision on R4: leave model-uncertainty closure cases and current text
  unchanged because these are special cases without proper reco, as stated by
  the user. This records the accepted exception, not independent validation of
  model coverage or of the universal AN wording. Move discussion to R5.

- R5 closed by user review on 2026-09-05 after inspecting AN Figures 47/48.
  User considers the near-unity refolding ratios satisfactory and the issue
  non-concerning. No further investigation, code change, or AN edit requested.
  Closure records the review decision; the previously quoted measured-covariance
  statistics remain uncalibrated diagnostics, not validated goodness-of-fit tests.

## R7: Model/producer fiducial difference — accepted limitation

- User decision on 2026-09-05: the expected effect is too small to warrant
  further investigation. Close this review item without a diagnostic, selection
  change, or new production.
- The 185 versus 200 GeV dijet pair-floor mismatch remains. Its numerical
  impact was not quantified; closure records the user's assessment and scope
  decision, not a measured demonstration of negligible impact.

## R8: JES exclusion cross-check and model fake treatment

- Read-only cross-check on 2026-09-05 of canonical 2018 dijet groomed reco
  input: all six RelativeJER Up/Down histograms are populated and differ from
  nominal. EC1/EC2/HF are exactly identical to one another per direction and
  nearly identical to ordinary JER (six bins differ; aggregate absolute
  difference relative to absolute JER yield about 2.19e-6). This supports the
  documented stale dispatch-bug diagnosis, not a zero-histogram explanation.
  Other eras/channels were not freshly checked. Exclusions remain unchanged.
- Constructed Vincia/CR/fragmentation legs hold absolute unmatched reco yields
  fixed while reweighting GEN and matched response. Their fake fractions and
  hence data correction can change. The limitation is absence of a direct
  alternative-model prediction for the unmatched yield/shape, not absence
  of all fake-related uncertainty. Directly produced detector/weight
  variations use their own reco and response arrays. No additional nuisance
  or production requested; numerical impact is not established.

- User decision: accept unchanged absolute fakes under the constructed model
  variations, based on their small contribution; no additional fake nuisance
  or diagnostic requested. The smallness was not newly quantified here.
- User authorizes restoring the JES categories if they are genuine variations.
  The checked stored categories fail that condition: nonzero content alone
  does not establish a genuine JES source, and these reproduce the documented
  defective JER-like behavior. Keep exclusions until valid replacement inputs
  are verified. No production rerun was authorized by this conditional request.

## R10: Note wording

- On 2026-09-08 the user authorized replacing the exact recoil-bound claim
  with an approximate 200 GeV recoil scale for approximately balanced jets.
  Updated EventReconstruction.tex in AN-25-039. Selection unchanged.
- Built a separate full PDF and visually checked the edited paragraph on
  PDF page 19. Saved review copy: scratchpad/review_2026-09-05/AN-25-039_recoil_wording.pdf.
- User authorized the jet-mass notation correction on 2026-09-08.
  Equation 5 now squares the sum of constituent four-momenta, rather than
  summing their squares. Rebuilt the 115-page AN and inspected PDF pages
  14 and 15; the equation and following page are clean. Saved the verified
  PDF beside the AN sources. Documentation only; no analysis results changed.
  R10 is closed; the numerical R5 investigation remains closed.

- User authorized moving the bottom-line probability criterion and numerical
  table into the AN and removing diagnostic status labels from plots.
  Updated AN-25-039.tex and its bibliography; added 20 canonical comparisons,
  existing covariance/dof methodology, and the CMS source. Removed the
  overclaim that the test establishes a unique physical origin for residuals.
  All eight diagnostic plots now omit probability/status annotations.
  Rebuilt and inspected PDF pages 73 and 78, with no new overflows in the
  edited passage. The updated PDF is beside the AN sources; prior PDF backed
  up in scratchpad/review_2026-09-08_bottomline/. No unfolding rerun.
