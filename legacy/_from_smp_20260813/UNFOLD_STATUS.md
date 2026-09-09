# Dijet/trijet unfolding on the pair_split production — live status

Session started 2026-08-11. Goal: first full unfolding pass on the new
pair_split inputs (2018), all cross-checks, configuration comparison,
plot inspection, plain-language report at the end.

## Progress
- [x] Recon: locked recipe = coarse nested gen binning, tau=0 TUnfold,
      area constraint, groomed rho. Machinery: scripts/studies/unfold/.
- [x] AN-24-162 cross-check list extracted (18 items, all zjet-only so far).
- [x] Research-log survey: old-selection results + model-unc ingredient status.
- [x] New pair_split pickles verified compatible (keys, 77 systematics, axes).
- [x] Phase A DONE: combined Run 2 NPZs (bit-exact era sum), 22 files/channel.
      pair_split much healthier than old selection: dijet misses 35%->16%,
      every reported bin >0.5 purity AND stability. Self-closure exact, 0
      negative bins, amp median 1.6 (dijet coarse candidate) / 3.2 (adopted).
      Flag: folded data/MC chi2 large (dijet ~4825/9 adopted row) = data vs
      Pythia shape, not a construction bug. Correction applied: adopted row =
      proposal_coarse_tail (10/6 reported bins per pt), NOT proposal_2to1.
- [x] Phase B DONE: 28 checks PASS, 0 FAIL, 8 UNAVAILABLE (no pair_split
      Herwig etc). Pulls/coverage in tolerance; bottom-line holds both channels;
      production config (tau=0 + area + full cov) beats all challengers.
      FSR band ~0.8-1.1% median (the usable model leg); ISR ~15x smaller.
      Detector nuisances absorb <9% of the folded data/MC chi2 (JMS/JMR pin
      at bounds) -> genuine data-vs-Pythia shape. Per-era chi2 decomposition DONE:
      no bad era — 2018 is anomalously GOOD (trijet chi2@common-yield 18 vs
      63-97 for others); 93%/79% of mismatch common to all eras; dijet chi2
      dominated by pT>570 spectrum (removed by per-pT normalization); per-era
      data/MC norm spread 4.5-7.2% > lumi unc (prescale bookkeeping, fine for
      shapes, blocks absolute xsec).
- [x] Phase B2: 37 figures produced; inspector flagged a shared-helper label
      collision (24 figs) + amplification legend placement (2 figs) -> fix
      cycle running; 11 figures clean on first pass
- [x] Figures: 37/37 pass inspection after two fix cycles (label spacing in
      24 figs; amplification top-margin units bug, root-caused)
- [x] Knowledge bases logged: research-notes topic
      hadronic_rho_run2_pairsplit_unfolding (+ plots), old fullstat note
      superseded, trijet per-era caveat resolved, model plan FSR leg done,
      zjet dashboard updated; ai-wiki pipeline topic + variation-NPZ bug page,
      index/log updated
- [x] REPORT DELIVERED: UNFOLDING_REPORT_2026-08-11.md (repo root)

- [x] Aritra's plot findings: trijet 200-290 oscillation confirmed
      unfolding-made (2to1 merge recommended); [-1.8,-0.55] normalization
      window validated; variant figures rendered; report + notes corrected
- [x] 29-slide Typst review deck built and inspected:
      smp25010-docs/internal/2026-08-11-hadronic-rho-unfolding/
- [x] Report trimmed to minimal per Aritra

- [x] Appendix plot set: 90 new figures (closure, per-variation closure, signed
      syst shifts, pulls/coverage per bin, config per bin, Bayes iterations,
      refold, per-era reco/purity, ladder), inspector-clean; per-bin pull/coverage
      tails found and documented
- [x] Exhaustive appendix deck DONE: 44 pages, all 138 figures each exactly
      once (machine-checked), hadronic-rho-unfolding-appendix.pdf

- [x] JES propagation DONE (0.470%/0.356% core median, FlavorQCD dominant; JES_RelativeJER* substring bug found+chipped) (Aritra request): all 27 sources through the
      Run 2 unfolding with the zjet year-correlation prescription (rho per
      source, 3 era groups, 4 legs each); corrected sqrt(rho) split as default
      + legacy linear for comparison; JER decorrelated variant too. Found and
      chipped: zjet rho=0.5 split under-covers (linear vs sqrt), AN table
      AbsoluteScale/AbsoluteStat swap, JER text/code contradiction.

## SESSION COMPLETE (previous deliverables)
- [x] Phase C: per-era purity/stability DONE — adopted binnings survive all
      eras (dijet clean; trijet min purity 0.501-0.506, margin ~0, era-independent;
      results in outputs/pairsplit_unfold/era_check/)
- [ ] Report
