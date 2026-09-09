# Dijet and trijet unfolding on the pair split production, Run 2 combined

Aritra, short version of tonight's pass. The full story with every plot is in
the review deck:
`~/Projects/smp25010-docs/internal/2026-08-11-hadronic-rho-unfolding/`.
Per-bin tables live in `outputs/pairsplit_unfold/phase{A,B}_summary.md`.

## Verdict

The unfolding works on the new selection, combined into one Run 2 unfolding
(one response, one input, 137.6 fb). All 28 cross checks pass, none fail.
Keep the production configuration: tau=0, area constraint, full covariance,
coarse nested gen binning. One change is recommended: move trijet to the
coarser 2to1 gen merge, see below.

## Key numbers

- Inputs got better: dijet misses 35 to 16 percent, trijet 41 to 33. Every
  reported bin clears 0.5 purity and stability, in every era.
- Self closure exact, toy pulls 0.98 and 0.99 with proper coverage, zero
  negative bins, bottom line test passes both channels.
- Stat error on the normalized result: 1.3 percent median dijet, 4.5 trijet
  (trijet is statistics limited everywhere).
- Detector band (JER, JMS, JMR): 0.6 percent median dijet core, 0.3 trijet.
- FSR band, now propagated from the production PSWeights: 0.8 to 1.1 percent
  core median. ISR is 15 times smaller and stays out by policy.
- The large folded data/MC chi square is real physics, not detector effects
  and not one bad era: nuisances absorb under 9 percent of it, over 90
  percent (dijet) is common to all four eras, and most of the dijet part is
  the pT spectrum above 570 GeV, which per pT normalization removes. 2018 is
  the era that agrees best with Pythia, which flattered the old numbers.

## The two findings from your look at the plots

- Trijet 200 to 290 zigzag: made by the unfolding, not the data (reco is
  smooth at 0.94 and 1.02 where the unfold gives 0.74 and 1.24). It sits on
  the weakest cell of the binning (stability 0.46). The 2to1 merge removes it
  and is my recommendation for trijet.
- Normalization window: renormalizing over -1.8 to -0.55 flattens the dijet
  core to one (rms 12.3 to 3.2 percent at 290 to 400). The story becomes:
  core agrees, Pythia overshoots the soft shoulder by 10 to 25 percent. Do
  not include the catch all bin above -0.55 in the window. Variant figures
  are in the figure directory under variants.

## Still missing from the uncertainty

Vincia (expected dominant, 6 to 10 percent) and hadronization need the gen
space reweighting built; no pair split Herwig response exists; JES sources
sit unpropagated in the production files. Also: the per era data/MC
normalization spread (4.5 and 7.2 percent) exceeds the lumi uncertainty,
fine for shapes, must be understood before an absolute cross section.

## Where things are

Figures (all inspected): `outputs/figs/hadronic_rho_pairsplit_run2/`.
Results and NPZs: `outputs/pairsplit_unfold/`. New drivers untracked in
`scripts/studies/unfold/`; tests pass; nothing committed. Research notes and
ai-wiki are updated (topic: hadronic_rho_run2_pairsplit_unfolding).
