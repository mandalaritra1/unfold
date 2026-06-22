# Z+jet rho — statistical robustness of D'Agostini (2018, original tag)

Same channel-agnostic pipeline as dijet/trijet
(`scripts/dijet_methods/`, `outputs/dijet/2018/rho/method_study/README.md`),
applied to the Z+jet rho inputs (`inputs/zjet/rho/original/pythia_all.pkl` etc.).
This dir holds the **statistical** tests (toy self-closure + noise
amplification); the data overlay is intentionally NOT produced here (see caveat).

## Z+jet specifics
- **Lowest stats of the three channels** (gen ~4.6e5 vs trijet 3e7, dijet 5e9),
  but the binning is coarser (3 reported pT bins `[200,290,400,inf]`, gen rho 12
  bins), so per-bin counts -> stat spread ~2% (smaller than trijet's 4%).
- **Reco efficiency ~0.67** (lepton selection), lower than dijet/trijet's ~0.95;
  every reported bin still has eff >= 0.54 (well-conditioned). Adaptive merge:
  groomed gen low-edge -6, ungroomed -4.
- **87 systematics** available (incl. lepton reco/id/trig and JES variations) --
  richer than the 23 for dijet/trijet.

## Results (n_iter=4, merged binning)
Self-closure exact (2e-15). Toy self-closure with data-stat noise (500 toys):

| mode | bias (med / max bin) | stat spread (med / p90) |
|---|---|---|
| groomed   | ~0 / 0.8% | 1.9% / 5.4% |
| ungroomed | ~0 / 1.1% | 1.7% / 6.5% |

-> unbiased, controlled spread. Noise amplification (same toys, per method):

| mode | D'Agostini (med / tail) | inversion (Tik/SVD/near-inv) tail |
|---|---|---|
| groomed   | 1.8% / 5-20% | p90 ~84%, max ~200% |
| ungroomed | 1.7% / 7-20% | p90 870-1400%, max ~1700% |

Same conclusion as dijet/trijet: D'Agostini stays at the input-noise level and
unbiased; matrix inversion amplifies the same noise by 1-2 orders of magnitude in
the low-rho tail.

## Files
`toy_closure_{groomed,ungroomed}_{data,mc}.png`,
`noise_amplification_{groomed,ungroomed}.png`.
Run: `python -m scripts.dijet_methods.toy_closure zjet` and
`python -m scripts.dijet_methods.noise_amplification zjet`.

## Comparison to the locked-in TUnfold result
`compare_locked.py` runs D'Agostini on the *exact* locked-in reporting binning
(`bin_edges.gen_rho_edges_by_pt`: groomed 10 bins `[-10,-4.5,...,0]`, ungroomed
6 bins `[-10,-2.5,...,0]`) and overlays it with the committed TUnfold result
(`outputs/zjet/rho/original/unfold/unfolded_2d_<mode>.pkl`), per pT slice,
shape-normalized. Background (~1%) ignored. Findings (`compare_locked_*.png`):
- **Agreement in the well-measured bulk: ~5-9% median** (within uncertainties);
  PYTHIA-truth alignment check passes (my gen vs their `true_2dnorm`: groomed
  <=4%, ungroomed exact).
- D'Agostini is **smoother / more stable in the hard bins**: where the locked-in
  TUnfold has a near-unconstrained low-rho bin (huge error bar / near-zero value,
  e.g. ungroomed `[-10,-2.5]` at 290-400 GeV) or dips below PYTHIA at the highest
  rho, D'Agostini stays controlled and closer to the prior.
So D'Agostini reproduces the locked-in result where it is trustworthy and is
better-behaved where TUnfold struggles -- a mutual cross-check, not a conflict.

## Caveat -- background not subtracted in the data overlay
Z+jet data contains non-DY backgrounds (single top `st_all.pkl`, plus others).
Per direction these are ~1% and ignored in `compare_locked.py`; the agreement
with the locked-in TUnfold (which uses the full background-subtracted workflow)
confirms the ~1% background is negligible for the shape. For a final number a
proper subtraction is still the right thing. The toy/noise tests are MC-based and
unaffected regardless.
