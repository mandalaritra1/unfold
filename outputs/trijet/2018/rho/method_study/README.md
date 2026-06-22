# Trijet rho unfolding — numerical method study (2018)

Same D'Agostini iterative-Bayes pipeline as the dijet study
(`outputs/dijet/2018/rho/method_study/README.md`), applied to the 2018 trijet
rho inputs. **It works the same way** — smooth, positive, tracks PYTHIA8 gen
within uncertainty, exact self-closure — for both groomed and ungroomed.

## Trijet vs dijet differences
- **Much lower stats:** ~3e7 events vs ~5e9 for dijet (~150x fewer) -> larger
  statistical error bars, especially in the low-rho tail. Data is consistent
  with PYTHIA at ~1 sigma almost everywhere.
- **No giant miss sink:** trijet gen/reco are comparable (eff ~0.95 overall),
  unlike dijet's `pT<200` sink. Instead trijet has a **~18% fake rate**
  (groomed) handled by the MC fake-fraction subtraction.
- **No HERWIG sample** for trijet -> the HERWIG-as-data model-bias test is not
  available (only self-closure + reweighted-MC prior-independence).
- **Adaptive low-rho binning** picks a different boundary per mode: groomed gen
  low-edge -6 (worst reported eff 0.68), ungroomed -3 (worst eff 0.75).

## Validation (groomed, n_iter=4)
- Self-closure: exact, 2e-15 at every iteration.
- Reweighted-MC (+-15% rho tilt, nominal prior): recovered to within ~5-7%.
- Response systematics: same sources as dijet (scale/PS-dominated; JES absent).

## Files / reproduce
`overlay_*`, `closure_*`, `methods_*` (groomed/ungroomed x native/merged).
```bash
source .venv/bin/activate
python -m scripts.dijet_methods.plots trijet     # trijet only
python -m scripts.dijet_methods.plots            # both channels
```
Same code as dijet: `scripts/dijet_methods/`. The pipeline is channel-agnostic
(`prepare(mode, binning, channel)`, `channel_paths(channel)`); HERWIG is optional.

## Caveats
- Low trijet stats make the tail bins genuinely uncertain ("bad bins" expected
  to scatter; that is fine).
- JES absent from response variations (syst budget incomplete), as for dijet.
