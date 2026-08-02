# Which rho binning is in these files

**2026-07-25**: `minimal_rho_dijet_{data,mg_pythia8}_2018.pkl` were regenerated
at full statistics on the new hadronic groomed-rho axis
(`smp_jetmass_run2/hist_utils.py::_had_rho_gen_g`, 14 gen / 28 reco):

```
gen   -10 -5 -4 -3.4 -2.85 -2.25 -1.8 -1.5 -1.3 -1.1 -0.9 -0.75 -0.65 -0.55 0
reco  the exact 2:1 refinement of that
pt    185 200 290 400 480 570 680 760 820 13000   (unchanged)
```

The reported binning is a *nested merge* of these edges, applied at unfold time
(dijet: two lowest gen pairs merged -> 10 bins over pT 200/290/400/480/570/inf).

**`minimal_rho_dijet_herwig_2018.pkl` (Jun 8) is still on the OLD axis**
(`-10 -8 -7 -6 -5 -4.75 ... -1.5 -1 0`) and will NOT rebin onto the new one --
none of -3.4, -2.85, -1.8, -1.3, -1.1, -0.9, -0.75, -0.65, -0.55 exists on it.
Regenerate it before using herwig with the new inputs.

`prebinning_20260725/` holds the previous data/mg_pythia8 files on the old axis.
