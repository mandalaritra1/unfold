# Handoff: LO vs NLO DY comparison (rho channel)

**From:** the producer side (`smp_jetmass_run2`). **For:** the unfold agent.
**Goal:** use the new **NLO (amcatnloFXFX)** DY pickles to answer the ARC's LO-vs-NLO
questions — AN **L237**, modelling comment **BC3 / M-our#3**, and **Figs 61–62**
("many bins agree with neither Pythia nor Herwig; give Chi²/ndof").

The producer's working hypothesis (stated to the ARC): *the LO→NLO change predominantly
affects the last rho bin `(-0.5, 0.0)`*. This study should confirm or refute that.

---

## 1. Inputs (stage in an explicit local directory)

Four per-era coffea pickles, `mode=minimal_rho`, `systematic_profile=no_syst`:

```
minimal_rho_nlo_2016.pkl
minimal_rho_nlo_2016APV.pkl
minimal_rho_nlo_2017.pkl
minimal_rho_nlo_2018.pkl
```

Produced by `smp_jetmass_run2/configs/zjet_nlo_all.json` (`dataset=nlo`, `era=all`,
`group_mode=per_group`). They hold the **same histogram set** as the LO
`minimal_rho_pythia_<era>.pkl` you already use (reco rho, gen rho, and
`response_matrix_rho_{u,g}`) — introspect one to confirm names.

The exploratory scripts preserve the local `~/Downloads/` location as a
development default, but take an explicit input path when run. Do not treat the
Downloads path as a portable provenance record; record the staged source and
its hash with any result.

### ⚠️ Gotcha 1 — dataset-axis key
Inside these pickles the **`dataset` axis value is `inclusive_<ERATAG>`**
(`inclusive_UL16NanoAODv9`, `inclusive_UL16NanoAODAPVv9`, `inclusive_UL17NanoAODv9`,
`inclusive_UL18NanoAODv9`) — **not** `pythia_UL...`. Wherever the input maker slices
the dataset axis by `pythia_UL<era>` (see `notebooks/input_maker_rho.ipynb` era→dataset
map), point it at `inclusive_UL<era>` for the NLO tag, or sum over the dataset axis.

### ⚠️ Gotcha 2 — normalization (do NOT re-normalize)
The NLO pickles are **already lumi-normalized** in the producer `postprocess`:
`xs·lumi·1000/sumw` with **xs = 6404 pb** (xsdb GenXSecAnalyzer for amcatnloFXFX),
**no k-factor**. The LO `pythia` pickles use HT-binned xs × **1.1297638966** k-factor.
→ Both are already on the same absolute footing; do not scale again. (If you want the
comparison at NNLO 6077 pb to match the LO k-factor convention exactly, multiply NLO by
`6077/6404 ≈ 0.949` — but for any shape/unfolding result it cancels.)

---

## 2. Suggested order of deliverables

**(a) Direct histogram comparison (light, do first)**
- Reco rho, **groomed & ungroomed**: LO vs NLO vs data, ratio panel.
- Gen rho: LO vs NLO — show where they diverge (expect mostly the last bin `(-0.5,0)`).
- Response matrix + **purity/stability**: LO vs NLO — does NLO change migrations?

**(b) NLO-response unfolding (heavier)**
- Build an `nlo` tag whose signal/response is the NLO sample, keep data/herwig/jk from
  `original` (so only the prior/response changes), and unfold.
- Compare the unfolded data from the **NLO** response vs the **LO** response →
  this is the prior/model dependence the ARC is implicitly probing.

**(c) Final overlay + Chi²/ndof (answers Figs 61–62)**
- Overlay unfolded data against **LO-Pythia, NLO, and Herwig** gen predictions.
- Quote **Chi²/ndof** of data vs each hypothesis (this is exactly the Fig 61/62 ask).

## 3. Wiring into the unfolder (guidance, not prescriptive)
The unfolder expects `inputs/zjet/rho/<tag>/` with files named `pythia_<era>.pkl`,
`data_all.pkl`, `herwig_all.pkl`, … (`scripts/staging/organize_inputs.py` renames producer
pickles and back-fills missing ones from a base tag via symlink). To treat NLO as the
signal:
1. Put the 4 NLO pickles in a raw folder.
2. Rename them to `pythia_<era>.pkl` inside a new tag `nlo` (so the unfolder uses them
   as the signal/response), back-filling data/herwig/jk from `original`.
   `organize_inputs.py` keeps the sample token, so a drop named `nlo_<era>.pkl` lands as
   `nlo_<era>.pkl` — you'll want them as `pythia_<era>.pkl`; rename accordingly.
3. Apply **Gotcha 1** (dataset-axis key) when the loader reads the histograms.
4. Run `scripts/run_unfolding.py --channel zjet --observable rho --tag nlo` and diff
   against `--tag original`.

## 4. Caveats
- **Stats:** inclusive amcatnloFXFX is **not** pT/HT-binned, so its high-pT tail is
  thinner than the HT-binned LO `pythia`. Watch statistics in the top pt and last rho
  bins (where the LO/NLO difference is expected). PtZ-binned NLO exists centrally if the
  tail is too thin — ask the producer to add it.
- **Observable:** `minimal_rho` gives **rho** only (+ rho response). The ARC's Figs
  32–33 are reco **mass** (groomed/ungroomed). If you need the mass comparison too, ask
  the producer for a `mode=minimal` (mass) NLO run — same config, `mode: minimal`.
- **No systematics:** nominal-only (`no_syst`). Fine for a shape/prior study.

## 5. Producer-side pointers
- Config: `smp_jetmass_run2/configs/zjet_nlo_all.json`
- XS branch (6404 pb): `smp_jetmass_run2/smp_jetmass_run2/zjet_processor.py` (`inclusive_` case)
- Sample lists: `smp_jetmass_run2/samples/zjet/mc/inclusive_UL*.txt`
