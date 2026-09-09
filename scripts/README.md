# scripts

Everything here reads finished runs under `outputs/` or the input pickles;
the unfolding itself is the `unfold` command (see the top-level README).

| script | what it makes |
|---|---|
| `pairsplit_plot_book.py` | the dijet / trijet plot book PDF from the four `original` run manifests |
| `pairsplit_slide_deck.py` | slide deck built from the plot book and the combined figures |
| `combined_channels_rho.py` | Z+jet / dijet / trijet overlays from the saved artifacts |
| `datamc/` | reco-level data vs MC validation figures for the AN (Z+jet and pair-split), theory overlays, per-systematic reco variations |
| `hepdata/export_zjet.py` | reruns the zjet unfolding with all systematics and writes the HEPData intermediate npz |
| `hepdata/build_submission.py` | assembles the HEPData YAML submission from that npz |

Run them from the repository root with the venv active.  `scripts/datamc`
reads `inputs/zjet/validation/` and the arc_r2 / jmsjmr_unity pickles; the
per-era luminosities come from `unfold.systematics.RUN2_LUMI`.
