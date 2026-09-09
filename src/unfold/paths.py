"""Locations outside the repository, overridable through environment variables.

The defaults are the paths on the machine the analysis was developed on.
Set the variables to run elsewhere:

  UNFOLD_CERNBOX            CERNBox sync directory holding the pair-split skims
  UNFOLD_SMP_ROOT           checkout of smp_jetmass_run2 (generator campaign results)
  UNFOLD_PAIRSPLIT_INPUTS   <era>/<channel>_{mc,data}/ pair-split pickles
  UNFOLD_ROOUNFOLD_LIB      libRooUnfold, only for --method roounfold_bayes
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUTS = REPO_ROOT / "inputs"
OUTPUTS = REPO_ROOT / "outputs"

CERNBOX = Path(os.environ.get("UNFOLD_CERNBOX", "~/cernbox (2)")).expanduser()
SMP_ROOT = Path(os.environ.get("UNFOLD_SMP_ROOT", "~/Projects/smp_jetmass_run2")).expanduser()
PAIRSPLIT_INPUTS = Path(os.environ.get(
    "UNFOLD_PAIRSPLIT_INPUTS", CERNBOX / "hadronic_minimal_rho_pairsplit_aligned")).expanduser()

# standalone-generator campaigns used by the pair-split modelling uncertainty
MESS_CAMPAIGN = "mess_pairsplit_dijet_20260813_v1"
MESS_RESULTS = SMP_ROOT / "rivet/hadronic_prod/results" / MESS_CAMPAIGN
MESS_CAMPAIGN_DIR = CERNBOX / "hadronic_model_prod_mess" / MESS_CAMPAIGN
INTERNAL_CAMPAIGN = "mgmlm_internal_trijet_20260802_v1"
INTERNAL_RESULTS = SMP_ROOT / "rivet/hadronic_prod/results" / INTERNAL_CAMPAIGN
INTERNAL_CAMPAIGN_DIR = CERNBOX / "hadronic_model_prod_internal" / INTERNAL_CAMPAIGN
