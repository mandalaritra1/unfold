"""Run the Z+jet rho unfolding (full systematics) and export the HEPData intermediate .npz.

Usage (ROOT on the path, i.e. `source setup_root.sh` first):

    .venv/bin/python scripts/hepdata/export_zjet.py [--tag original] [--out DIR]
"""
import argparse
import time

from unfold.binning import ZJET_BINNINGS
from unfold.config import TAGS, get_tag
from unfold.engine import Unfolder
from unfold.hepdata import export_all
from unfold.zjet_inputs import load_zjet_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="original", choices=list(TAGS[("zjet", "rho")]))
    ap.add_argument("--out", default="outputs/zjet/rho/hepdata")
    args = ap.parse_args()

    spec = get_tag("zjet", "rho", args.tag)
    unfolders = {}
    for mode, groomed in [("ungroomed", False), ("groomed", True)]:
        t0 = time.time()
        print(f"\n===== Building {args.tag} / {mode} (all systematics) =====", flush=True)
        binning = ZJET_BINNINGS[spec.binning_name(groomed)]
        inputs = load_zjet_inputs(spec, groomed, binning, do_syst=True)
        unfolders[mode] = Unfolder(inputs, spec, groomed).run()
        print(f"  done in {time.time() - t0:.1f}s, {len(unfolders[mode].systematics)} systematics", flush=True)

    info = export_all(unfolders, out_dir=args.out)
    print("\n===== Exported =====", flush=True)
    for mode, v in info.items():
        nb = [b["n_bins"] for b in v["manifest"]["published_pt_bins"]]
        print(f"  {mode}: {v['npz']}  published bins={nb}", flush=True)


if __name__ == "__main__":
    main()
