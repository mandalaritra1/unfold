"""Run the rho unfolder (full systematics) and export the HEPData intermediate .npz.

Usage (ROOT must be on PYTHONPATH, i.e. `source scripts/setup_root.sh` first):

    .venv/bin/python scripts/run_hepdata_export.py [--spec fixed_jec|original] [--out DIR]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from unfold.tools.unfolder_core import Unfolder, RHO_SPECS  # noqa: E402
from unfold.tools.hepdata_export import export_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="fixed_jec", choices=list(RHO_SPECS))
    ap.add_argument("--out", default="outputs/rho/hepdata")
    args = ap.parse_args()

    spec = RHO_SPECS[args.spec]
    unfolders = {}
    for mode, groomed in [("ungroomed", False), ("groomed", True)]:
        t0 = time.time()
        print(f"\n===== Building {args.spec} / {mode} (do_syst=True) =====", flush=True)
        unfolders[mode] = Unfolder(spec, groomed=groomed, do_syst=True)
        print(f"  done in {time.time() - t0:.1f}s, "
              f"{len(unfolders[mode].systematics)} systematics", flush=True)

    info = export_all(unfolders, out_dir=args.out)
    print("\n===== Exported =====", flush=True)
    for mode, v in info.items():
        nb = [b["n_bins"] for b in v["manifest"]["published_pt_bins"]]
        print(f"  {mode}: {v['npz']}  published bins={nb}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
