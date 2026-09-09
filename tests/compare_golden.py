"""Compare the data products of two output trees, key by key.

    python tests/compare_golden.py <golden_dir> <new_dir> [--rtol 1e-9] [--atol 0]

Walks every ``.npz`` and ``.pkl`` under ``golden_dir`` (skipping ``_previews``),
finds the file with the same relative path under ``new_dir`` and compares
every array (and every ``hist`` object's values and variances).  Prints a
line per key with the maximum absolute and relative difference, and exits
non-zero if anything differs beyond the tolerance.  Also compares the numeric
entries of ``run_manifest.json`` diagnostics when both exist.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def _arrays(obj, prefix=""):
    """Yield (name, array) for everything numeric inside a loaded object."""
    if isinstance(obj, np.lib.npyio.NpzFile):
        for key in obj.files:
            yield from _arrays(obj[key], f"{prefix}{key}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _arrays(value, f"{prefix}{key}/")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from _arrays(value, f"{prefix}{i}/")
    elif hasattr(obj, "values") and hasattr(obj, "axes"):      # hist.Hist
        yield prefix.rstrip("/") + ":values", np.asarray(obj.values(flow=True), float)
        variances = obj.variances(flow=True)
        if variances is not None:
            yield prefix.rstrip("/") + ":variances", np.asarray(variances, float)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in "biufc":
            yield prefix.rstrip("/"), obj.astype(float)
        elif obj.dtype.kind in "US":
            yield prefix.rstrip("/") + ":str", obj
    elif isinstance(obj, (int, float, np.generic)):
        yield prefix.rstrip("/"), np.asarray(float(obj))
    elif isinstance(obj, str):
        yield prefix.rstrip("/") + ":str", np.asarray(obj)


def load(path):
    if path.suffix == ".npz":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if path.name == "run_manifest.json":
        return json.loads(path.read_text()).get("diagnostics", {})
    raise ValueError(path)


def compare_file(golden, new, rtol, atol):
    g = dict(_arrays(load(golden)))
    n = dict(_arrays(load(new)))
    worst = 0.0
    lines = []
    for key in sorted(set(g) | set(n)):
        if key not in g or key not in n:
            lines.append(f"  MISSING {'in new' if key not in n else 'in golden'}: {key}")
            worst = np.inf
            continue
        a, b = np.atleast_1d(g[key]), np.atleast_1d(n[key])
        if a.dtype.kind in "US" or b.dtype.kind in "US":
            if not np.array_equal(a, b):
                lines.append(f"  DIFF (strings) {key}: {a} vs {b}")
                worst = np.inf
            continue
        if a.shape != b.shape:
            lines.append(f"  SHAPE {key}: {a.shape} vs {b.shape}")
            worst = np.inf
            continue
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.abs(a - b)
        diff[both_nan] = 0.0
        scale = np.maximum(np.abs(a), np.abs(b))
        rel = np.where(scale > 0, diff / np.maximum(scale, 1e-300), 0.0)
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        max_rel = float(np.nanmax(rel)) if rel.size else 0.0
        ok = np.all(diff <= atol + rtol * scale) or np.all(diff == 0)
        worst = max(worst, max_rel if np.isfinite(max_rel) else np.inf)
        if not ok:
            lines.append(f"  DIFF {key}: max|d|={max_abs:.3g} max rel={max_rel:.3g}")
    return worst, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("golden", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    files = sorted(p for p in args.golden.rglob("*")
                   if p.suffix in {".npz", ".pkl"} or p.name == "run_manifest.json")
    files = [p for p in files if "_previews" not in p.parts]
    failed = 0
    for golden in files:
        rel = golden.relative_to(args.golden)
        new = args.new / rel
        if not new.exists():
            print(f"MISSING {rel}")
            failed += 1
            continue
        worst, lines = compare_file(golden, new, args.rtol, args.atol)
        status = "ok " if not lines else "BAD"
        print(f"{status} {rel}  (worst rel diff {worst:.2e})")
        if lines and (args.verbose or len(lines) <= 40):
            print("\n".join(lines))
        elif lines:
            print("\n".join(lines[:40]) + f"\n  ... {len(lines) - 40} more")
        failed += bool(lines)
    print(f"\n{len(files) - failed}/{len(files)} files match at rtol={args.rtol:g}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
