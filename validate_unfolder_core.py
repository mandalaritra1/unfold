"""Validate that unfolder_core.Unfolder(MASS_SPEC) matches legacy Unfolder_mass.Unfolder.

Usage (from repo root, with .venv active):
    python validate_unfolder_core.py

Runs both classes for one (groomed, do_syst) configuration, then compares
the key analysis arrays. Prints PASS / FAIL per array.

Default config is groomed=False, do_syst=False because that's the fastest;
bump DO_SYST to True for a full check after the fast one passes.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from unfold.tools.Unfolder_mass import Unfolder as LegacyUnfolder  # noqa: E402
from unfold.tools.unfolder_core import Unfolder as CoreUnfolder, MASS_SPEC  # noqa: E402


GROOMED = False
DO_SYST = True       # set True for full coverage (slow)
RTOL = 1e-10
ATOL = 1e-12


def dump_state(u):
    """Capture the analysis state we care about."""
    keys_scalar = ["groomed", "closure", "herwig_closure"]
    arr_keys = [
        "y_unf", "y_true", "y_meas", "y_true_herwig",
        "x_folded",
        "mosaic", "mosaic_2d", "mosaic_gen",
        "fake_fraction_2d",
        "fakes_2d", "misses_2d",
        "stat_unc_frac", "input_stat_unc_frac", "matrix_stat_unc_frac",
    ]
    snap = {k: getattr(u, k, None) for k in keys_scalar}
    for k in arr_keys:
        v = getattr(u, k, None)
        snap[k] = None if v is None else np.asarray(v)
    # sys_matrix_dic: compare keyset and a few representative entries
    snap["sys_matrix_keys"] = sorted(getattr(u, "sys_matrix_dic", {}).keys())
    snap["sys_matrix_nominal"] = np.asarray(u.sys_matrix_dic["nominal"]) if "nominal" in getattr(u, "sys_matrix_dic", {}) else None
    # normalized_results is a dict keyed by pt-index containing dicts of arrays
    normalized = getattr(u, "normalized_results", None)
    snap["normalized_results"] = normalized
    return snap


def compare(a, b, name):
    if a is None and b is None:
        print(f"  [SKIP] {name}: both None")
        return True
    if (a is None) != (b is None):
        print(f"  [FAIL] {name}: one is None ({a is None=}, {b is None=})")
        return False
    if isinstance(a, list) and isinstance(b, list):
        if a != b:
            print(f"  [FAIL] {name}: list mismatch")
            return False
        print(f"  [ OK ] {name}: lists equal (len={len(a)})")
        return True
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            print(f"  [FAIL] {name}: shape {a.shape} vs {b.shape}")
            return False
        if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            diff = np.nanmax(np.abs(a - b))
            print(f"  [FAIL] {name}: max_abs_diff={diff:.3e}")
            return False
        print(f"  [ OK ] {name}: shape={a.shape}")
        return True
    if a == b:
        print(f"  [ OK ] {name}: equal")
        return True
    print(f"  [FAIL] {name}: {a!r} vs {b!r}")
    return False


def _walk_compare(a, b, path, report):
    """Recursively compare nested dict/list/array structures. Appends failures to report."""
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        report.append(f"  [FAIL] {path}: one is None ({a is None=}, {b is None=})")
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            report.append(f"  [FAIL] {path}: key mismatch {set(a) ^ set(b)}")
            return False
        ok = True
        for k in sorted(a.keys(), key=str):
            ok &= _walk_compare(a[k], b[k], f"{path}[{k!r}]", report)
        return ok
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            report.append(f"  [FAIL] {path}: list len {len(a)} vs {len(b)}")
            return False
        ok = True
        for i, (ai, bi) in enumerate(zip(a, b)):
            ok &= _walk_compare(ai, bi, f"{path}[{i}]", report)
        return ok
    # leaf: treat as array-like
    try:
        va = np.asarray(a)
        vb = np.asarray(b)
    except Exception as e:
        report.append(f"  [FAIL] {path}: cannot coerce to array ({e})")
        return False
    if va.shape != vb.shape:
        report.append(f"  [FAIL] {path}: shape {va.shape} vs {vb.shape}")
        return False
    if va.dtype == object or vb.dtype == object:
        if not np.array_equal(va, vb):
            report.append(f"  [FAIL] {path}: object-array mismatch")
            return False
        return True
    if not np.allclose(va, vb, rtol=RTOL, atol=ATOL, equal_nan=True):
        diff = np.nanmax(np.abs(va - vb))
        report.append(f"  [FAIL] {path}: max_abs_diff={diff:.3e}")
        return False
    return True


def compare_normalized(a, b):
    if a is None and b is None:
        print("  [SKIP] normalized_results: both None")
        return True
    if (a is None) != (b is None):
        print(f"  [FAIL] normalized_results: one is None")
        return False
    # normalize list → dict(enumerate) at the top level so pt indices line up
    if isinstance(a, list):
        a = dict(enumerate(a))
    if isinstance(b, list):
        b = dict(enumerate(b))
    report = []
    ok = _walk_compare(a, b, "normalized_results", report)
    for line in report:
        print(line)
    if ok:
        n = len(a) if isinstance(a, dict) else "?"
        print(f"  [ OK ] normalized_results: {n} pt bins match")
    return ok


def main():
    print(f"Config: groomed={GROOMED}, do_syst={DO_SYST}")
    print("--- instantiating LEGACY Unfolder_mass.Unfolder ---")
    legacy = LegacyUnfolder(groomed=GROOMED, do_syst=DO_SYST)
    print("--- instantiating CORE unfolder_core.Unfolder(MASS_SPEC) ---")
    core = CoreUnfolder(MASS_SPEC, groomed=GROOMED, do_syst=DO_SYST)

    print("--- dumping state ---")
    a = dump_state(legacy)
    b = dump_state(core)

    print("--- comparing ---")
    all_ok = True
    for k in a:
        if k == "normalized_results":
            all_ok &= compare_normalized(a[k], b[k])
        else:
            all_ok &= compare(a[k], b[k], k)

    print("")
    print("ALL PASS" if all_ok else "SOME FAIL — investigate above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
