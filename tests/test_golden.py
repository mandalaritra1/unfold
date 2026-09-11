"""Golden regression: the data products must match the pre-restructure outputs.

Skipped unless both trees exist locally:

    outputs/_golden_legacy/   produced by the snapshot-2026-09-09 tree
    outputs/_golden_new/      produced by `unfold run --channel <c> --output-dir outputs/_golden_new/<c>/rho/original --no-gallery --no-stamp`
                              (Z+jet with --era-split linear; dijet/trijet with --stat-method analytic)

See README.md ("Checking that nothing changed") for the commands.
"""

from pathlib import Path

import pytest

from compare_golden import compare_file

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "outputs" / "_golden_legacy"
NEW = REPO / "outputs" / "_golden_new"

# zjet "original" = the former jmsjmr_unity_groomed400_floor3 configuration
CASES = {"zjet_original": "zjet/rho/original/data"}


def _pairsplit_cases():
    """Pair-split run directories are named by a config fingerprint; pair them by channel/mode."""
    cases = {}
    if not GOLDEN.exists():
        return cases
    for artifact in GOLDEN.glob("pairsplit_run2/*/aligned/**/artifacts/*.npz"):
        rel = artifact.relative_to(GOLDEN)
        channel = rel.parts[1]
        mode = artifact.stem.replace("_results", "")
        new = sorted(NEW.glob(f"{channel}/rho/original/{mode}/artifacts/{mode}_results.npz"))
        cases[f"pairsplit_{channel}_{mode}"] = (artifact, new[0] if new else None)
    return cases


@pytest.mark.skipif(not (GOLDEN.exists() and NEW.exists()), reason="golden trees not present")
@pytest.mark.parametrize("name", list(CASES))
def test_zjet_products_match(name):
    golden_dir = GOLDEN / CASES[name]
    new_dir = NEW / CASES[name]
    files = sorted(p for p in golden_dir.iterdir() if p.suffix in {".npz", ".pkl"})
    assert files, f"no golden products under {golden_dir}"
    for golden in files:
        worst, lines = compare_file(golden, new_dir / golden.name, rtol=1e-9, atol=0.0)
        assert not lines, f"{golden.name}:\n" + "\n".join(lines)


@pytest.mark.skipif(not (GOLDEN.exists() and NEW.exists()), reason="golden trees not present")
@pytest.mark.parametrize("name", list(_pairsplit_cases()))
def test_pairsplit_artifacts_match(name):
    golden, new = _pairsplit_cases()[name]
    assert new is not None, f"no new artifact for {name}"
    worst, lines = compare_file(golden, new, rtol=1e-9, atol=0.0)
    assert not lines, "\n".join(lines)
