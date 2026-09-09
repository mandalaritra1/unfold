"""Raw internal-Pythia transfer ratios for pair-split model uncertainties.

This is the ungroomed counterpart of the existing harvested groomed transfer
file.  It deliberately reproduces ``harvest_internal_variations.py``: within
each HT slice, raw selected-event weights are divided by the pooled generated
luminosity ``sum(sumw / xsec_pb)`` for that Pythia configuration.  The final
model input is the normalized variation / normalized standalone-CP5 ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from unfold.paths import INTERNAL_CAMPAIGN_DIR, INTERNAL_RESULTS


CONFIGS = ("cp5", "cr1", "cr2", "fraghard", "fragsoft")
VARIATIONS = CONFIGS[1:]
TRANSFER_PT_BINS = ((200.0, 290.0), (290.0, 400.0), (400.0, 13000.0))
DEFAULT_CAMPAIGN = "mgmlm_internal_trijet_20260802_v1"
DEFAULT_CAMPAIGN_DIRECTORY = INTERNAL_CAMPAIGN_DIR
DEFAULT_FINAL_DIRECTORY = INTERNAL_RESULTS / "final_1000of1000"


class PairSplitInternalVariationError(RuntimeError):
    """The raw internal-variation campaign violates its harvest contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PairSplitInternalTransfers:
    channel: str
    grooming_mode: str
    coordinate_edges: tuple[float, ...]
    ratio_by_source_and_pt_low: Mapping[str, Mapping[int, np.ndarray]]
    provenance: Mapping[str, object]


def load_pairsplit_internal_transfers(
    channel: str,
    *,
    grooming_mode: str,
    coordinate_edges: Sequence[float],
    campaign_directory: Path = DEFAULT_CAMPAIGN_DIRECTORY,
    final_directory: Path = DEFAULT_FINAL_DIRECTORY,
    expected_campaign: str = DEFAULT_CAMPAIGN,
    required_manifest_count: int = 1000,
) -> PairSplitInternalTransfers:
    """Derive exact raw variation/CP5 ratios on ``coordinate_edges``."""

    if channel not in {"dijet", "trijet"}:
        raise ValueError("channel must be 'dijet' or 'trijet'")
    if grooming_mode not in {"groomed", "ungroomed"}:
        raise ValueError("grooming_mode must be 'groomed' or 'ungroomed'")
    coordinate_edges = np.asarray(coordinate_edges, dtype=float)
    if coordinate_edges.ndim != 1 or not np.all(np.diff(coordinate_edges) > 0.0):
        raise ValueError("coordinate_edges must be a strictly increasing 1D array")

    campaign_directory = Path(campaign_directory).resolve()
    final_directory = Path(final_directory).resolve()
    snapshot_path = final_directory / "manifest_snapshot.txt"
    validation_path = final_directory / "campaign_validation.json"
    if not snapshot_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("internal-variation final snapshot or validation is missing")
    validation = json.loads(validation_path.read_text())
    if (
        validation.get("complete") is not True
        or validation.get("errors")
        or validation.get("campaign") != expected_campaign
    ):
        raise PairSplitInternalVariationError("internal-variation campaign validation failed")

    selected: list[tuple[Path, dict]] = []
    seen: set[tuple[str, int]] = set()
    for raw_path in snapshot_path.read_text().splitlines():
        if not raw_path.strip():
            continue
        original = Path(raw_path.strip())
        path = original if original.is_file() else campaign_directory / original.parent.name / original.name
        if not path.is_file():
            raise FileNotFoundError(f"missing internal-variation manifest: {path}")
        manifest = json.loads(path.read_text())
        if manifest.get("mode") != "five_million":
            continue
        if not manifest.get("complete") or manifest.get("campaign") != expected_campaign:
            raise PairSplitInternalVariationError(f"invalid manifest: {path}")
        key = (str(manifest["bin"]), int(manifest["seed"]))
        if key in seen:
            raise PairSplitInternalVariationError(f"duplicate internal manifest key {key}")
        seen.add(key)
        selected.append((path, manifest))
    expected_count = int(validation.get("mode_counts", {}).get("five_million", -1))
    if len(selected) != expected_count or expected_count != required_manifest_count:
        raise PairSplitInternalVariationError(
            f"expected {required_manifest_count} five_million manifests, "
            f"found {len(selected)}"
        )

    shape = len(coordinate_edges) - 1
    numerators: dict[tuple[str, str, int], np.ndarray] = {}
    denominators: dict[tuple[str, str], float] = {}
    manifest_hashes: list[str] = []
    ntuple_inventory: list[dict[str, object]] = []
    coordinate_column = 4 if grooming_mode == "groomed" else 3
    for manifest_path, manifest in selected:
        ht_bin = str(manifest["bin"])
        manifest_hashes.append(_sha256(manifest_path))
        entries = {str(entry["config"]): entry for entry in manifest["configs"]}
        if tuple(entries) != CONFIGS:
            raise PairSplitInternalVariationError(f"wrong config set in {manifest_path}")
        for config in CONFIGS:
            entry = entries[config]
            generated_lumi = float(entry["sumw"]) / float(entry["xsec_pb"])
            if not np.isfinite(generated_lumi) or generated_lumi <= 0.0:
                raise PairSplitInternalVariationError(
                    f"invalid generated luminosity in {manifest_path} ({config})"
                )
            denominators[(ht_bin, config)] = (
                denominators.get((ht_bin, config), 0.0) + generated_lumi
            )
            ntuple_path = manifest_path.parent / str(entry[f"{channel}_ntuple"])
            payload = ntuple_path.read_bytes()
            try:
                rows = np.loadtxt(io.BytesIO(payload), comments="#", ndmin=2)
            except ValueError:
                rows = np.empty((0, 6), dtype=float)
            rows = np.asarray(rows, dtype=float)
            if rows.size == 0:
                rows = np.empty((0, 6), dtype=float)
            if rows.shape != (int(entry[f"{channel}_rows"]), 6):
                raise PairSplitInternalVariationError(
                    f"row-count/schema mismatch in {ntuple_path}: {rows.shape}"
                )
            if not np.all(np.isfinite(rows)):
                raise PairSplitInternalVariationError(f"non-finite row in {ntuple_path}")
            ntuple_inventory.append(
                {
                    "path": str(ntuple_path),
                    "sha256": _sha256_bytes(payload),
                    "rows": len(rows),
                }
            )
            pt = rows[:, 0]
            coordinate = rows[:, coordinate_column]
            weights = rows[:, 5]
            for pt_low, pt_high in TRANSFER_PT_BINS:
                key = (ht_bin, config, int(pt_low))
                target = numerators.setdefault(key, np.zeros(shape, dtype=float))
                mask = (
                    (pt >= pt_low)
                    & (pt < pt_high)
                    & np.isfinite(coordinate)
                    & (coordinate > -99.0)
                )
                target += np.histogram(
                    coordinate[mask], bins=coordinate_edges, weights=weights[mask]
                )[0]

    spectra: dict[str, dict[int, np.ndarray]] = {config: {} for config in CONFIGS}
    ht_bins = sorted({key[0] for key in denominators})
    for config in CONFIGS:
        for pt_low, _ in TRANSFER_PT_BINS:
            spectrum = np.zeros(shape, dtype=float)
            for ht_bin in ht_bins:
                spectrum += numerators[(ht_bin, config, int(pt_low))] / denominators[
                    (ht_bin, config)
                ]
            total = float(spectrum.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise PairSplitInternalVariationError(
                    f"empty internal {config} {channel} spectrum at {pt_low:g} GeV"
                )
            spectra[config][int(pt_low)] = spectrum / total

    ratios: dict[str, dict[int, np.ndarray]] = {source: {} for source in VARIATIONS}
    for source in VARIATIONS:
        for pt_low, _ in TRANSFER_PT_BINS:
            nominal = spectra["cp5"][int(pt_low)]
            varied = spectra[source][int(pt_low)]
            ratios[source][int(pt_low)] = np.divide(
                varied,
                nominal,
                out=np.full_like(varied, np.nan),
                where=nominal > 0.0,
            )

    inventory_hash = hashlib.sha256(
        json.dumps(ntuple_inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PairSplitInternalTransfers(
        channel=channel,
        grooming_mode=grooming_mode,
        coordinate_edges=tuple(float(value) for value in coordinate_edges),
        ratio_by_source_and_pt_low=ratios,
        provenance={
            "campaign": expected_campaign,
            "mode": "five_million",
            "manifest_count": len(selected),
            "campaign_directory": str(campaign_directory),
            "manifest_snapshot": {"path": str(snapshot_path), "sha256": _sha256(snapshot_path)},
            "campaign_validation": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
            },
            "manifest_inventory_sha256": hashlib.sha256(
                "".join(sorted(manifest_hashes)).encode()
            ).hexdigest(),
            "ntuple_inventory_sha256": inventory_hash,
            "normalization": (
                "per HT and configuration: sum selected row weights divided by "
                "sum(sumw/xsec_pb), then sum HT spectra and normalize to unit area"
            ),
            "coordinate": f"2*log10(rho), {grooming_mode}",
        },
    )
