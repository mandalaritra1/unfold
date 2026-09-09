"""Fail-closed MESS+Vincia inputs for the Run-2 pair-split unfolding plots.

The hadronic MESS prediction is intentionally separate from the Z+jet model
cache in :mod:`unfold.model`.  It consumes the audited
``final-all`` allowlist directly, checks every selected manifest and ntuple,
and histograms the event rows onto the *current* Unfolder truth binning.

The physical variable is ``rho = m / (pT * R)`` with ``R = 0.8``; the stored
coordinate is ``2 * log10(rho)``.  No mass floor or shape reweighting is
applied.  Tiny negative MESS FastJet residuals admitted by the campaign audit
are mapped to zero and therefore, correctly, have no finite coordinate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from unfold.model import normalized_prediction_covariance
from unfold.paths import MESS_CAMPAIGN_DIR, MESS_RESULTS


PAIR_SPLIT_CHANNELS = ("dijet", "trijet")
JET_RADIUS = 0.8
NUMERICAL_MASS_TOLERANCE_GEV = 1.0e-4
PREDICTION_LABEL = "Vincia"
# Frozen campaign identifier used in the run identity hash.  The display
# label above was shortened to "Vincia" (2026-09-02); keeping the identity
# string unchanged keeps the immutable run directory names stable.
CAMPAIGN_IDENTITY_LABEL = "MESS+Vincia"
SELECTION = "CMS_HADRONIC_PAIR_SPLIT"
ALLOWED_PHASES = frozenset(("pilot2", "full"))

DEFAULT_FINAL_ALL_DIRECTORY = MESS_RESULTS / "final-all"
DEFAULT_ALLOWLIST = DEFAULT_FINAL_ALL_DIRECTORY / "final_all_allowlist.json"
DEFAULT_AUDIT = DEFAULT_FINAL_ALL_DIRECTORY / "final_all_audit.json"
DEFAULT_CAMPAIGN_DIRECTORY = MESS_CAMPAIGN_DIR
DEFAULT_COMPILED_DIJET_REFERENCE = (
    DEFAULT_FINAL_ALL_DIRECTORY / "model_comparison/model_comparison_arrays.npz"
)


class PairSplitVinciaValidationError(RuntimeError):
    """An audited MESS+Vincia input does not satisfy the final-all contract."""


def sha256(path: Path) -> str:
    """Return the content hash used by the final-all allowlist."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> np.ndarray:
    """Load one six-column MESS ntuple, retaining zero-row outputs exactly."""

    with path.open() as source:
        has_data = any(
            line.strip() and not line.lstrip().startswith("#") for line in source
        )
    if not has_data:
        return np.empty((0, 6), dtype=float)
    try:
        rows = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError:
        return np.empty((0, 6), dtype=float)
    rows = np.asarray(rows, dtype=float)
    if rows.size == 0:
        return np.empty((0, 6), dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise PairSplitVinciaValidationError(
            f"unexpected MESS ntuple schema in {path}: {rows.shape}; expected (N, 6)"
        )
    if not np.all(np.isfinite(rows[:, (0, 1, 2, 5)])):
        raise PairSplitVinciaValidationError(f"non-finite pT, mass, or weight in {path}")
    return rows


def _resolve_manifest_path(record_path: Path, campaign_directory: Path) -> Path:
    """Resolve an allowlisted original path against the audited campaign mirror."""

    if record_path.is_file():
        return record_path
    candidate = campaign_directory / record_path.parent.name / record_path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"cannot resolve allowlisted MESS manifest {record_path}")


def _validate_allowlist_and_audit(
    allowlist_path: Path,
    audit_path: Path,
) -> tuple[dict, dict]:
    if not allowlist_path.is_file():
        raise FileNotFoundError(f"MESS final-all allowlist is missing: {allowlist_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"MESS final-all audit is missing: {audit_path}")
    allowlist = json.loads(allowlist_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("valid") is not True or audit.get("errors"):
        raise PairSplitVinciaValidationError("MESS final-all audit is not valid")
    if allowlist.get("audit_sha256") != sha256(audit_path):
        raise PairSplitVinciaValidationError(
            "MESS allowlist audit hash does not match the final-all audit"
        )
    manifests = allowlist.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise PairSplitVinciaValidationError("MESS final-all allowlist has no manifests")
    if int(allowlist.get("manifest_count", -1)) != len(manifests):
        raise PairSplitVinciaValidationError("MESS allowlist manifest_count is inconsistent")
    if int(audit.get("valid_manifests", -1)) != len(manifests):
        raise PairSplitVinciaValidationError("MESS audit and allowlist manifest counts differ")
    if allowlist.get("campaign") != audit.get("campaign"):
        raise PairSplitVinciaValidationError("MESS audit and allowlist campaigns differ")
    selected_phases = set(audit.get("selected_phases", []))
    if not selected_phases or not selected_phases <= ALLOWED_PHASES:
        raise PairSplitVinciaValidationError(
            f"MESS audit has unsupported selected phases: {sorted(selected_phases)}"
        )
    return allowlist, audit


@dataclass(frozen=True)
class PairSplitVinciaSource:
    """Validated raw event rows and stable provenance for one pair-split channel."""

    channel: str
    campaign: str
    campaign_directory: Path
    allowlist_path: Path
    audit_path: Path
    allowlist_sha256: str
    audit_sha256: str
    manifest_ntuple_inventory_sha256: str
    rows_by_ht: Mapping[str, np.ndarray]
    # Each entry already applies the originating manifest's input cross
    # section over the pooled HT denominator. Cross sections are not constant
    # within an HT slice in the final-all campaign.
    row_weights_by_ht: Mapping[str, np.ndarray]
    per_ht: Mapping[str, Mapping[str, object]]

    @property
    def label(self) -> str:
        return PREDICTION_LABEL

    def identity_payload(self) -> dict[str, object]:
        """JSON-safe source identity used in immutable output directory names."""

        return {
            "label": CAMPAIGN_IDENTITY_LABEL,
            "channel": self.channel,
            "campaign": self.campaign,
            "campaign_directory": str(self.campaign_directory),
            "allowlist": {
                "path": str(self.allowlist_path),
                "sha256": self.allowlist_sha256,
            },
            "audit": {"path": str(self.audit_path), "sha256": self.audit_sha256},
            "manifest_ntuple_inventory_sha256": self.manifest_ntuple_inventory_sha256,
            "normalization": (
                "per row: rows[:,5] * manifest_input_xsec_pb / "
                "sum_ht_lhe_events"
            ),
            "row_schema": "jet_pt m_u m_g rho_u rho_g weight",
            "physical_rho_definitions": {
                "groomed": "rho = m_g/(pt*0.8)",
                "ungroomed": "rho = m_u/(pt*0.8)",
            },
            "analyzed_coordinate_definition": "2*log10(rho)",
        }

    def provenance_payload(self) -> dict[str, object]:
        """Stable, compact provenance stored in the run manifest."""

        return {**self.identity_payload(), "per_ht": self.per_ht}


@dataclass(frozen=True)
class PairSplitVinciaPrediction:
    """MESS+Vincia density arrays on the exact Unfolder truth binning."""

    source: PairSplitVinciaSource
    pt_edges: tuple[float, ...]
    gen_edges_by_pt: tuple[tuple[float, ...], ...]
    density_by_pt: tuple[np.ndarray, ...]
    stat_unc_by_pt: tuple[np.ndarray, ...]
    stat_covariance_by_pt: tuple[np.ndarray, ...]
    sumw_by_pt: tuple[np.ndarray, ...]
    sumw2_by_pt: tuple[np.ndarray, ...]
    normalization_totals_by_pt: tuple[float, ...]
    normalization_masks_by_pt: tuple[np.ndarray, ...]
    grooming_mode: str = "groomed"
    normalization_window: tuple[float, float] = (-4.0, 0.0)

    @property
    def label(self) -> str:
        return self.source.label

    def truth_by_pt(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return the core plotting contract: density and MC-stat error by pT bin."""

        return {
            index: (self.density_by_pt[index], self.stat_unc_by_pt[index])
            for index in range(len(self.density_by_pt))
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        """Stable flattened arrays for downstream comparisons without plot scraping."""

        bin_offsets = np.concatenate(
            ([0], np.cumsum([values.size for values in self.density_by_pt], dtype=int))
        )
        edge_offsets = np.concatenate(
            ([0], np.cumsum([len(edges) for edges in self.gen_edges_by_pt], dtype=int))
        )
        covariance = np.zeros((bin_offsets[-1], bin_offsets[-1]), dtype=float)
        for index, matrix in enumerate(self.stat_covariance_by_pt):
            block = slice(bin_offsets[index], bin_offsets[index + 1])
            covariance[block, block] = matrix
        return {
            "mess_vincia_grooming_mode": np.asarray(self.grooming_mode),
            "mess_vincia_pt_edges": np.asarray(self.pt_edges, dtype=float),
            "mess_vincia_gen_edges_flat": np.concatenate(
                [np.asarray(edges, dtype=float) for edges in self.gen_edges_by_pt]
            ),
            "mess_vincia_gen_edge_offsets": edge_offsets,
            "mess_vincia_bin_offsets": bin_offsets,
            "mess_vincia_density_flat": np.concatenate(self.density_by_pt),
            "mess_vincia_stat_unc_flat": np.concatenate(self.stat_unc_by_pt),
            "mess_vincia_stat_covariance": covariance,
            "mess_vincia_sumw_flat": np.concatenate(self.sumw_by_pt),
            "mess_vincia_sumw2_flat": np.concatenate(self.sumw2_by_pt),
            "mess_vincia_normalization_totals": np.asarray(
                self.normalization_totals_by_pt, dtype=float
            ),
            "mess_vincia_normalization_masks_flat": np.concatenate(
                self.normalization_masks_by_pt
            ).astype(np.uint8),
        }

    def provenance_payload(self) -> dict[str, object]:
        """Derived-curve provenance, including the exact binned ordering."""

        return {
            "label": self.label,
            "grooming_mode": self.grooming_mode,
            "source": self.source.provenance_payload(),
            "pt_edges_GeV": list(self.pt_edges),
            "gen_two_log10_rho_edges_by_pt": [list(edges) for edges in self.gen_edges_by_pt],
            "flat_bin_order": "concatenate pT slices in ascending pT, then ascending 2log10(rho)",
            "normalization_totals_by_pt": list(self.normalization_totals_by_pt),
            "normalization": (
                "unit area over bins fully contained in "
                f"[{self.normalization_window[0]:g}, {self.normalization_window[1]:g}]"
            ),
            "normalization_window": list(self.normalization_window),
            "statistical_uncertainty": "sqrt(diag(J diag(sumw2) J.T)); J includes normalization denominator",
            "statistical_covariance_limitation": (
                "per-jet sumw2 input; event-level and cross-pT correlations unavailable"
            ),
        }


def load_pairsplit_vincia_source(
    channel: str,
    *,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
    audit_path: Path = DEFAULT_AUDIT,
    campaign_directory: Path = DEFAULT_CAMPAIGN_DIRECTORY,
) -> PairSplitVinciaSource:
    """Read the final-all raw rows after validating every input hash and count.

    This deliberately does not inspect the coarse ``model_comparison_arrays``
    product: the unfolding can have more pT slices and a different truth
    binning, so the raw final-all rows are the only valid source.
    """

    if channel not in PAIR_SPLIT_CHANNELS:
        raise ValueError(f"unsupported pair-split MESS channel {channel!r}")
    allowlist_path = Path(allowlist_path).resolve()
    audit_path = Path(audit_path).resolve()
    campaign_directory = Path(campaign_directory).resolve()
    if not campaign_directory.is_dir():
        raise FileNotFoundError(f"MESS campaign directory is missing: {campaign_directory}")
    allowlist, audit = _validate_allowlist_and_audit(allowlist_path, audit_path)
    audited_campaign_directory = audit.get("campaign_dir")
    if (
        audited_campaign_directory is not None
        and Path(audited_campaign_directory).resolve() != campaign_directory
    ):
        raise PairSplitVinciaValidationError(
            "requested MESS campaign directory does not match the final-all audit"
        )

    manifests_by_ht: dict[str, list[tuple[dict, Path, str]]] = defaultdict(list)
    inventory: list[dict[str, object]] = []
    selected_phases = set(audit["selected_phases"])
    for record in sorted(allowlist["manifests"], key=lambda item: (item["bin"], int(item["seed"]))):
        manifest_path = _resolve_manifest_path(Path(record["path"]), campaign_directory)
        manifest_hash = sha256(manifest_path)
        if manifest_hash != record.get("sha256"):
            raise PairSplitVinciaValidationError(
                f"allowlisted manifest hash mismatch: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("campaign") != allowlist.get("campaign"):
            raise PairSplitVinciaValidationError(
                f"campaign mismatch in {manifest_path}"
            )
        if manifest.get("selection") != SELECTION:
            raise PairSplitVinciaValidationError(f"selection mismatch in {manifest_path}")
        if manifest.get("phase") not in selected_phases or manifest.get("phase") not in ALLOWED_PHASES:
            raise PairSplitVinciaValidationError(f"disallowed final-all phase in {manifest_path}")
        for field in ("bin", "seed", "lhe_events", "phase"):
            if manifest.get(field) != record.get(field):
                raise PairSplitVinciaValidationError(
                    f"allowlist/manifest {field} mismatch in {manifest_path}"
                )
        if not manifest.get("complete", False):
            raise PairSplitVinciaValidationError(f"incomplete MESS manifest: {manifest_path}")
        manifests_by_ht[str(manifest["bin"])].append(
            (manifest, manifest_path, manifest_hash)
        )

    rows_by_ht: dict[str, np.ndarray] = {}
    row_weights_by_ht: dict[str, np.ndarray] = {}
    per_ht: dict[str, dict[str, object]] = {}
    for ht_bin, entries in sorted(manifests_by_ht.items()):
        denominator = sum(int(manifest["lhe_events"]) for manifest, _, _ in entries)
        if denominator <= 0:
            raise PairSplitVinciaValidationError(f"nonpositive LHE denominator in {ht_bin}")
        ntuple_rows: list[np.ndarray] = []
        ntuple_weights: list[np.ndarray] = []
        xsecs: list[float] = []
        expected_rows = 0
        selected_events = 0
        for manifest, manifest_path, manifest_hash in entries:
            ntuple_name = manifest.get(f"{channel}_ntuple")
            if not isinstance(ntuple_name, str):
                raise PairSplitVinciaValidationError(
                    f"missing {channel} ntuple name in {manifest_path}"
                )
            ntuple_path = manifest_path.parent / ntuple_name
            if not ntuple_path.is_file():
                raise FileNotFoundError(f"missing MESS ntuple {ntuple_path}")
            ntuple_hash = sha256(ntuple_path)
            if ntuple_hash != manifest.get("files_sha256", {}).get(ntuple_name):
                raise PairSplitVinciaValidationError(f"ntuple hash mismatch: {ntuple_path}")
            rows = _read_rows(ntuple_path)
            manifest_rows = int(manifest.get(f"{channel}_rows", -1))
            if len(rows) != manifest_rows:
                raise PairSplitVinciaValidationError(
                    f"{channel} row count mismatch in {ntuple_path}: "
                    f"{len(rows)} != {manifest_rows}"
                )
            ntuple_rows.append(rows)
            ntuple_weights.append(
                rows[:, 5] * float(manifest["input_xsec_pb"]) / denominator
            )
            expected_rows += manifest_rows
            selected_events += int(manifest.get(f"{channel}_selected_events", 0))
            xsecs.append(float(manifest["input_xsec_pb"]))
            inventory.append(
                {
                    "ht_bin": ht_bin,
                    "manifest": str(manifest_path),
                    "manifest_sha256": manifest_hash,
                    "ntuple": str(ntuple_path),
                    "ntuple_sha256": ntuple_hash,
                    "rows": manifest_rows,
                    "lhe_events": int(manifest["lhe_events"]),
                    "input_xsec_pb": float(manifest["input_xsec_pb"]),
                }
            )
        rows_by_ht[ht_bin] = (
            np.concatenate(ntuple_rows, axis=0) if ntuple_rows else np.empty((0, 6), dtype=float)
        )
        row_weights_by_ht[ht_bin] = (
            np.concatenate(ntuple_weights, axis=0)
            if ntuple_weights
            else np.empty(0, dtype=float)
        )
        if len(rows_by_ht[ht_bin]) != expected_rows:
            raise PairSplitVinciaValidationError(f"assembled row count mismatch in {ht_bin}")
        if len(row_weights_by_ht[ht_bin]) != expected_rows:
            raise PairSplitVinciaValidationError(
                f"assembled normalized-weight count mismatch in {ht_bin}"
            )
        audit_rows = audit.get(f"{channel}_rows_by_bin", {}).get(ht_bin)
        if audit_rows is not None and int(audit_rows) != expected_rows:
            raise PairSplitVinciaValidationError(
                f"audit {channel} rows disagree in {ht_bin}: {expected_rows} != {audit_rows}"
            )
        per_ht[ht_bin] = {
            "manifests": len(entries),
            "lhe_events": denominator,
            "input_xsec_pb_min": float(min(xsecs)),
            "input_xsec_pb_max": float(max(xsecs)),
            "rows": expected_rows,
            "selected_events": selected_events,
        }

    return PairSplitVinciaSource(
        channel=channel,
        campaign=str(allowlist["campaign"]),
        campaign_directory=campaign_directory,
        allowlist_path=allowlist_path,
        audit_path=audit_path,
        allowlist_sha256=sha256(allowlist_path),
        audit_sha256=sha256(audit_path),
        manifest_ntuple_inventory_sha256=_canonical_hash(inventory),
        rows_by_ht=rows_by_ht,
        row_weights_by_ht=row_weights_by_ht,
        per_ht=per_ht,
    )


def validate_compiled_reference(
    source: PairSplitVinciaSource,
    *,
    target_prediction: PairSplitVinciaPrediction | None = None,
    reference_path: Path = DEFAULT_COMPILED_DIJET_REFERENCE,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-12,
) -> dict[str, object]:
    """Cross-check pair-split MESS rows against the harvested final-all NPZ.

    The reference was produced on the coarser ``400--infinity`` pT interval.
    We deliberately recombine the raw ``400--480``, ``480--570``, and
    ``570--13000`` pieces here, so the test also protects the fine high-pT
    bin handling used by the Unfolder-facing prediction.
    """

    if target_prediction is not None and target_prediction.grooming_mode != "groomed":
        return {
            "channel": source.channel,
            "grooming_mode": target_prediction.grooming_mode,
            "applies": False,
            "reason": "the harvested compiled regression contains groomed coordinates only",
        }

    if source.channel == "trijet":
        return _validate_trijet_compiled_reference(
            source,
            target_prediction=target_prediction,
            reference_path=reference_path,
            rtol=rtol,
            atol=atol,
        )
    if source.channel != "dijet":
        raise ValueError(f"unsupported pair-split MESS channel {source.channel!r}")
    reference_path = Path(reference_path).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"compiled MESS dijet reference is missing: {reference_path}")
    with np.load(reference_path) as reference:
        prefix = "mess_dijet_groomed_rho_pt400"
        expected_sumw = np.asarray(reference[f"{prefix}_sumw"], dtype=float)
        expected_sumw2 = np.asarray(reference[f"{prefix}_sumw2"], dtype=float)
        edges = np.asarray(reference[f"{prefix}_edges"], dtype=float)

    sumw = np.zeros(len(edges) - 1, dtype=float)
    sumw2 = np.zeros_like(sumw)
    high_pt_slices = ((400.0, 480.0), (480.0, 570.0), (570.0, 13000.0))
    for pt_low, pt_high in high_pt_slices:
        for ht_bin, rows in source.rows_by_ht.items():
            pt = rows[:, 0]
            groomed_mass = rows[:, 2]
            if np.any(groomed_mass < -NUMERICAL_MASS_TOLERANCE_GEV):
                raise PairSplitVinciaValidationError(
                    f"groomed mass below audited numerical tolerance in {ht_bin}"
                )
            groomed_mass = np.where(groomed_mass < 0.0, 0.0, groomed_mass)
            valid = (pt > 0.0) & (groomed_mass > 0.0)
            coordinate = np.full(len(rows), np.nan, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                coordinate[valid] = 2.0 * np.log10(
                    groomed_mass[valid] / (pt[valid] * JET_RADIUS)
                )
            selected = (
                (pt >= pt_low)
                & (pt < pt_high)
                & np.isfinite(coordinate)
            )
            row_weight = source.row_weights_by_ht[ht_bin]
            sumw += np.histogram(coordinate[selected], bins=edges, weights=row_weight[selected])[0]
            sumw2 += np.histogram(
                coordinate[selected], bins=edges, weights=row_weight[selected] ** 2
            )[0]
    if not np.allclose(sumw, expected_sumw, rtol=rtol, atol=atol):
        raise PairSplitVinciaValidationError(
            "raw MESS dijet 400--480 + 480--570 + 570--13000 sumw does not "
            "match the harvested final-all reference"
        )
    if not np.allclose(sumw2, expected_sumw2, rtol=rtol, atol=atol):
        raise PairSplitVinciaValidationError(
            "raw MESS dijet 400--480 + 480--570 + 570--13000 sumw2 does not "
            "match the harvested final-all reference"
        )
    target_rebin_validation: dict[str, object] | None = None
    if target_prediction is not None:
        high_pt_indices = [
            index
            for index, (pt_low, pt_high) in enumerate(
                zip(target_prediction.pt_edges[:-1], target_prediction.pt_edges[1:])
            )
            if pt_low >= 400.0 and pt_high <= 13000.0
        ]
        if not high_pt_indices:
            raise PairSplitVinciaValidationError(
                "target MESS+Vincia prediction has no 400 GeV high-pT slices"
            )
        target_edges = np.asarray(
            target_prediction.gen_edges_by_pt[high_pt_indices[0]], dtype=float
        )
        if not all(
            np.array_equal(target_edges, target_prediction.gen_edges_by_pt[index])
            for index in high_pt_indices
        ):
            raise PairSplitVinciaValidationError(
                "compiled high-pT regression requires identical target coordinate edges"
            )
        # The compiled reference is frozen on the pre-2026-08-27 free-edge
        # grid.  When the current target edges do not nest in it (the aligned
        # quarter lattice does not), this REDUNDANT rebin regression is
        # geometrically impossible; the mandatory raw-row check above already
        # protected the rows against the harvested reference.  Record the
        # skip honestly instead of failing the run.
        try:
            rebinned_reference = _rebin_histogram_to_target_edges(
                expected_sumw, edges, target_edges
            )
            rebinned_reference2 = _rebin_histogram_to_target_edges(
                expected_sumw2, edges, target_edges
            )
        except PairSplitVinciaValidationError as nesting_error:
            target_rebin_validation = {
                "applies": False,
                "target_gen_edges": list(target_edges),
                "reason": (
                    "target edges do not nest in the compiled reference edges: "
                    f"{nesting_error}"
                ),
            }
        else:
            target_sumw = np.sum(
                [target_prediction.sumw_by_pt[index] for index in high_pt_indices], axis=0
            )
            target_sumw2 = np.sum(
                [target_prediction.sumw2_by_pt[index] for index in high_pt_indices], axis=0
            )
            if not np.allclose(target_sumw, rebinned_reference, rtol=rtol, atol=atol):
                raise PairSplitVinciaValidationError(
                    "target-binned MESS+Vincia high-pT sumw does not match the "
                    "rebinned harvested final-all reference"
                )
            if not np.allclose(target_sumw2, rebinned_reference2, rtol=rtol, atol=atol):
                raise PairSplitVinciaValidationError(
                    "target-binned MESS+Vincia high-pT sumw2 does not match the "
                    "rebinned harvested final-all reference"
                )
            target_rebin_validation = {
                "target_gen_edges": list(target_edges),
                "target_pt_indices": high_pt_indices,
                "max_abs_sumw_difference": float(
                    np.max(np.abs(target_sumw - rebinned_reference))
                ),
                "max_abs_sumw2_difference": float(
                    np.max(np.abs(target_sumw2 - rebinned_reference2))
                ),
            }
    return {
        "channel": "dijet",
        "applies": True,
        "reference_path": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "pt_slices_GeV": [list(values) for values in high_pt_slices],
        "coordinate_edges": list(edges),
        "max_abs_sumw_difference": float(np.max(np.abs(sumw - expected_sumw))),
        "max_abs_sumw2_difference": float(np.max(np.abs(sumw2 - expected_sumw2))),
        "rtol": rtol,
        "atol": atol,
        "target_binning_rebin": target_rebin_validation,
    }


def _validate_trijet_compiled_reference(
    source: PairSplitVinciaSource,
    *,
    target_prediction: PairSplitVinciaPrediction | None,
    reference_path: Path,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    """Rebin each exact trijet pT reference into the active truth bins."""

    if target_prediction is None:
        raise PairSplitVinciaValidationError(
            "trijet compiled-reference regression requires the target prediction"
        )
    reference_path = Path(reference_path).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"compiled MESS trijet reference is missing: {reference_path}")
    expected_pt_slices = ((200.0, 290.0), (290.0, 400.0), (400.0, 13000.0))
    validations: list[dict[str, object]] = []
    with np.load(reference_path) as reference:
        for pt_low, pt_high in expected_pt_slices:
            matching = [
                index
                for index, bounds in enumerate(
                    zip(target_prediction.pt_edges[:-1], target_prediction.pt_edges[1:])
                )
                if bounds == (pt_low, pt_high)
            ]
            if matching != [len(validations)]:
                raise PairSplitVinciaValidationError(
                    "trijet target pT binning no longer matches the final-all reference"
                )
            index = matching[0]
            prefix = f"mess_trijet_groomed_rho_pt{int(pt_low)}"
            source_edges = np.asarray(reference[f"{prefix}_edges"], dtype=float)
            reference_sumw = np.asarray(reference[f"{prefix}_sumw"], dtype=float)
            reference_sumw2 = np.asarray(reference[f"{prefix}_sumw2"], dtype=float)
            target_edges = np.asarray(target_prediction.gen_edges_by_pt[index], dtype=float)
            # Same nesting caveat as the dijet leg: the compiled reference is
            # frozen on the pre-2026-08-27 free-edge grid, so aligned-lattice
            # target edges cannot nest and this redundant regression is
            # recorded as skipped rather than failing the run (raw-row
            # integrity is enforced by the allowlist/audit hash contract in
            # load_pairsplit_vincia_source).
            try:
                rebinned_sumw = _rebin_histogram_to_target_edges(
                    reference_sumw, source_edges, target_edges
                )
                rebinned_sumw2 = _rebin_histogram_to_target_edges(
                    reference_sumw2, source_edges, target_edges
                )
            except PairSplitVinciaValidationError as nesting_error:
                validations.append(
                    {
                        "pt_range_GeV": [pt_low, pt_high],
                        "applies": False,
                        "target_gen_edges": list(target_edges),
                        "reason": (
                            "target edges do not nest in the compiled "
                            f"reference edges: {nesting_error}"
                        ),
                    }
                )
                continue
            target_sumw = target_prediction.sumw_by_pt[index]
            target_sumw2 = target_prediction.sumw2_by_pt[index]
            if not np.allclose(target_sumw, rebinned_sumw, rtol=rtol, atol=atol):
                raise PairSplitVinciaValidationError(
                    f"target-binned MESS+Vincia trijet {int(pt_low)} GeV sumw does not "
                    "match the rebinned harvested final-all reference"
                )
            if not np.allclose(target_sumw2, rebinned_sumw2, rtol=rtol, atol=atol):
                raise PairSplitVinciaValidationError(
                    f"target-binned MESS+Vincia trijet {int(pt_low)} GeV sumw2 does not "
                    "match the rebinned harvested final-all reference"
                )
            validations.append(
                {
                    "pt_range_GeV": [pt_low, pt_high],
                    "target_gen_edges": list(target_edges),
                    "max_abs_sumw_difference": float(
                        np.max(np.abs(target_sumw - rebinned_sumw))
                    ),
                    "max_abs_sumw2_difference": float(
                        np.max(np.abs(target_sumw2 - rebinned_sumw2))
                    ),
                }
            )
    return {
        "applies": True,
        "channel": "trijet",
        "reference_path": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "source_coordinate_rebin_by_pt": validations,
        "rtol": rtol,
        "atol": atol,
    }


def _normalization_mask(edges: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    return (edges[:-1] >= low - 1e-9) & (edges[1:] <= high + 1e-9)


def _rebin_histogram_to_target_edges(
    values: np.ndarray,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Sum a source-coordinate histogram into nested target coordinate bins."""

    values = np.asarray(values, dtype=float)
    source_edges = np.asarray(source_edges, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)
    if len(values) != len(source_edges) - 1:
        raise ValueError("source histogram values do not match source edges")
    rebinned = np.zeros(len(target_edges) - 1, dtype=float)
    for index, (low, high) in enumerate(zip(target_edges[:-1], target_edges[1:])):
        selected = (source_edges[:-1] >= low - 1e-9) & (source_edges[1:] <= high + 1e-9)
        selected_edges = source_edges[:-1][selected]
        if not selected.any() or not np.isclose(selected_edges[0], low) or not np.isclose(
            source_edges[1:][selected][-1], high
        ):
            raise PairSplitVinciaValidationError(
                f"target bin [{low}, {high}] is not nested in the compiled reference edges"
            )
        rebinned[index] = values[selected].sum()
    return rebinned


def derive_pairsplit_vincia_prediction(
    source: PairSplitVinciaSource,
    *,
    pt_edges: Sequence[float],
    gen_edges_by_pt: Sequence[Sequence[float]],
    normalization_window: tuple[float, float] = (-4.0, 0.0),
    grooming_mode: str = "groomed",
) -> PairSplitVinciaPrediction:
    """Histogram final-all event rows on the exact current truth binning."""

    if grooming_mode not in {"groomed", "ungroomed"}:
        raise ValueError("grooming_mode must be 'groomed' or 'ungroomed'")
    mass_column = 2 if grooming_mode == "groomed" else 1

    pt_edges_array = np.asarray(pt_edges, dtype=float)
    edges_by_pt = tuple(tuple(float(value) for value in edges) for edges in gen_edges_by_pt)
    if pt_edges_array.ndim != 1 or len(pt_edges_array) != len(edges_by_pt) + 1:
        raise ValueError("pT edges must bound exactly one truth-edge sequence per pT slice")
    if not np.all(np.diff(pt_edges_array) > 0):
        raise ValueError("pT edges must be strictly increasing")
    density_by_pt: list[np.ndarray] = []
    stat_unc_by_pt: list[np.ndarray] = []
    stat_covariance_by_pt: list[np.ndarray] = []
    sumw_by_pt: list[np.ndarray] = []
    sumw2_by_pt: list[np.ndarray] = []
    totals_by_pt: list[float] = []
    masks_by_pt: list[np.ndarray] = []
    for index, edge_values in enumerate(edges_by_pt):
        edges = np.asarray(edge_values, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"invalid truth edges for pT slice {index}")
        sumw = np.zeros(len(edges) - 1, dtype=float)
        sumw2 = np.zeros_like(sumw)
        for ht_bin, rows in source.rows_by_ht.items():
            pt = rows[:, 0]
            jet_mass = rows[:, mass_column]
            if np.any(jet_mass < -NUMERICAL_MASS_TOLERANCE_GEV):
                raise PairSplitVinciaValidationError(
                    f"{grooming_mode} mass below audited numerical tolerance in {ht_bin}"
                )
            # The MESS audit allows only bounded FastJet numerical residuals.
            # They represent zero mass and are excluded from this logarithmic coordinate.
            jet_mass = np.where(jet_mass < 0.0, 0.0, jet_mass)
            valid_coordinate = (pt > 0.0) & (jet_mass > 0.0)
            coordinate = np.full(len(rows), np.nan, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                coordinate[valid_coordinate] = 2.0 * np.log10(
                    jet_mass[valid_coordinate] / (pt[valid_coordinate] * JET_RADIUS)
                )
            in_pt = (
                (pt >= pt_edges_array[index])
                & (pt < pt_edges_array[index + 1])
                & np.isfinite(coordinate)
            )
            row_weight = source.row_weights_by_ht[ht_bin]
            sumw += np.histogram(coordinate[in_pt], bins=edges, weights=row_weight[in_pt])[0]
            sumw2 += np.histogram(
                coordinate[in_pt], bins=edges, weights=row_weight[in_pt] ** 2
            )[0]
        mask = _normalization_mask(edges, normalization_window)
        total = float(sumw[mask].sum())
        if total <= 0.0:
            raise PairSplitVinciaValidationError(
                f"MESS+Vincia has nonpositive shown normalization in pT slice {index}"
            )
        widths = np.diff(edges)
        density_by_pt.append(sumw / widths / total)
        # Propagate the same finite-sample fluctuations through numerator and
        # normalization denominator. Available sumw2 is per jet; event-level
        # and cross-pT correlations cannot be recovered from these rows.
        covariance = normalized_prediction_covariance(sumw, np.diag(sumw2), widths, mask)
        stat_covariance_by_pt.append(covariance)
        stat_unc_by_pt.append(np.sqrt(np.clip(np.diag(covariance), 0.0, None)))
        sumw_by_pt.append(sumw)
        sumw2_by_pt.append(sumw2)
        totals_by_pt.append(total)
        masks_by_pt.append(mask)
    return PairSplitVinciaPrediction(
        source=source,
        grooming_mode=grooming_mode,
        pt_edges=tuple(float(value) for value in pt_edges_array),
        gen_edges_by_pt=edges_by_pt,
        density_by_pt=tuple(density_by_pt),
        stat_unc_by_pt=tuple(stat_unc_by_pt),
        stat_covariance_by_pt=tuple(stat_covariance_by_pt),
        sumw_by_pt=tuple(sumw_by_pt),
        sumw2_by_pt=tuple(sumw2_by_pt),
        normalization_totals_by_pt=tuple(totals_by_pt),
        normalization_masks_by_pt=tuple(masks_by_pt),
        normalization_window=tuple(float(value) for value in normalization_window),
    )


def attach_pairsplit_vincia_prediction(unfolder, prediction: PairSplitVinciaPrediction) -> None:
    """Attach a prevalidated prediction to the shared core plotting workflow."""

    core_pt_edges = tuple(float(value) for value in np.asarray(unfolder.pt_edges, dtype=float))
    core_gen_edges = tuple(
        tuple(float(value) for value in np.asarray(edges, dtype=float))
        for edges in unfolder.gen_edges_by_pt
    )
    if prediction.pt_edges != core_pt_edges or prediction.gen_edges_by_pt != core_gen_edges:
        raise PairSplitVinciaValidationError(
            "MESS+Vincia prediction binning does not match the current Unfolder"
        )
    # The core checks this explicit flag before considering its legacy Z+jet
    # cache, so a missing pair-split prediction cannot silently fall back.
    unfolder.pairsplit_vincia_required = True
    unfolder.pairsplit_vincia_prediction = prediction
    unfolder.vincia_stat_covariance_by_pt = prediction.stat_covariance_by_pt
