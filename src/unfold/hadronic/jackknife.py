"""Run-2 delete-one-group statistics with the nominal fake fraction fixed.

Data and MC are independent samples. Their covariance matrices are estimated
separately and added, never by pairing equally numbered data and MC groups.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle

import hist
import numpy as np

from unfold.binning import Binning
from unfold.inputs import prepared_inputs
from unfold.hadronic.inputs import (
    LEGACY_HISTOGRAM_KEYS, HADRONIC_ERAS, HADRONIC_FINE_AXES,
    HadronicModeArrays, HadronicRun2Inputs, prepare_hadronic_inputs,
)

AXES = {
    "reco": ("ptreco", "mpt_reco"),
    "gen": ("ptgen", "mpt_gen"),
    "response": ("ptreco", "mpt_reco", "ptgen", "mpt_gen"),
}


def covariance(replicas):
    """Delete-one-group covariance of already scaled/normalized estimates."""
    replicas = np.asarray(replicas, dtype=float)
    if replicas.ndim != 2 or len(replicas) < 2 or not np.isfinite(replicas).all():
        raise ValueError("Jackknife estimates must be a finite (groups >= 2, bins) array")
    delta = replicas - replicas.mean(axis=0)
    return (len(replicas) - 1) / len(replicas) * (delta.T @ delta)


def full_sample(replicas):
    # Values AND sumw2 count each event g-1 times. Histogram scaling by 1/(g-1)
    # would wrongly divide sumw2 by (g-1)^2.
    return {key: value.sum(axis=0) / (len(value) - 1) for key, value in replicas.items()}


def extract_histogram(histogram, role, mode):
    """Select labels explicitly: producer category order differs by histogram."""
    labels = sorted(histogram.axes["jk"])
    if labels != list(range(10)):
        raise ValueError(f"Expected the ten campaign jackknife labels 0..9, got {labels}")
    if list(histogram.axes["systematic"]) != ["nominal"]:
        raise ValueError("Jackknife histograms must contain nominal only")
    fine = HADRONIC_FINE_AXES[mode]
    expected = {"ptreco": fine.pt_edges, "ptgen": fine.pt_edges,
                "mpt_reco": fine.two_log10_rho_reco_edges,
                "mpt_gen": fine.two_log10_rho_gen_edges}
    for axis in AXES[role]:
        if not np.allclose(histogram.axes[axis].edges, expected[axis], rtol=0, atol=1e-12):
            raise ValueError(f"Jackknife {mode} {role} has incompatible {axis} edges")
    projected = histogram[{"systematic": "nominal"}].project("jk", *AXES[role])
    values, variances = [], []
    for label in labels:
        replica = projected[{"jk": hist.loc(label)}]
        variance = replica.variances(flow=True)
        if (not np.isfinite(replica.values(flow=True)).all() or variance is None
                or not np.isfinite(variance).all() or np.any(variance < 0)):
            raise ValueError(f"Invalid jackknife values/sumw2 in {mode} {role} label {label}")
        values.append(replica.values())
        variances.append(replica.variances())
    return np.asarray(values), np.asarray(variances)


@dataclass
class JackknifeInputs:
    channel: str
    mode: str
    data: dict
    mc: dict
    metadata: dict

    def prepare(self, spec, variant, measured_covariance, *, label=None):
        """Use the production binning adapter, retaining the fixed full fit metric."""
        mc = full_sample(self.mc) if label is None else {k: v[label] for k, v in self.mc.items()}
        data = full_sample(self.data) if label is None else {k: v[label] for k, v in self.data.items()}
        source = HadronicModeArrays(
            grooming_mode=self.mode, fine_axes=HADRONIC_FINE_AXES[self.mode],
            systematics=("nominal",),
            **{f"{role}_by_systematic": {"nominal": mc[role]} for role in AXES},
            **{f"{role}_variance_by_systematic": {"nominal": mc[role + "_variance"]} for role in AXES},
            nominal_data=data["reco"], nominal_data_variance=data["reco_variance"],
            # The adapter needs a fine-bin covariance. This diagonal is not
            # used as the fit metric; the full nominal covariance is supplied below.
            nominal_data_covariance=np.diag(data["reco_variance"].ravel()),
        )
        bundle = HadronicRun2Inputs(
            channel=self.channel, eras=HADRONIC_ERAS, modes={self.mode: source},
            source_files=(), observable_metadata={"data_covariance_source_by_mode": {
                self.mode: "fixed nominal full covariance"}},
        )
        adapted = prepare_hadronic_inputs(bundle, variant, ("nominal",), grooming_mode=self.mode)
        b = adapted.analysis_binning
        binning = Binning(pt_edges=b.pt_edges, reco_edges=b.two_log10_rho_reco_edges,
                          gen_edges=b.two_log10_rho_gen_edges,
                          reco_edges_by_pt=b.reco_two_log10_rho_edges_by_pt,
                          gen_edges_by_pt=b.gen_two_log10_rho_edges_by_pt)
        return prepared_inputs(
            spec, self.mode == "groomed", binning,
            mc_inputs=adapted.mc_inputs, data_inputs=adapted.data_inputs,
            systematics=("nominal",), measured_covariance=measured_covariance,
            first_reported_pt_bin=adapted.first_reported_pt_bin,
        )


def load_jackknife_inputs(root, channel, mode, *, requested="jackknife"):
    """Fall back only for absent files; malformed existing inputs are errors."""
    root = Path(root).expanduser().resolve()
    metadata = {"requested": requested, "resolved": "analytic", "input_root": str(root)}
    if requested == "analytic":
        return None, metadata
    if requested != "jackknife":
        raise ValueError(f"Unknown statistical method: {requested}")
    paths = [(kind, era, root / kind / f"rho_jk_{channel}_{sample}_{era}.pkl")
             for kind, sample in (("data", "data"), ("mc", "mg_pythia8"))
             for era in HADRONIC_ERAS]
    missing = [str(path) for _, _, path in paths if not path.exists()]
    if missing:
        metadata.update(fallback_reason="required replica files absent", missing_files=missing)
        return None, metadata
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    expected = {row["path"]: row for row in manifest.values()}
    merged = {"data": {}, "mc": {}}
    records = []
    for kind, era, path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            handle.seek(0)
            reference = expected.get(str(path.relative_to(root)))
            if reference and (digest.hexdigest() != reference["sha256"]
                              or path.stat().st_size != reference["bytes"]):
                raise ValueError(f"Jackknife file differs from its harvest manifest: {path}")
            payload = pickle.load(handle)
        for role in (("reco",) if kind == "data" else AXES):
            values, variances = extract_histogram(payload[LEGACY_HISTOGRAM_KEYS[mode][role]], role, mode)
            for key, array in ((role, values), (role + "_variance", variances)):
                merged[kind][key] = merged[kind].get(key, 0) + array
        records.append({"kind": kind, "era": era, "path": str(path), "sha256": digest.hexdigest(),
                        "coverage_issues": reference.get("coverage_issues", []) if reference else []})
    metadata.update(resolved="jackknife", groups=10, labels=list(range(10)), files=records,
                    fake_fraction="fixed nominal in both data and MC replicas",
                    mc_variation="response and GEN/misses together",
                    fit_covariance="fixed nominal full measured covariance",
                    normalization="normalize each replica before estimating covariance",
                    covariance="(g-1)/g * sum of centered outer products; independent data + MC",
                    absolute_data_scale="g/(g-1)", tau="fixed to the nominal fit",
                    grouping="producer chunk-local modulo ten")
    return JackknifeInputs(channel, mode, merged["data"], merged["mc"], metadata), metadata


def check_data_sample(nominal, reconstructed):
    """Reject a different data cohort instead of silently borrowing its covariance."""
    if not np.allclose(nominal, reconstructed, rtol=1e-7, atol=1e-7):
        difference = np.abs(nominal - reconstructed).sum() / np.abs(nominal).sum()
        raise ValueError(
            f"Jackknife and nominal data samples differ (relative L1 difference {difference:.2%}). "
            "Use matching nominal/replica inputs, or --stat-method analytic. "
            "No jackknife covariance was applied to the different nominal sample."
        )


def apply_jackknife(unfolder, replicas, variant):
    """Install selected statistics before the engine builds its bands and totals."""
    from unfold.engine import Unfolder

    spec = replace(unfolder.spec, tau=float(unfolder.tau or 0.0), model_envelope=False)
    nominal = unfolder.inputs
    estimates = {kind: [] for kind in ("data", "matrix")}
    normalized = {kind: [] for kind in estimates}
    global_normalized = {kind: [] for kind in estimates}
    groups = replicas.metadata["groups"]
    for label in range(groups):
        varied = replicas.prepare(spec, variant, nominal.measured_covariance, label=label)
        # Data replicas see the same production MC and fake survival. MC
        # replicas vary response and misses coherently, keeping nominal data.
        data_inputs = replace(nominal, systematics=["nominal"], mosaic_2d=varied.mosaic_2d)
        mc_inputs = replace(
            varied, mosaic_2d=nominal.mosaic_2d,
            fake_fraction_2d=nominal.fake_fraction_2d,
            fake_fraction_2d_dict=nominal.fake_fraction_2d_dict,
            fake_survival_dict=nominal.fake_survival_dict,
            corrected_measured_covariance_dict=nominal.corrected_measured_covariance_dict,
        )
        for kind, inputs in (("data", data_inputs), ("matrix", mc_inputs)):
            fit = Unfolder(inputs, spec, unfolder.groomed)
            fit._perform_unfold()
            fit._normalize_result()
            scale = groups / (groups - 1) if kind == "data" else 1.0
            estimates[kind].append(fit.y_unf * scale)
            normalized[kind].append(np.concatenate([r["unfolded"] for r in fit.normalized_results]))
            widths = np.concatenate([np.diff(e) for e in fit.gen_edges_by_pt])
            global_normalized[kind].append(fit.y_unf / fit.y_unf.sum() / widths)
        print(f"Jackknife: completed data and fixed-fake MC group {label + 1}/{groups}", flush=True)
    jacobian = unfolder._normalization_jacobian()
    analytic = {"input": unfolder.cov_data_np.copy(), "matrix": unfolder.cov_uncorr_np.copy()}
    absolute = {kind: covariance(values) for kind, values in estimates.items()}
    norm = {kind: covariance(values) for kind, values in normalized.items()}
    unfolder.jackknife_normalized_covariances = (norm["data"], norm["matrix"])
    unfolder.cov_data_np = absolute["data"]
    unfolder.cov_uncorr_np = absolute["matrix"]
    unfolder.cov_np = absolute["data"] + absolute["matrix"]
    unfolder.ye_unf = np.sqrt(np.clip(np.diag(unfolder.cov_np), 0, None))
    unfolder.stat_uncertainty_method = "fixed-fake jackknife"
    unfolder.jackknife_global_normalized_covariance = sum(covariance(x) for x in global_normalized.values())
    unfolder.jackknife_artifact_arrays = {
        **{f"analytic_{key}_covariance": value for key, value in analytic.items()},
        **{f"analytic_normalized_{key}_covariance": jacobian @ value @ jacobian.T
           for key, value in analytic.items()},
        **{f"replicas_{key}_absolute": np.asarray(value) for key, value in estimates.items()},
        **{f"replicas_{key}_normalized": np.asarray(value) for key, value in normalized.items()},
        "covariance_global_normalized": unfolder.jackknife_global_normalized_covariance,
    }
    replicas.metadata["covariance_ranks"] = {
        "data": int(np.linalg.matrix_rank(norm["data"])),
        "matrix": int(np.linalg.matrix_rank(norm["matrix"])),
        "stat": int(np.linalg.matrix_rank(norm["data"] + norm["matrix"])),
    }
