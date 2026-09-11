#!/usr/bin/env python3
"""Verify harvested hadronic jackknife files and extract label-aligned arrays.

Read-only with respect to source pickles. The outputs retain all ten replicas;
they are diagnostic inputs, not a replacement of the nominal production tag.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle

import hist
import numpy as np

from unfold.hadronic.inputs import (
    LEGACY_HISTOGRAM_KEYS, HADRONIC_FINE_AXES,
)

AXES = {
    "reco": ("ptreco", "mpt_reco"),
    "gen": ("ptgen", "mpt_gen"),
    "response": ("ptreco", "mpt_reco", "ptgen", "mpt_gen"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.input_root / "manifest.json").read_text())
    report = {
        "input_root": str(args.input_root),
        "replica_labels": list(range(10)),
        "convention": "ten delete-one-tenth replicas; full sumw normalization in MC",
        "limitations": [
            "Chunk-local modulo-10 partition, not invariant under rechunking.",
            "No full-sample jk=-1 histogram. Summing replica values or sumw2 and dividing by nine reconstructs their full-coverage averages, provided event processing is replica-independent.",
            "This audit does not establish replica-independent stochastic jet corrections.",
            "Accepted data coverage shortfalls are retained from the harvest manifest.",
        ],
        "files": [],
    }
    for name, metadata in manifest.items():
        path = args.input_root / metadata["path"]
        record = {"path": str(path), "bytes": path.stat().st_size,
                  "sha256": sha256(path), "histograms": {}}
        assert record["bytes"] == metadata["bytes"], path
        assert record["sha256"] == metadata["sha256"], path
        record["coverage_issues"] = metadata.get("coverage_issues", [])
        record["acceptance_status"] = metadata.get("acceptance_status", "see MC harvest")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        output = {}
        for mode, keys in LEGACY_HISTOGRAM_KEYS.items():
            fine = HADRONIC_FINE_AXES[mode]
            expected_edges = {"ptreco": fine.pt_edges, "ptgen": fine.pt_edges,
                              "mpt_reco": fine.two_log10_rho_reco_edges,
                              "mpt_gen": fine.two_log10_rho_gen_edges}
            for role in (("reco",) if metadata["sample_type"] == "data"
                         else ("reco", "gen", "response")):
                histogram = payload[keys[role]]
                labels = [int(value) for value in histogram.axes["jk"]]
                assert sorted(labels) == list(range(10)), (path, keys[role], labels)
                assert list(histogram.axes["systematic"]) == ["nominal"], path
                for axis in AXES[role]:
                    np.testing.assert_allclose(histogram.axes[axis].edges, expected_edges[axis],
                                               rtol=0, atol=1e-12)
                projected = histogram[{"systematic": "nominal"}].project("jk", *AXES[role])
                values, variances = [], []
                for label in range(10):
                    replica = projected[{"jk": hist.loc(label)}]
                    assert np.isfinite(replica.values(flow=True)).all(), path
                    variance = replica.variances(flow=True)
                    assert variance is not None and np.isfinite(variance).all(), path
                    assert (variance >= 0).all(), path
                    values.append(replica.values())
                    variances.append(replica.variances())
                values, variances = np.asarray(values), np.asarray(variances)
                sums = values.reshape(10, -1).sum(axis=1)
                assert np.all(sums > 0), path
                prefix = f"{mode}_{role}"
                output[prefix] = values
                output[f"{prefix}_variance"] = variances
                record["histograms"][keys[role]] = {
                    "stored_labels": labels, "shape": list(values.shape),
                    "replica_sumw": sums.tolist(),
                    "relative_replica_total_range": [float(sums.min()/sums.mean()),
                                                     float(sums.max()/sums.mean())],
                }
            record["histograms"][f"{mode}_event_covariance_present"] = keys["reco_covariance"] in payload
        result_path = args.output / f"{path.stem}.npz"
        np.savez_compressed(result_path, **output)
        record["extracted_arrays"] = str(result_path)
        record["extracted_sha256"] = sha256(result_path)
        report["files"].append(record)
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verified {name}", flush=True)
        del payload, histogram, projected, replica, output, values, variances
        gc.collect()
    assert len(report["files"]) == 16
    print("All 16 files: hashes, axes, finite values/sumw2 and replica labels verified.")


if __name__ == "__main__":
    main()
