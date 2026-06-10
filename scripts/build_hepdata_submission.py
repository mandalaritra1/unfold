"""Generate a HEPData submission directory + archive from the exporter .npz files.

Reads ``outputs/rho/hepdata/hepdata_export_{groomed,ungroomed}.npz`` (and the
JSON manifests) and writes a ``submission/`` directory containing
``submission.yaml`` plus one YAML table per (grooming, pT bin) measurement and
per (grooming, kind) correlation matrix, then packs it into
``hepdata_submission.tar.gz``.

Usage:
    .venv/bin/python scripts/build_hepdata_submission.py [--root outputs/rho/hepdata]
"""
import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml

CME_GEV = 13000


def r(x, sig=6):
    """Round to `sig` significant figures, returning a clean python float."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return 0.0
    return float(f"{x:.{sig}g}")


def pt_label(pt_low, pt_high):
    if pt_high is None:
        return f"{int(pt_low)}-Inf"
    return f"{int(pt_low)}-{int(pt_high)}"


def pt_qual(pt_low, pt_high):
    if pt_high is None:
        return f"> {int(pt_low)} GeV"
    return f"{int(pt_low)}-{int(pt_high)} GeV"


def measurement_table(d, man, bin_info):
    """One Histo1D-style table: normalized xsec vs log10(rho^2) for one pT bin."""
    i = bin_info["out_index"]
    edges = d[f"pt{i}__edges"]
    value = d[f"pt{i}__value"]
    stat = d[f"pt{i}__stat"]
    su = d[f"pt{i}__syst_up"]
    sd = d[f"pt{i}__syst_down"]
    truth = d[f"pt{i}__true_pythia"]
    grooming = man["grooming"]

    indep = {
        "header": {"name": "$\\log_{10}(\\rho^2)$, $\\rho = m/(p_T R)$"},
        "values": [{"low": r(edges[k]), "high": r(edges[k + 1])} for k in range(len(value))],
    }

    common_quals = [
        {"name": "SQRT(S)", "value": CME_GEV, "units": "GeV"},
        {"name": "Jet $p_T$", "value": pt_qual(bin_info["pt_low"], bin_info["pt_high"])},
        {"name": "Grooming", "value": grooming},
        {"name": "Jet algorithm", "value": "anti-$k_T$, R=0.8"},
    ]

    dep_data = {
        "header": {"name": "$(1/\\sigma)\\, d\\sigma/d\\log_{10}(\\rho^2)$"},
        "qualifiers": common_quals,
        "values": [],
    }
    for k in range(len(value)):
        dep_data["values"].append({
            "value": r(value[k]),
            "errors": [
                {"label": "stat", "symerror": r(stat[k])},
                {"label": "sys", "asymerror": {"plus": r(su[k]), "minus": r(-sd[k])}},
            ],
        })

    dep_theory = {
        "header": {"name": "PYTHIA8 (prediction)"},
        "qualifiers": common_quals,
        "values": [{"value": r(truth[k])} for k in range(len(value))],
    }

    return {"independent_variables": [indep],
            "dependent_variables": [dep_data, dep_theory]}


def global_bin_labels(d, man):
    """Self-describing label per flattened global bin: pT range + rho range."""
    labels = []
    for b in man["published_pt_bins"]:
        i = b["out_index"]
        edges = d[f"pt{i}__edges"]
        ptl = pt_label(b["pt_low"], b["pt_high"])
        for k in range(len(edges) - 1):
            labels.append(f"pt[{ptl}] log10rho2[{r(edges[k])},{r(edges[k+1])}]")
    return labels


def matrix_table(matrix, labels, kind, grooming):
    n = len(labels)
    iv1 = {"header": {"name": "Bin i (pT, log10(rho^2))"}, "values": []}
    iv2 = {"header": {"name": "Bin j (pT, log10(rho^2))"}, "values": []}
    dep = {"header": {"name": f"{kind} correlation"},
           "qualifiers": [{"name": "SQRT(S)", "value": CME_GEV, "units": "GeV"},
                          {"name": "Grooming", "value": grooming}],
           "values": []}
    for a in range(n):
        for b in range(n):
            iv1["values"].append({"value": labels[a]})
            iv2["values"].append({"value": labels[b]})
            dep["values"].append({"value": r(matrix[a, b])})
    return {"independent_variables": [iv1, iv2], "dependent_variables": [dep]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/rho/hepdata")
    ap.add_argument("--out", default=None, help="submission dir (default <root>/submission)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "submission"
    out.mkdir(parents=True, exist_ok=True)

    docs = []  # submission.yaml documents (header first, then one per table)

    header = {
        "comment": (
            "Normalized differential cross sections for the leading AK8 (anti-kT, R=0.8) "
            "jet in Z(->ll)+jet events at sqrt(s)=13 TeV (CMS Run 2). The observable is "
            "log10(rho^2) with rho = m/(pT R), measured separately for groomed (Soft Drop "
            "beta=0, zcut=0.1) and ungroomed jet mass, in three leading-jet pT bins "
            "(200-290, 290-400, >400 GeV). Each distribution is normalized to unit area "
            "within its pT bin. Statistical uncertainties are from a 10-way jackknife; "
            "systematic uncertainties are the quadrature sum of all sources. Bin-to-bin "
            "correlation matrices (statistical and total) are provided as separate tables."
        ),
    }
    docs.append(header)

    table_files = []

    for mode in ["groomed", "ungroomed"]:
        npz = root / f"hepdata_export_{mode}.npz"
        man = json.loads((root / f"hepdata_export_{mode}.json").read_text())
        d = np.load(npz)
        labels = global_bin_labels(d, man)

        keywords = [
            {"name": "reactions", "values": ["P P --> Z0 < LEPTON+ LEPTON- > JET X"]},
            {"name": "observables", "values": ["DSIG/DLOG10(RHO**2)"]},
            {"name": "cmenergies", "values": [CME_GEV]},
            {"name": "phrases", "values": ["Inclusive", "Z production", "Jet Production",
                                            "Jet mass", "Jet substructure",
                                            "Differential cross section"]},
        ]

        # measurement tables (one per pT bin)
        for b in man["published_pt_bins"]:
            tbl = measurement_table(d, man, b)
            fname = f"table_{mode}_pt{b['out_index']}.yaml"
            (out / fname).write_text(yaml.safe_dump(tbl, sort_keys=False, allow_unicode=True))
            table_files.append(fname)
            docs.append({
                "name": f"{mode.capitalize()} {pt_qual(b['pt_low'], b['pt_high'])}",
                "location": f"Normalized cross section vs log10(rho^2), {mode}, "
                            f"pT {pt_qual(b['pt_low'], b['pt_high'])}",
                "description": (
                    f"Normalized differential cross section (1/sigma) dsigma/dlog10(rho^2) "
                    f"for the {mode} leading AK8 jet, {pt_qual(b['pt_low'], b['pt_high'])}. "
                    f"rho = m/(pT R), R=0.8. Errors: stat (jackknife) and sys (total)."
                ),
                "keywords": keywords,
                "data_file": fname,
            })

        # correlation matrices (stat + total)
        for kind, key in [("Statistical", "corr_stat_norm"), ("Total", "corr_total_norm")]:
            tbl = matrix_table(d[key], labels, kind, mode)
            fname = f"table_{mode}_corr_{kind.lower()}.yaml"
            (out / fname).write_text(yaml.safe_dump(tbl, sort_keys=False, allow_unicode=True))
            table_files.append(fname)
            docs.append({
                "name": f"{mode.capitalize()} {kind} correlation",
                "location": f"{kind} bin-to-bin correlation matrix, {mode}, all pT bins",
                "description": (
                    f"{kind} correlation matrix for the {mode} normalized cross section, "
                    f"flattened over the three pT bins (order: increasing pT, then "
                    f"increasing log10(rho^2)). Self-describing bin labels included."
                ),
                "keywords": keywords,
                "data_file": fname,
            })

    sub_path = out / "submission.yaml"
    with sub_path.open("w") as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, allow_unicode=True,
                           explicit_start=True, default_flow_style=False)

    # package
    archive = root / "hepdata_submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(sub_path, arcname="submission.yaml")
        for f in table_files:
            tar.add(out / f, arcname=f)

    print("submission dir:", out)
    print("tables:", len(table_files))
    print("archive:", archive)
    print("archive size: %.1f KB" % (archive.stat().st_size / 1024))


if __name__ == "__main__":
    main()
