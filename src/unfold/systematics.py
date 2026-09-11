"""Systematic-source bookkeeping shared by every channel.

Tables (JES correlations across years, luminosities) and the small name
helpers (Up/Down splitting, display labels, summary groups) live here so that
no other module carries its own copy.
"""

from __future__ import annotations

import re

import numpy as np

# UL Run-2 integrated luminosities in fb^-1 (2016 = post-VFP, 2016APV = pre-VFP).
RUN2_LUMI = {"2016APV": 19.52, "2016": 16.81, "2017": 41.48, "2018": 59.83}
RUN2_LUMI_TOTAL = 138.0        # rounded value printed in the CMS label
COM_TEV = 13.0

# Per-year producer dataset names of the Z+jet skims, keyed by era.
ZJET_ERA_DATASETS = {
    "2016": "pythia_UL16NanoAODv9",
    "2016APV": "pythia_UL16NanoAODAPVv9",
    "2017": "pythia_UL17NanoAODv9",
    "2018": "pythia_UL18NanoAODv9",
}

# Era groups for the JES year-correlation split: the two 2016 halves move
# together.
RUN2_ERA_GROUPS = {
    "2016": ("2016APV", "2016"),
    "2017": ("2017",),
    "2018": ("2018",),
}

# Correlation of each JES source between years (JetMET Run-2 prescription).
JES_RUN2_CORRELATIONS = {
    "AbsoluteMPFBias": 1.0,
    "AbsoluteScale": 1.0,
    "AbsoluteStat": 0.0,
    "FlavorQCD": 1.0,
    "Fragmentation": 1.0,
    "PileUpDataMC": 0.5,
    "PileUpPtBB": 0.5,
    "PileUpPtEC1": 0.5,
    "PileUpPtEC2": 0.5,
    "PileUpPtHF": 0.5,
    "PileUpPtRef": 0.5,
    "RelativeFSR": 0.5,
    "RelativeJEREC1": 0.0,
    "RelativeJEREC2": 0.0,
    "RelativeJERHF": 0.5,
    "RelativePtBB": 0.5,
    "RelativePtEC1": 0.0,
    "RelativePtEC2": 0.0,
    "RelativePtHF": 0.5,
    "RelativeBal": 0.5,
    "RelativeSample": 0.0,
    "RelativeStatEC": 0.0,
    "RelativeStatFSR": 0.0,
    "RelativeStatHF": 0.0,
    "SinglePionECAL": 1.0,
    "SinglePionHCAL": 1.0,
    "TimePtEta": 0.0,
}
JER_RUN2_CORRELATION = 0.0

# Producer-side JES sources known to be broken in the hadronic skims.
DEFECTIVE_JES_SOURCES = ("RelativeJEREC1", "RelativeJEREC2", "RelativeJERHF")

JES_SOURCES = tuple(JES_RUN2_CORRELATIONS)
JES_SYSTEMATICS = tuple(f"JES_{s}{d}" for s in JES_SOURCES for d in ("Up", "Down")) + ("JERUp", "JERDown")
ZJET_NON_JES_SYSTEMATICS = (
    "nominal", "puUp", "puDown", "elerecoUp", "elerecoDown", "eleidUp", "eleidDown",
    "eletrigUp", "eletrigDown", "murecoUp", "murecoDown", "muidUp", "muidDown",
    "mutrigUp", "muisoUp", "muisoDown", "mutrigDown", "pdfUp", "pdfDown", "q2Up", "q2Down",
    "l1prefiringUp", "l1prefiringDown", "isrUp", "isrDown", "fsrUp", "fsrDown",
    "JMRUp", "JMRDown", "JMSUp", "JMSDown",
)


def era_split_coefficients(rho, prescription):
    """(correlated, uncorrelated) amplitudes for a source with year correlation ``rho``.

    ``"sqrt"`` is the JetMET prescription (the split is on variances, so the
    amplitudes are sqrt(rho) and sqrt(1-rho)); every channel uses it since
    2026-09-09.  ``"linear"`` uses rho and 1-rho directly: the Z+jet
    production before that date did this, which under-covers partially
    correlated sources (rho = 0.5 gives half the correct variance).  It is
    kept only to reproduce those outputs (``unfold zjet --era-split linear``).
    """
    if prescription == "sqrt":
        return float(np.sqrt(rho)), float(np.sqrt(1.0 - rho))
    if prescription == "linear":
        return float(rho), float(1.0 - rho)
    raise ValueError(f"unknown era-split prescription {prescription!r}")


_UPDOWN = re.compile(r"^(.*?)(Up|Down)(?:_.*)?$")


def split_updown(name):
    """'JES_AbsoluteScaleUp_corr' -> ('JES_AbsoluteScale', 'Up'); 'nominal' -> ('nominal', None)."""
    match = _UPDOWN.match(name)
    if match:
        return match.group(1), match.group(2)
    return name, None


# Pairs "<base>Up[_corr|_uncorr_YYYY]" with the matching Down key.  The
# Up/Down must sit immediately before the era suffix: a naive
# replace("Up", "Down") would corrupt e.g. JES_PileUpPtBB.
UPDOWN_KEY = re.compile(r"^(.+)(Up|Down)((?:_corr|_uncorr_\d+)?)$")


def partner_name(name):
    """The Down key for an Up key and vice versa, or None for unpaired names."""
    match = UPDOWN_KEY.match(name)
    if not match:
        return None
    base, direction, suffix = match.groups()
    other = "Down" if direction == "Up" else "Up"
    return f"{base}{other}{suffix}"


def group_name(name):
    """Coarse group used by the grouped uncertainty summary, or None."""
    s = name.lower()
    if s.startswith(("jes", "jer")):
        return "Jet Energy"
    if s.startswith(("jms", "jmr")):
        return "Jet Mass"
    if s.startswith(("ele", "mu")):
        return "Lepton SFs"
    if s.startswith("showermodel"):
        return "Shower Model"
    if s.startswith("hadmodel"):
        return "Hadronization Model"
    if s.startswith(("isr", "fsr")):
        return "Parton Shower"
    if s.startswith(("modelenvelope", "herwig")):
        return "Model Uncertainty"
    if s.startswith(("pu", "pdf", "q2", "l1prefiring")):
        return "Other Theory"
    return None


_LABELS = {
    "pu": "Pileup",
    "l1prefiring": "L1 Prefiring",
    "q2": r"Q$^2$ Scale",
    "pdf": "PDF",
    "herwig": "Model Uncertainty",
    "modelenvelope": "Model Uncertainty",
    "showermodel": "Shower Model",
    "hadmodel": "Hadronization Model",
    "isr": "ISR",
    "fsr": "FSR",
    "jms": "JMS",
    "jmr": "JMR",
}


def label(name):
    base, _ = split_updown(name)
    return _LABELS.get(base.lower(), base)


_SUMMARY = {
    "elereco": "Electron RECO",
    "eleid": "Electron ID",
    "eletrig": "Electron Trigger",
    "mureco": "Muon RECO",
    "muid": "Muon ID",
    "mutrig": "Muon Trigger",
    "muiso": "Muon ISO",
    "pu": "Pileup",
    "pdf": "PDF",
    "q2": "Q2 Scale",
    "l1prefiring": "L1 Prefiring",
    "herwig": "Model Uncertainty",
    "modelenvelope": "Model Uncertainty",
    "showermodel": "Shower Model",
    "hadmodel": "Hadronization Model",
}


def summary_name(name, grouped=False):
    """Name under which a source is summed in the uncertainty summaries."""
    if grouped:
        group = group_name(name)
        return group if group is not None else label(name)
    base, _ = split_updown(name)
    b = base.lower()
    for prefix, out in (("jes", "JES"), ("jer", "JER"), ("isr", "ISR"), ("fsr", "FSR"),
                        ("jmr", "JMR"), ("jms", "JMS")):
        if b.startswith(prefix):
            return out
    return _SUMMARY.get(b, label(name))
