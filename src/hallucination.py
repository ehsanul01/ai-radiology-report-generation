from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np

_SEVERITY_RE = re.compile(r"\b(mild|moderate|severe|large|small|minimal|trace)\b", re.IGNORECASE)
_LATERAL_RE = re.compile(r"\b(left|right|bilateral)\b", re.IGNORECASE)

@dataclass
class HallucinationCounts:
    fabricated: int = 0
    missed: int = 0
    severity: int = 0
    laterality: int = 0
    def to_dict(self):
        return {"FABRICATED":self.fabricated,"MISSED":self.missed,
                "SEVERITY":self.severity,"LATERALITY":self.laterality}

def score_errors(ref_labels, hyp_labels, ref_text, hyp_text):
    counts = HallucinationCounts()
    counts.fabricated = int(((ref_labels==0)&(hyp_labels==1)).sum())
    counts.missed = int(((ref_labels==1)&(hyp_labels==0)).sum())
    for ref, hyp in zip(ref_text, hyp_text):
        ref_sev = set(m.group(1).lower() for m in _SEVERITY_RE.finditer(ref))
        hyp_sev = set(m.group(1).lower() for m in _SEVERITY_RE.finditer(hyp))
        ref_lat = set(m.group(1).lower() for m in _LATERAL_RE.finditer(ref))
        hyp_lat = set(m.group(1).lower() for m in _LATERAL_RE.finditer(hyp))
        if ref_sev and hyp_sev and ref_sev != hyp_sev: counts.severity += 1
        if ref_lat and hyp_lat and ref_lat != hyp_lat: counts.laterality += 1
    return counts

def format_example_coding(ref, hyp, ref_labels, hyp_labels, label_names):
    fab = [label_names[i] for i in range(len(label_names)) if ref_labels[i]==0 and hyp_labels[i]==1]
    miss = [label_names[i] for i in range(len(label_names)) if ref_labels[i]==1 and hyp_labels[i]==0]
    return f"REF: {ref}\nHYP: {hyp}\nFabricated: {fab}\nMissed: {miss}"
