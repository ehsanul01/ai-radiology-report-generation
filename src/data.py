from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset

_FINDINGS_RE = re.compile(r"FINDINGS\s*:\s*(.+?)(?=IMPRESSION\s*:|$)", re.DOTALL | re.IGNORECASE)
_IMPRESSION_RE = re.compile(r"IMPRESSION\s*:\s*(.+?)$", re.DOTALL | re.IGNORECASE)

def parse_report(raw_text):
    fm = _FINDINGS_RE.search(raw_text)
    im = _IMPRESSION_RE.search(raw_text)
    findings = re.sub(r"\s+", " ", fm.group(1).strip()) if fm else ""
    impression = re.sub(r"\s+", " ", im.group(1).strip()) if im else ""
    return {"findings": findings, "impression": impression}

@dataclass
class CXRRecord:
    image_path: str
    findings: str
    impression: str
    subject_id: str
    study_id: str
    split: str
    def to_json(self):
        return json.dumps({"image_path":self.image_path,"findings":self.findings,
                           "impression":self.impression,"subject_id":self.subject_id,
                           "study_id":self.study_id,"split":self.split})
    @classmethod
    def from_json(cls, line):
        return cls(**json.loads(line))

def read_manifest(path):
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(CXRRecord.from_json(line))
    return records

def write_manifest(records, path):
    with open(path, "w") as fh:
        for r in records:
            fh.write(r.to_json() + "\n")

class MIMICCXRDataset(Dataset):
    def __init__(self, manifest_path, subset=None, min_findings_len=5):
        all_records = read_manifest(manifest_path)
        records = [r for r in all_records if len(r.findings.split()) >= min_findings_len and len(r.impression) > 0]
        if subset is not None:
            records = records[:subset]
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        image = Image.open(r.image_path).convert("RGB")
        return {"image": image, "findings": r.findings, "impression": r.impression,
                "subject_id": r.subject_id, "study_id": r.study_id}
