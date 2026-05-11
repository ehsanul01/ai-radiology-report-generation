from __future__ import annotations
import re
from typing import List, Sequence
import numpy as np

CHEXBERT_LABELS = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
                   "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis",
                   "Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]

_KEYWORDS = {
    "No Finding": ["no acute","normal chest","no abnormalit","unremarkable"],
    "Enlarged Cardiomediastinum": ["enlarged cardiomediastinum","widened mediastinum"],
    "Cardiomegaly": ["cardiomegaly","enlarged cardiac","enlarged heart"],
    "Lung Opacity": ["opacity","opacification"],
    "Lung Lesion": ["lung lesion","pulmonary nodule","pulmonary mass"],
    "Edema": ["pulmonary edema","edema"],
    "Consolidation": ["consolidation"],
    "Pneumonia": ["pneumonia"],
    "Atelectasis": ["atelectasis","atelectatic"],
    "Pneumothorax": ["pneumothorax"],
    "Pleural Effusion": ["pleural effusion","effusion"],
    "Pleural Other": ["pleural thickening","pleural scar"],
    "Fracture": ["fracture","rib fracture"],
    "Support Devices": ["endotracheal tube","chest tube","central line","pacemaker","picc line"],
}

class RuleBasedCheXLabeler:
    def __init__(self, labels=CHEXBERT_LABELS):
        self.labels = list(labels)
    def __call__(self, reports):
        out = np.zeros((len(reports), len(self.labels)), dtype=np.int8)
        for i, report in enumerate(reports):
            text = report.lower()
            for j, label in enumerate(self.labels):
                for kw in _KEYWORDS.get(label, []):
                    if kw in text:
                        idx = text.find(kw)
                        prefix = text[max(0,idx-12):idx]
                        if any(neg in prefix for neg in ["no ","without ","negative for ","not "]):
                            continue
                        out[i,j] = 1; break
        return out
