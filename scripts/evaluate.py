"""Compute NLP and clinical metrics from a predictions file.

Usage
-----
    python scripts/evaluate.py \
        --pred-file outputs/lora_preds.jsonl \
        --out-file outputs/lora_metrics.json

    # with paired significance test against a baseline predictions file:
    python scripts/evaluate.py \
        --pred-file outputs/lora_preds.jsonl \
        --compare-file outputs/zero_shot_preds.jsonl \
        --paired \
        --out-file outputs/paired_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chexbert_labeler import (  # noqa: E402
    CHEXBERT_LABELS, RuleBasedCheXLabeler, label_reports_chexbert,
)
from src.hallucination import score_errors  # noqa: E402
from src.metrics import (  # noqa: E402
    compute_bertscore, compute_bleu, compute_chexbert_f1,
    compute_rouge_l, paired_t_test,
)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def _load_preds(path: Path):
    refs, hyps, meta = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            refs.append(
                f"Findings: {obj['reference_findings']} "
                f"Impression: {obj['reference_impression']}"
            )
            hyps.append(obj["prediction"])
            meta.append(obj)
    return refs, hyps, meta


def _get_labeler(args):
    if args.chexbert_repo and args.chexbert_ckpt:
        def labeler(reports):
            return label_reports_chexbert(
                reports,
                chexbert_repo=args.chexbert_repo,
                checkpoint=args.chexbert_ckpt,
            )
        return labeler
    print("NOTE: CheXBert repo/checkpoint not provided — using rule-based "
          "fallback. For publication-quality results install CheXBert and "
          "pass --chexbert-repo / --chexbert-ckpt.")
    rb = RuleBasedCheXLabeler()
    return lambda reports: rb(reports)


# ------------------------------------------------------------------
# Single-file evaluation
# ------------------------------------------------------------------


def evaluate_single(refs, hyps, labeler):
    bleu = compute_bleu(refs, hyps)
    rouge = compute_rouge_l(refs, hyps)
    bert = compute_bertscore(refs, hyps)

    ref_labels = labeler(refs)
    hyp_labels = labeler(hyps)
    chex = compute_chexbert_f1(ref_labels, hyp_labels, CHEXBERT_LABELS)
    errors = score_errors(ref_labels, hyp_labels, refs, hyps).to_dict()

    return {
        "n": len(refs),
        "bleu_1": bleu["bleu_1"],
        "bleu_4": bleu["bleu_4"],
        "rouge_l_f1": rouge["rouge_l_f1"],
        "bertscore_f1": bert["bertscore_f1"],
        "chexbert_macro_f1": chex["chexbert_macro_f1"],
        "chexbert_per_class_f1": chex["chexbert_per_class_f1"],
        "hallucination_counts": errors,
        "_per_example": {
            "rouge_l_f1": rouge["rouge_l_f1_per_example"],
            "bertscore_f1": bert["bertscore_f1_per_example"],
        },
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(args) -> None:
    refs, hyps, _meta = _load_preds(Path(args.pred_file))
    print(f"Loaded {len(refs)} predictions")
    labeler = _get_labeler(args)

    primary = evaluate_single(refs, hyps, labeler)
    summary = {"primary": primary}

    if args.compare_file:
        refs_b, hyps_b, _ = _load_preds(Path(args.compare_file))
        # Sanity check: alignment by position. In practice we align
        # by (subject_id, study_id); the predictions files are produced
        # by the same loader with the same subset_size, so positions
        # match. We still refuse to compare mismatched lengths.
        assert len(refs_b) == len(refs), (
            "predictions files differ in length; cannot run paired test"
        )
        baseline = evaluate_single(refs_b, hyps_b, labeler)
        summary["baseline"] = baseline

        if args.paired:
            paired = {
                "rouge_l_f1": paired_t_test(
                    baseline["_per_example"]["rouge_l_f1"],
                    primary["_per_example"]["rouge_l_f1"],
                ),
                "bertscore_f1": paired_t_test(
                    baseline["_per_example"]["bertscore_f1"],
                    primary["_per_example"]["bertscore_f1"],
                ),
            }
            summary["paired_ttest"] = paired

    # Strip per-example arrays from the saved summary to keep it readable.
    for k in ("primary", "baseline"):
        if k in summary:
            summary[k].pop("_per_example", None)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--compare-file", default=None)
    p.add_argument("--paired", action="store_true",
                   help="Run paired t-test between pred-file and compare-file.")
    p.add_argument("--chexbert-repo", default=None)
    p.add_argument("--chexbert-ckpt", default=None)
    main(p.parse_args())
