"""Inference with a fine-tuned LoRA adapter.

Usage
-----
    python scripts/run_inference.py \
        --config configs/default.yaml \
        --adapter-path checkpoints/lora-rank16 \
        --split test \
        --n-samples 200 \
        --out-file outputs/lora_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import MIMICCXRDataset  # noqa: E402
from src.model import load_for_inference  # noqa: E402
from src.prompts import TRAIN_PROMPT  # noqa: E402


def run(args) -> None:
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    split_to_file = {"train": "train.jsonl", "val": "val.jsonl",
                     "test": "test.jsonl"}
    manifest = Path(cfg["data"]["manifest_dir"]) / split_to_file[args.split]
    subset_key = {"train": "train_subset", "val": "val_subset",
                  "test": "test_subset"}[args.split]
    subset = args.n_samples or cfg["data"][subset_key]

    dataset = MIMICCXRDataset(manifest, subset=subset)
    print(f"Loaded {len(dataset)} examples from {manifest}")

    print(f"Loading model with adapter {args.adapter_path} ...")
    model, processor = load_for_inference(
        base_model_name=cfg["model"]["base_model"],
        adapter_path=args.adapter_path,
        load_in_4bit=cfg["model"]["load_in_4bit"],
        compute_dtype=cfg["model"]["bnb_4bit_compute_dtype"],
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        for idx in tqdm(range(len(dataset)), desc="lora-inference"):
            ex = dataset[idx]
            inputs = processor(
                images=ex["image"],
                text=TRAIN_PROMPT,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg["infer"]["max_new_tokens"],
                    do_sample=cfg["infer"]["do_sample"],
                    num_beams=cfg["infer"]["num_beams"],
                )
            decoded = processor.batch_decode(
                out_ids, skip_special_tokens=True
            )[0]
            pred = decoded.split("ASSISTANT:")[-1].strip()
            fh.write(json.dumps({
                "subject_id": ex["subject_id"],
                "study_id": ex["study_id"],
                "reference_findings": ex["findings"],
                "reference_impression": ex["impression"],
                "prediction": pred,
                "method": "lora",
            }) + "\n")

    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--out-file", required=True)
    run(p.parse_args())
