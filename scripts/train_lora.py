"""QLoRA fine-tuning of LLaVA-1.5 on MIMIC-CXR report generation.

Usage
-----
    python scripts/train_lora.py --config configs/default.yaml

Design notes
------------
* We train only the LoRA adapters on the LLM (q_proj, v_proj). The
  vision encoder and the multimodal projector are frozen.
* We mask prompt tokens from the loss so that cross-entropy is
  computed over the report only.
* Mixed precision (bfloat16) + 4-bit weight quantization fit the 7B
  model on a 16 GB P100.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import MIMICCXRDataset  # noqa: E402
from src.model import LoRAConfig, load_base_model, wrap_with_lora  # noqa: E402
from src.prompts import TRAIN_PROMPT, format_target  # noqa: E402


# ------------------------------------------------------------------
# Collator
# ------------------------------------------------------------------


class LlavaCollator:
    """Batch raw (image, findings, impression) tuples into model inputs.

    Masks prompt tokens in the labels so that only the generated report
    contributes to the loss.
    """

    def __init__(self, processor, max_length: int = 512) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [ex["image"] for ex in batch]
        targets = [format_target(ex["findings"], ex["impression"])
                   for ex in batch]
        # Full sequence = prompt + target
        full_texts = [
            TRAIN_PROMPT.replace("<image>", "<image>") + " " + t
            for t in targets
        ]

        enc = self.processor(
            images=images,
            text=full_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()

        # Mask the prompt portion in each row so loss is target-only.
        prompt_ids = self.tokenizer(
            TRAIN_PROMPT.replace("<image>", "<image>") + " ",
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)
        labels[:, :prompt_len] = -100

        # Pad tokens -> -100 so they do not contribute to loss
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(args) -> None:
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    train_mf = Path(cfg["data"]["manifest_dir"]) / "train.jsonl"
    val_mf = Path(cfg["data"]["manifest_dir"]) / "val.jsonl"

    train_ds = MIMICCXRDataset(train_mf, subset=cfg["data"]["train_subset"])
    val_ds = MIMICCXRDataset(val_mf, subset=cfg["data"]["val_subset"])
    print(f"Train={len(train_ds)} | Val={len(val_ds)}")

    print("Loading base model ...")
    model, processor = load_base_model(
        model_name=cfg["model"]["base_model"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        compute_dtype=cfg["model"]["bnb_4bit_compute_dtype"],
    )

    lora_cfg = LoRAConfig(
        r=cfg["lora"]["r"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        target_modules=tuple(cfg["lora"]["target_modules"]),
        bias=cfg["lora"]["bias"],
    )
    model = wrap_with_lora(model, lora_cfg)

    collator = LlavaCollator(processor)

    training_args = TrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        eval_steps=cfg["train"]["eval_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=cfg["train"]["seed"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])
    print(f"Saved adapter to {cfg['train']['output_dir']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    main(p.parse_args())
