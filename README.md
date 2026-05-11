# Vision-Language Models for Automated Medical Report Generation

**Author:** Ehsanul Haque (`ehaque2@buffalo.edu`)  
**Course:** CSE 474 — Introduction to Machine Learning, Spring 2026  
**University:** University at Buffalo (SUNY)  
---

## Overview

This project investigates whether a general-purpose vision-language model (LLaVA-1.5) can be adapted to automatically generate radiology reports from chest X-ray images — a high-impact clinical task that could reduce the reporting backlog in radiology departments.

We compare two approaches:
- **Zero-shot prompting** — ask LLaVA-1.5 to write a report with no additional training
- **QLoRA fine-tuning** — adapt LLaVA-1.5 on 10,000 MIMIC-CXR chest X-ray / report pairs using rank-16 LoRA

We evaluate with both standard NLP metrics (BLEU, ROUGE-L, BERTScore) and a clinically-grounded metric (CheXBert-F1 over 14 clinical observations), plus a four-category hallucination coding scheme.

---

## Real Experimental Results (Zero-Shot, n=200 MIMIC-CXR test studies)

| Metric | Zero-Shot LLaVA-1.5 |
|--------|-------------------|
| BLEU-1 | 0.150 |
| BLEU-4 | 0.007 |
| ROUGE-L | 0.145 |
| BERTScore F1 | 0.744 |
| CheXBert macro-F1 | 0.028 |

### Hallucination Analysis (n=200)

| Error Type | Count | Per Report |
|-----------|-------|------------|
| Fabricated findings | 54 | 0.27 |
| Missed findings | 546 | 2.73 |
| Severity errors | 50 | — |
| Laterality errors | 60 | — |

Key finding: Zero-shot LLaVA-1.5 tends to **under-report** findings (546 missed) rather than fabricate them (54 fabricated), suggesting the model is conservative but incomplete in its clinical descriptions.

---

## Repository Structure

```
code/
├── README.md                  ← this file
├── requirements.txt           ← pip dependencies
├── configs/
│   └── default.yaml           ← all hyperparameters (model, LoRA, training, eval)
├── src/
│   ├── data.py                ← MIMICCXRDataset, report parsing, manifest I/O
│   ├── model.py               ← LLaVA-1.5 loading, QLoRA wrapping, inference
│   ├── prompts.py             ← zero-shot and training prompt templates
│   ├── metrics.py             ← BLEU, ROUGE-L, BERTScore, CheXBert-F1, t-test
│   ├── chexbert_labeler.py    ← CheXBert wrapper + rule-based fallback labeler
│   └── hallucination.py       ← 4-category hallucination coding utilities
└── scripts/
    ├── prepare_data.py        ← build train/val/test JSONL manifests from CSV
    ├── run_zero_shot.py       ← zero-shot inference (no training)
    ├── train_lora.py          ← QLoRA fine-tuning entry point
    ├── run_inference.py       ← inference with fine-tuned LoRA adapter
    └── evaluate.py            ← compute all metrics + paired t-test

demo/
└── CSE574_Full_Pipeline.ipynb ← end-to-end Colab notebook
```

---

## Setup

### Requirements
- Python 3.11
- CUDA GPU with ≥14 GB VRAM (tested on NVIDIA T4 16GB and P100 16GB)
- MIMIC-CXR-JPG dataset (credentialed PhysioNet access required)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data access
MIMIC-CXR is a credentialed dataset. To access it:
1. Create an account at [physionet.org](https://physionet.org)
2. Complete the CITI "Data or Specimens Only Research" training
3. Request access to [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

Once approved, place the dataset under `data/archive/` or update `configs/default.yaml` with your path.

---

## Running the Pipeline

### Step 1 — Build data manifests
```bash
python scripts/prepare_data.py \
    --csv data/archive/mimic_cxr_aug_train.csv \
    --image-root data/archive/official_data_iccv_final \
    --out-dir data/manifests \
    --train 10000 --val 500 --test 200
```

### Step 2 — Zero-shot inference (baseline)
```bash
python scripts/run_zero_shot.py \
    --config configs/default.yaml \
    --split test \
    --n-samples 200 \
    --out-file outputs/zero_shot_preds.jsonl
```
Runtime: ~25 minutes on T4 GPU.

### Step 3 — LoRA fine-tuning
```bash
python scripts/train_lora.py --config configs/default.yaml
```
Runtime: ~18 hours on T4 GPU (3 epochs, 10k training samples).

### Step 4 — Fine-tuned inference
```bash
python scripts/run_inference.py \
    --config configs/default.yaml \
    --adapter-path checkpoints/lora-rank16 \
    --split test \
    --n-samples 200 \
    --out-file outputs/lora_preds.jsonl
```

### Step 5 — Evaluate
```bash
python scripts/evaluate.py \
    --pred-file outputs/lora_preds.jsonl \
    --compare-file outputs/zero_shot_preds.jsonl \
    --paired \
    --out-file outputs/paired_report.json
```

---

## Running on Google Colab

Open `demo/CSE574_Full_Pipeline.ipynb` in Google Colab with a T4 GPU runtime.

Upload these files to the Colab sidebar:
- `code.zip` — this code package
- `manifests.zip` — pre-built data manifests
- `test_images.zip` — 200 MIMIC-CXR test images

Then run all cells top to bottom. Zero-shot inference completes in ~25 minutes.

---

## Model Architecture

We use **LLaVA-1.5-7B** (`llava-hf/llava-1.5-7b-hf`):

```
Chest X-ray image
    ↓
CLIP-ViT-L/14 (frozen, 307M params)
    ↓ 576 visual tokens
Linear projection (frozen)
    ↓
Vicuna-7B-v1.5 (4-bit NF4 base + LoRA r=16)
    ↓ greedy decode
Findings: ... Impression: ...
```

**LoRA configuration:**
- Rank: 16, Alpha: 32
- Target modules: `q_proj`, `v_proj`
- Trainable parameters: ~4.2M (0.06% of 7B)
- Quantization: 4-bit NF4 (QLoRA)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU-1, BLEU-4 | N-gram overlap with reference reports |
| ROUGE-L | Longest common subsequence F1 |
| BERTScore F1 | Semantic similarity via DistilBERT embeddings |
| CheXBert macro-F1 | Clinical accuracy across 14 chest findings |
| Hallucination coding | Fabricated / Missed / Severity / Laterality errors |

---

## Note on Results

The **zero-shot results in this repository are real experimental results** obtained by running LLaVA-1.5 on 200 actual MIMIC-CXR test images on a Colab T4 GPU. The LoRA fine-tuned results in the report are estimates based on published literature (LLaVA-Med, R2Gen) due to the 18-hour training time constraint.

---

## Citation

If you use this code, please cite the key works this project builds on:

- LLaVA: Liu et al., "Visual Instruction Tuning," NeurIPS 2023
- QLoRA: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023  
- MIMIC-CXR: Johnson et al., Scientific Data, 2019
- CheXBert: Smit et al., EMNLP 2020

---

## License

Code is released under the MIT license. MIMIC-CXR data is governed by the PhysioNet credentialed data use agreement — no patient data or derived reports are included in this repository.