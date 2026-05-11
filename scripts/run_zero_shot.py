# Write run_zero_shot.py
open('scripts/run_zero_shot.py', 'w').write('''
import argparse, json, sys, torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/content/code')
from src.data import MIMICCXRDataset
from src.model import load_for_inference
from src.prompts import build_zero_shot

def run(args):
    import yaml
    cfg = yaml.safe_load(open(args.config))
    manifest = Path(cfg["data"]["manifest_dir"]) / f"{args.split}.jsonl"
    subset = args.n_samples or cfg["data"][f"{args.split}_subset"]
    dataset = MIMICCXRDataset(manifest, subset=subset)
    print(f"Loaded {len(dataset)} examples")
    model, processor = load_for_inference(
        base_model_name=cfg["model"]["base_model"],
        adapter_path=None,
        load_in_4bit=cfg["model"]["load_in_4bit"],
        compute_dtype=cfg["model"]["bnb_4bit_compute_dtype"],
    )
    prompt = build_zero_shot()
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as fh:
        for idx in tqdm(range(len(dataset)), desc="zero-shot"):
            ex = dataset[idx]
            inputs = processor(images=ex["image"], text=prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(**inputs, max_new_tokens=cfg["infer"]["max_new_tokens"], do_sample=False)
            decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            pred = decoded.split("ASSISTANT:")[-1].strip()
            fh.write(json.dumps({
                "subject_id": ex["subject_id"],
                "study_id": ex["study_id"],
                "reference_findings": ex["findings"],
                "reference_impression": ex["impression"],
                "prediction": pred,
                "method": "zero_shot",
            }) + "\\n")
    print(f"Saved to {args.out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--out-file", required=True)
    run(p.parse_args())
''')
print("run_zero_shot.py written")

# Write run_inference.py
open('scripts/run_inference.py', 'w').write('''
import argparse, json, sys, torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/content/code')
from src.data import MIMICCXRDataset
from src.model import load_for_inference
from src.prompts import TRAIN_PROMPT

def run(args):
    import yaml
    cfg = yaml.safe_load(open(args.config))
    manifest = Path(cfg["data"]["manifest_dir"]) / f"{args.split}.jsonl"
    subset = args.n_samples or cfg["data"][f"{args.split}_subset"]
    dataset = MIMICCXRDataset(manifest, subset=subset)
    print(f"Loaded {len(dataset)} examples")
    model, processor = load_for_inference(
        base_model_name=cfg["model"]["base_model"],
        adapter_path=args.adapter_path,
        load_in_4bit=cfg["model"]["load_in_4bit"],
        compute_dtype=cfg["model"]["bnb_4bit_compute_dtype"],
    )
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as fh:
        for idx in tqdm(range(len(dataset)), desc="lora-inference"):
            ex = dataset[idx]
            inputs = processor(images=ex["image"], text=TRAIN_PROMPT, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(**inputs, max_new_tokens=cfg["infer"]["max_new_tokens"], do_sample=False)
            decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            pred = decoded.split("ASSISTANT:")[-1].strip()
            fh.write(json.dumps({
                "subject_id": ex["subject_id"],
                "study_id": ex["study_id"],
                "reference_findings": ex["findings"],
                "reference_impression": ex["impression"],
                "prediction": pred,
                "method": "lora",
            }) + "\\n")
    print(f"Saved to {args.out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--out-file", required=True)
    run(p.parse_args())
''')
print("run_inference.py written")
print("All scripts ready")