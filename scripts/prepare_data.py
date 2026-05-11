from __future__ import annotations
import argparse, ast, json, os, random, re
from pathlib import Path

def parse_findings_impression(text):
    fm = re.search(r"[Ff]indings?\s*:\s*(.+?)(?=[Ii]mpression\s*:|$)", text, re.DOTALL)
    im = re.search(r"[Ii]mpression\s*:\s*(.+?)$", text, re.DOTALL)
    f = " ".join(fm.group(1).split()) if fm else ""
    i = " ".join(im.group(1).split()) if im else ""
    return f, i

def safe_parse_list(val):
    if not val or val in ("[]", "nan", ""): return []
    try:
        r = ast.literal_eval(val)
        return r if isinstance(r, list) else []
    except: return []

def main(args):
    import pandas as pd
    print("Reading CSV:", args.csv)
    df = pd.read_csv(args.csv)
    print("  Total rows:", len(df))
    records, skipped = [], 0
    for _, row in df.iterrows():
        pa = safe_parse_list(str(row.get("PA", "[]")))
        ap = safe_parse_list(str(row.get("AP", "[]")))
        frontal = pa if pa else ap
        if not frontal: skipped += 1; continue
        rel = frontal[0]
        image_path = rel if args.no_images else str(Path(args.image_root) / rel)
        if not args.no_images and not os.path.exists(image_path): skipped += 1; continue
        rl = safe_parse_list(str(row.get("text", "[]")))
        if not rl: skipped += 1; continue
        findings = impression = ""
        for rep in rl:
            fi, im = parse_findings_impression(str(rep))
            if len(fi.split()) >= 5 and im: findings, impression = fi, im; break
        if not findings or not impression: skipped += 1; continue
        records.append({"image_path": image_path, "findings": findings, "impression": impression, "subject_id": str(row["subject_id"]), "study_id": rel.split("/")[3] if "/" in rel else "", "split": "train"})
    print("  Valid:", len(records), " Skipped:", skipped)
    random.seed(42); random.shuffle(records)
    nt = min(args.test, len(records))
    nv = min(args.val, len(records)-nt)
    nr = min(args.train, len(records)-nt-nv)
    sp = [("test",records[:nt]),("val",records[nt:nt+nv]),("train",records[nt+nv:nt+nv+nr])]
    for n,d in sp:
        for r in d: r["split"]=n
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for n,d in sp:
        out = Path(args.out_dir)/f"{n}.jsonl"
        open(out,"w").writelines(json.dumps(r)+"" for r in d)
        print(f"  {n}: {len(d)} -> {out}")
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--image-root", default=".")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train", type=int, default=10000)
    p.add_argument("--val", type=int, default=500)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--no-images", action="store_true")
    main(p.parse_args())
