from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from scipy import stats

def compute_bleu(references, hypotheses):
    refs_tokens = [[r.split()] for r in references]
    hyps_tokens = [h.split() for h in hypotheses]
    sf = SmoothingFunction().method1
    bleu_1 = corpus_bleu(refs_tokens, hyps_tokens, weights=(1,0,0,0), smoothing_function=sf)
    bleu_4 = corpus_bleu(refs_tokens, hyps_tokens, weights=(.25,.25,.25,.25), smoothing_function=sf)
    return {"bleu_1": bleu_1, "bleu_4": bleu_4}

def compute_rouge_l(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f_scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    return {"rouge_l_f1": float(np.mean(f_scores)), "rouge_l_f1_per_example": f_scores}

def compute_bertscore(references, hypotheses, model_type="distilbert-base-uncased", batch_size=32):
    from bert_score import score as bert_score_fn
    p, r, f = bert_score_fn(cands=list(hypotheses), refs=list(references),
                             model_type=model_type, batch_size=batch_size, lang="en",
                             rescale_with_baseline=False, verbose=False)
    return {"bertscore_f1": float(f.mean().item()), "bertscore_f1_per_example": f.tolist()}

def compute_chexbert_f1(ref_labels, hyp_labels, label_names):
    per_class, f1s = {}, []
    for i, name in enumerate(label_names):
        y_true, y_pred = ref_labels[:, i], hyp_labels[:, i]
        tp = int(((y_true==1)&(y_pred==1)).sum())
        fp = int(((y_true==0)&(y_pred==1)).sum())
        fn = int(((y_true==1)&(y_pred==0)).sum())
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        per_class[name] = f1; f1s.append(f1)
    return {"chexbert_macro_f1": float(np.mean(f1s)), "chexbert_per_class_f1": per_class}

def paired_t_test(a, b):
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    t, p = stats.ttest_rel(b_arr, a_arr)
    return {"t": float(t), "p_value": float(p), "mean_diff": float((b_arr-a_arr).mean()), "n": int(len(a_arr))}
