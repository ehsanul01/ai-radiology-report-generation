"""Vision-Language Models for Medical Report Generation.

Package modules:
    data              - MIMIC-CXR dataset, report parsing, DataLoader helpers
    model             - LLaVA + LoRA model wrappers
    prompts           - Prompt templates for zero-shot and supervised settings
    metrics           - BLEU / ROUGE / BERTScore / CheXBert label F1
    chexbert_labeler  - Thin wrapper around CheXBert for 14-class labeling
    hallucination     - Structured hallucination coding utilities
"""

__version__ = "1.0.0"
__author__ = "Ehsanul Haque"
