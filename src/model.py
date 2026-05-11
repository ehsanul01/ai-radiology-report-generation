from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")
    bias: str = "none"
    def to_peft(self):
        return LoraConfig(r=self.r, lora_alpha=self.alpha, lora_dropout=self.dropout,
                         bias=self.bias, target_modules=list(self.target_modules), task_type="CAUSAL_LM")

def load_base_model(model_name="llava-hf/llava-1.5-7b-hf", load_in_4bit=True, compute_dtype="bfloat16"):
    dtype = getattr(torch, compute_dtype)
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        model = LlavaForConditionalGeneration.from_pretrained(model_name, quantization_config=bnb_config,
                                                               torch_dtype=dtype, device_map="auto")
    else:
        model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def wrap_with_lora(model, lora_cfg):
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg.to_peft())
    model.print_trainable_parameters()
    return model

def load_for_inference(base_model_name, adapter_path=None, load_in_4bit=True, compute_dtype="bfloat16"):
    model, processor = load_base_model(model_name=base_model_name, load_in_4bit=load_in_4bit, compute_dtype=compute_dtype)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor
