"""Gradio demo: paste an abstract + a question, get yes/no/maybe with a reason.

Local:      pip install gradio && python demo/app.py
HF Spaces:  upload demo/app.py, demo/requirements.txt and src/eval/prompts.py (as prompts.py); set the ADAPTER variable.
"""
import os

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.eval.prompts import structured_text
except ImportError:  # on HF Spaces prompts.py sits next to app.py
    from prompts import structured_text

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.environ.get("ADAPTER")  # e.g. you/vaidyalm-qwen1.5b-lora-r16

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
if ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def answer(context, question):
    msgs = [{"role": "user", "content": structured_text(question, context)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


gr.Interface(
    answer,
    [gr.Textbox(lines=10, label="Abstract / context"), gr.Textbox(label="Question")],
    gr.Textbox(label="Answer"),
    title="VaidyaLM",
    description=f"{BASE} + {ADAPTER or 'no adapter'} — research demo, not medical advice.",
    examples=[[
        "Smoking was associated with a marked decline in FEV1 over 10 years in this cohort of 2,000 adults.",
        "Does smoking improve lung function?",
    ]],
).launch()
