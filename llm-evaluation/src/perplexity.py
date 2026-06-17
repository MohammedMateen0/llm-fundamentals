from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch
import math

model_name="gpt2"

tokenizer=AutoTokenizer.from_pretrained(model_name)

model=AutoModelForCausalLM.from_pretrained(model_name)

text='''
Machine learning is a branch of artificial intelligence that enabels system to learn from data'''

inputs=tokenizer(
    text,
    return_tensors='pt'
)

with torch.no_grad():
    outputs=model(
        **inputs,
        labels=inputs["input_ids"]
    )
loss=outputs.loss
perplexity=math.exp(
    loss.item()
)
print(f"Perplexity: {perplexity:.2f}")