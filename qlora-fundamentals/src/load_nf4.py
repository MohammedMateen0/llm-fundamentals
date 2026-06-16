import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model=AutoModelForCausalLM.from_prerained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
print("NF4 model loaded")