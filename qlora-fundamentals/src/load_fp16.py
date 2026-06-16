from transformers import AutoModelForCausalLM
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)
print("FP!^ model loaded")