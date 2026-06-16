from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
    )
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name)

prompt='''Let's think step by step.

A store sells 15 books at ₹200 each.
Revenue?'''
inputs=tokenizer(
    prompt,
    return_tensors="pt"
)

outputs=model.generate(
    **inputs,
   
    temperature=0.1,
    top_p=0.9,
    top_k=50,
    max_new_tokens=100,
    do_sample=True
)

print(
    tokenizer.decode(
        outputs[0],
        skiip_special_tokens=True
    )
)
