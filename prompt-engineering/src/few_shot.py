from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name)

prompt='''Text: Great product
Sentiment: Positive

Text: Bad service
Sentiment: Negative

Text: Amazing quality
Sentiment:'''

inputs=tokenizer(
    prompt,
    return_tensors='pt'
)

output=model.generate(
    **inputs,
    do_sample=True,
    temperature=0.1,

    max_new_tokens=10
)

print(
    tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
)