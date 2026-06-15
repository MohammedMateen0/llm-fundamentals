from transformers import AutoTokenizer,AutoModelForCausalLM
model_name="gpt2"

tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(model_name)

def generate(prompt:str,temperature:float=0.5,top_k:int=10,top_p:float=0.5)->str:
    
    
    inputs=tokenizer(prompt,return_tensors='pt')

    outputs=model.generate(
        **inputs,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        max_new_tokens=50
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

if __name__ == "__main__":
    prompt = "Machine Learning is"

    result = generate(
        prompt,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    print(result)