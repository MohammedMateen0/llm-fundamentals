answers = {
    "Model A":
    "LoRA fine tunes all model parameters.",

    "Model B":
    "LoRA freezes base weights and trains low rank adapters."
}

question = """
What is LoRA?
"""

print(
    "Question:\n",
    question
)

for model, answer in answers.items():

    print(
        f"\n{model}"
    )

    print(answer)