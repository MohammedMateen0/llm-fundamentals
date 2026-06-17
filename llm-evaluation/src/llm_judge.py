judge_prompt = """
You are an expert evaluator.

Score the answer from 1 to 10
based on:

1. Correctness
2. Completeness
3. Clarity

Return JSON only.

Question:
What is LoRA?

Answer:
LoRA is a parameter efficient
fine tuning technique.

Output:
"""
print(judge_prompt)