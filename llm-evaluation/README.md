# Week 13 Day 5 - LLM Evaluation and Alignment

## Overview

This project explores evaluation techniques used to measure the quality of Large Language Models (LLMs).

Training and fine-tuning a model is only part of the process. An equally important challenge is determining whether the model has actually improved. Traditional metrics such as Perplexity and BLEU provide quantitative evaluation, while modern approaches such as LLM-as-a-Judge attempt to measure quality in ways that better align with human preferences.

This project implements:

* Perplexity Calculation
* BLEU Score Calculation
* LLM-as-a-Judge Prompting
* Output Comparison Framework

It also explores alignment concepts such as RLHF, DPO, Constitutional AI, and Catastrophic Forgetting.

---

## Why LLM Evaluation Matters

A model may:

* Have lower loss
* Have lower perplexity
* Achieve higher benchmark scores

but still produce poor responses for real users.

Evaluation helps answer questions such as:

* Is the model more accurate?
* Is the model more helpful?
* Is the model safer?
* Is the model aligned with human preferences?

---

## Perplexity

Perplexity measures how surprised a language model is by the actual next token.

Lower perplexity indicates that the model predicts text more effectively.

Mathematically:

```text
Perplexity = e^(Average Negative Log Likelihood)
```

### Interpretation

| Perplexity | Interpretation        |
| ---------- | --------------------- |
| Lower      | Better language model |
| Higher     | Worse language model  |

Example:

```text
Model A: PPL = 8
Model B: PPL = 15
```

Model A predicts text more accurately.

### Limitation

Lower perplexity does not always produce better conversations or more useful responses.

---

## BLEU Score

BLEU (Bilingual Evaluation Understudy) is a traditional NLP evaluation metric.

It measures n-gram overlap between:

```text
Reference Text
       ↓
Generated Text
```

Higher overlap results in a higher BLEU score.

### Example

Reference:

```text
The cat sits on the mat
```

Prediction:

```text
The cat sits on mat
```

BLEU Score:

```text
High
```

because many words overlap.

### Limitation

BLEU focuses on word overlap rather than meaning.

Example:

```text
The movie was excellent.
```

and

```text
The film was fantastic.
```

have nearly identical meaning but may receive different BLEU scores.

---

## ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is commonly used for summarization tasks.

Unlike BLEU, ROUGE emphasizes recall and measures how much of the reference text is captured by the generated output.

### Common Variants

* ROUGE-1
* ROUGE-2
* ROUGE-L

### Limitation

ROUGE still relies on word overlap and may not accurately reflect semantic similarity.

---

## LLM-as-a-Judge

Modern evaluation often uses a stronger language model to evaluate another model's outputs.

Example workflow:

```text
Question
    ↓
Candidate Answer
    ↓
Judge Model
    ↓
Score
```

The judge evaluates:

* Correctness
* Completeness
* Clarity
* Helpfulness

### Example Prompt

```text
You are an expert evaluator.

Score the answer from 1 to 10 based on:

1. Correctness
2. Completeness
3. Clarity

Return valid JSON.
```

### Advantages

* Scalable
* Faster than human review
* Better correlation with human judgment than BLEU or ROUGE

---

## Standard LLM Benchmarks

### MMLU

Massive Multitask Language Understanding.

Evaluates knowledge across:

* Mathematics
* Science
* History
* Medicine
* Law
* Humanities

A high MMLU score indicates strong general knowledge.

---

### HumanEval

Used to evaluate code generation models.

Measures whether generated code passes hidden unit tests.

Common metric:

```text
Pass@1
```

Meaning:

The first generated solution passes all tests.

---

### MT-Bench

A benchmark designed for evaluating conversational AI systems.

Measures:

* Reasoning
* Instruction following
* Helpfulness
* Dialogue quality

Often evaluated using LLM judges.

---

## RLHF

RLHF stands for:

```text
Reinforcement Learning from Human Feedback
```

It is one of the original techniques used to align language models with human preferences.

### RLHF Pipeline

```text
Pretrained Model
        ↓
Human Preference Data
        ↓
Reward Model
        ↓
PPO Training
        ↓
Aligned Model
```

---

## Reward Model

Humans compare multiple responses.

Example:

Prompt:

```text
Explain LoRA.
```

Response A

Response B

Human chooses the preferred answer.

The reward model learns these preferences and assigns scores to future outputs.

Purpose:

```text
Learn what humans prefer
```

---

## PPO

PPO stands for:

```text
Proximal Policy Optimization
```

PPO optimizes the language model using scores produced by the reward model.

Goal:

* Maximize reward
* Prevent unstable updates
* Preserve model quality

---

## DPO

DPO stands for:

```text
Direct Preference Optimization
```

DPO removes:

* Reward Model
* PPO Training

Instead, it directly learns from:

```text
Prompt
Chosen Response
Rejected Response
```

### Advantages

* Simpler training
* More stable optimization
* Lower computational cost

---

## RLHF vs DPO

| RLHF                   | DPO                     |
| ---------------------- | ----------------------- |
| Reward Model Required  | No Reward Model         |
| PPO Required           | No PPO                  |
| More Complex           | Simpler                 |
| Higher Cost            | Lower Cost              |
| Older Alignment Method | Modern Alignment Method |

---

## Constitutional AI

Constitutional AI is an alignment technique that reduces dependence on human labeling.

Process:

```text
Generate Response
       ↓
Self Critique
       ↓
Revision
```

The model critiques and improves its own outputs using predefined principles.

Benefits:

* Reduced human annotation
* Better scalability
* Improved safety

---

## Catastrophic Forgetting

A common fine-tuning problem.

Example:

```text
General LLM
      ↓
Fine-Tuned on Legal Data
```

Result:

```text
Better Legal Knowledge
Worse General Knowledge
```

The model forgets previously learned capabilities.

This phenomenon is known as catastrophic forgetting.

---

## Project Structure

```text
week13-llm-evaluation/
│
├── src/
│   ├── perplexity.py
│   ├── bleu_score.py
│   ├── llm_judge.py
│   └── compare_outputs.py
│
└── README.md
```

---

## Experiments

### Experiment 1 - Perplexity

Evaluate how well a language model predicts text.

---

### Experiment 2 - BLEU Score

Measure n-gram overlap between generated and reference text.

---

### Experiment 3 - LLM Judge Template

Create an evaluation prompt capable of scoring model outputs.

---

### Experiment 4 - Output Comparison

Compare responses generated by multiple models.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

### Perplexity

```bash
python src/perplexity.py
```

### BLEU Score

```bash
python src/bleu_score.py
```

### LLM Judge

```bash
python src/llm_judge.py
```

### Compare Outputs

```bash
python src/compare_outputs.py
```

---

## Key Learnings

* Perplexity measures language modeling quality.
* BLEU and ROUGE measure text similarity through n-gram overlap.
* LLM-as-a-Judge provides scalable evaluation aligned with human preferences.
* MMLU and HumanEval are standard LLM benchmarks.
* RLHF aligns models through reward modeling and PPO.
* DPO simplifies alignment by directly learning from preference pairs.
* Constitutional AI enables self-critique and revision.
* Catastrophic forgetting is a major challenge during fine-tuning.

---

## Next Step

The next stage is the Week 13 Project:

**Fine-Tune an LLM using LoRA/QLoRA and publish it on Hugging Face.**

This project combines:

* LLM Fundamentals
* LoRA
* QLoRA
* Prompt Engineering
* Evaluation

into a complete end-to-end LLM engineering workflow.
