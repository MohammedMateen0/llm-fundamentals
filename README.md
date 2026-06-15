# Week 13 Day 1 - LLM Fundamentals

## Overview

This project explores the fundamental concepts behind Large Language Models (LLMs) using GPT-2 from Hugging Face Transformers.

The goal of this project is to understand how autoregressive language models generate text and how different sampling strategies affect the quality and diversity of generated outputs.

Topics covered include:

* Causal Language Modeling
* Self-Supervised Learning
* Autoregressive Generation
* Context Window
* Tokenization
* Temperature Sampling
* Top-K Sampling
* Top-P Sampling
* KV Cache
* Emergent Capabilities
* Chinchilla Scaling Laws

---

## Project Structure

```text
llm-fundamentals/
│
├── src/
│   ├── generate.py
│   ├── temperature_demo.py
│   ├── topk_demo.py
│   └── topp_demo.py
│
├── outputs/
│   ├── temperature_results.txt
│   ├── topk_results.txt
│   └── topp_results.txt
│
├── requirements.txt
└── README.md
```

---

## What is a Large Language Model?

A Large Language Model (LLM) is trained to predict the next token given all previous tokens.

Mathematically:

P(token_t | token_1, token_2, ..., token_t-1)

Despite this seemingly simple objective, large-scale language models can perform tasks such as:

* Text Generation
* Summarization
* Translation
* Question Answering
* Code Generation
* Reasoning

These capabilities emerge from large-scale pretraining on massive text corpora.

---

## Causal Language Modeling

LLMs are trained using causal language modeling.

The model can only attend to previous tokens and cannot access future tokens during training.

Example:

Input:

```text
Machine learning is
```

Target:

```text
fun
```

The model learns to predict the next token repeatedly across billions of training examples.

---

## Self-Supervised Learning

Unlike traditional supervised learning, LLMs do not require manually labeled datasets.

The training text itself generates labels automatically.

Example:

Input:

```text
Deep learning is
```

Label:

```text
powerful
```

This enables training on trillions of tokens collected from books, websites, code repositories, and research papers.

---

## Sampling Strategies

### Temperature Sampling

Temperature controls randomness in generation.

| Temperature | Behaviour       |
| ----------- | --------------- |
| 0.1         | Deterministic   |
| 0.5         | Conservative    |
| 1.0         | Balanced        |
| 1.5         | Highly Creative |

### Observation

Lower temperatures produce more predictable outputs while higher temperatures increase diversity and randomness.

---

### Top-K Sampling

Top-K limits generation to the K most probable tokens.

Example:

| Token  | Probability |
| ------ | ----------- |
| Paris  | 0.70        |
| London | 0.20        |
| Berlin | 0.05        |
| Tokyo  | 0.03        |

For Top-K = 2:

Only Paris and London remain available for sampling.

---

### Top-P Sampling

Top-P selects the smallest set of tokens whose cumulative probability exceeds a threshold P.

For Top-P = 0.9:

Paris (0.70) + London (0.20) = 0.90

Only these tokens remain available for sampling.

Top-P is adaptive and is commonly used in modern LLM systems.

---

## KV Cache

During autoregressive generation, recomputing attention for previous tokens is computationally expensive.

KV Cache stores previously computed Key and Value tensors so they can be reused during generation.

Benefits:

* Faster inference
* Lower latency
* Reduced computation

---

## Context Window

The context window is the maximum number of tokens a model can attend to when generating the next token.

A larger context window allows the model to process longer documents and maintain more conversational history.

---

## Chinchilla Scaling Law

Chinchilla Scaling Laws demonstrated that model performance depends on both:

* Number of parameters
* Amount of training data

Large models that are undertrained can perform worse than smaller models trained on sufficient data.

A commonly cited compute-optimal ratio is approximately 20 training tokens per parameter.

---

## Emergent Capabilities

Emergent capabilities are abilities that appear as model scale increases.

Examples include:

* Code Generation
* Translation
* Mathematical Reasoning
* Instruction Following

These capabilities were not explicitly programmed but emerged from large-scale next-token prediction training.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* GPT-2

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Temperature Sampling

```bash
python src/temperature_demo.py
```

### Top-K Sampling

```bash
python src/topk_demo.py
```

### Top-P Sampling

```bash
python src/topp_demo.py
```

Generated outputs are saved inside the outputs directory.

---

## Key Learnings

* Understood how LLMs are trained using causal language modeling.
* Explored self-supervised learning and next-token prediction.
* Implemented GPT-2 text generation.
* Compared Temperature, Top-K, and Top-P sampling strategies.
* Learned the role of KV Cache in efficient inference.
* Studied scaling laws and emergent capabilities in modern LLMs.

---

## Next Step

Week 13 Day 2 focuses on Low-Rank Adaptation (LoRA), where parameter-efficient fine-tuning will be implemented from scratch using PyTorch.
