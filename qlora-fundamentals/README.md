# Week 13 Day 3 - QLoRA Fundamentals

## Overview

This project explores the fundamentals of QLoRA (Quantized Low-Rank Adaptation), a technique that combines 4-bit quantization with LoRA to enable efficient fine-tuning of Large Language Models on consumer GPUs.

The objective of this project is to understand how quantization reduces memory usage while preserving model performance and how QLoRA makes billion-parameter model fine-tuning accessible with limited hardware.

---

## Why QLoRA?

LoRA reduces the number of trainable parameters during fine-tuning but still requires the pretrained model to be loaded into memory.

For a 7B parameter model:

| Precision | Memory Required |
| --------- | --------------- |
| FP32      | ~28 GB          |
| FP16      | ~14 GB          |
| 4-bit     | ~3.5 GB         |

Many consumer GPUs cannot fit a 7B model in FP16 memory.

QLoRA solves this problem by storing pretrained model weights in 4-bit format while training only LoRA adapters.

---

## What is Quantization?

Quantization reduces the precision used to store model weights.

Example:

```text
FP32 → 32 bits
FP16 → 16 bits
INT8 → 8 bits
NF4 → 4 bits
```

Lower precision reduces memory usage and allows larger models to fit on limited hardware.

---

## Memory Calculation

For a 7 Billion parameter model:

### FP16

```text
7B × 2 bytes
= 14 GB
```

### 4-bit

```text
7B × 0.5 bytes
= 3.5 GB
```

Memory Reduction:

```text
14 / 3.5
= 4×
```

This reduction is the main reason QLoRA became widely adopted.

---

## NF4 (NormalFloat4)

QLoRA uses NF4, a 4-bit data type specifically designed for neural network weights.

Observation:

Neural network weights approximately follow a normal distribution.

NF4 takes advantage of this property and produces lower quantization error than generic INT4 quantization.

Benefits:

* Better accuracy
* Lower quantization error
* Optimized for LLM weights

---

## Double Quantization

Traditional quantization stores:

* Quantized weights
* Scaling constants

QLoRA introduces double quantization.

Instead of storing scaling constants in higher precision, the scaling constants themselves are quantized.

Benefits:

* Additional memory reduction
* Minimal impact on model quality

---

## Full Precision Computation

QLoRA does not perform computation directly in 4-bit precision.

Workflow:

```text
4-bit Stored Weights
        ↓
Dequantization
        ↓
BF16 Computation
        ↓
Forward Pass
        ↓
Backward Pass
```

Weights are stored in 4-bit format but computations are typically performed using BF16 for numerical stability.

---

## bitsandbytes

The bitsandbytes library provides efficient quantization implementations used by QLoRA.

Configuration used in this project:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Parameters

#### load_in_4bit=True

Loads model weights using 4-bit quantization.

#### bnb_4bit_quant_type="nf4"

Uses the NF4 quantization format.

#### bnb_4bit_compute_dtype=torch.bfloat16

Performs computation using BF16 precision.

---

## Project Structure

```text
week13-qlora-fundamentals/
│
├── notebooks/
│   └── qlora_memory_comparison.ipynb
│
├── src/
│   ├── load_fp16.py
│   ├── load_nf4.py
│   ├── generate.py
│   └── compare_memory.py
│
│
└── README.md
```

---

## Model Used

This project uses:

TinyLlama

for demonstrating quantized model loading and inference.

---

## Experiments

### Experiment 1: FP16 Loading

Loaded the model using standard precision and measured memory consumption.

---

### Experiment 2: NF4 Loading

Loaded the same model using:

```python
load_in_4bit=True
```

and compared memory usage.

---

### Experiment 3: Text Generation

Generated responses using the quantized model to verify that model quality remains high despite reduced storage precision.

---

## Paged Optimizer

Training large models can produce temporary memory spikes.

Paged optimizers solve this problem by moving overflow memory to CPU when GPU memory becomes constrained.

Benefits:

* Prevents Out Of Memory (OOM) errors
* Enables training larger models
* Improves stability during fine-tuning

---

## prepare_model_for_kbit_training()

When fine-tuning quantized models, some layers require higher precision for stable training.

The PEFT utility:

```python
prepare_model_for_kbit_training()
```

applies stability fixes such as:

* LayerNorm casting
* Gradient checkpointing compatibility
* Quantized training preparation

---

## Unsloth

Unsloth is an optimized framework for QLoRA training.

Benefits:

* Faster training
* Lower VRAM usage
* Efficient custom kernels
* Improved Colab performance

Unsloth is commonly used for modern LLM fine-tuning workflows.

---

## Key Concepts Learned

### Quantization

Store model weights using lower precision.

### NF4

4-bit format optimized for normally distributed weights.

### Double Quantization

Quantize quantization constants.

### BF16 Compute

Store in 4-bit, compute in BF16.

### Memory Efficiency

Reduce model memory requirements by approximately 4×.

### QLoRA

Combine quantization and LoRA for efficient LLM fine-tuning.

---

## Technologies Used

* Python
* PyTorch
* Transformers
* bitsandbytes
* Accelerate

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Load Standard Model

```bash
python src/load_fp16.py
```

### Load Quantized Model

```bash
python src/load_nf4.py
```

### Generate Text

```bash
python src/generate.py
```

### Compare Memory

```bash
python src/compare_memory.py
```

---

## Results

Successfully implemented:

* 4-bit model loading
* NF4 quantization
* Quantized inference
* Memory comparison
* BF16 computation workflow

---

## Next Step

Week 13 Day 4 focuses on Prompt Engineering, Instruction Tuning, RLHF, and DPO, which build on the foundation established by LoRA and QLoRA.
