# Week 13 Day 2 - LoRA From Scratch

## Overview

This project implements LoRA (Low-Rank Adaptation) from scratch using PyTorch.

LoRA is a Parameter-Efficient Fine-Tuning (PEFT) technique that enables large language models to be adapted to new tasks without updating all model parameters. Instead of training the entire weight matrix, LoRA learns a low-rank update that significantly reduces memory requirements and trainable parameters.

This project focuses on understanding the mathematics and implementation details behind LoRA rather than using existing libraries.

---

## Why LoRA?

Fine-tuning modern Large Language Models is expensive.

For a 7B parameter model:

* Model Weights ≈ 14 GB (FP16)
* Gradients ≈ 14 GB
* Optimizer States ≈ 28 GB+
* Activations ≈ Several GB

Total memory requirements can exceed 50 GB, making full fine-tuning impractical on consumer GPUs.

LoRA solves this problem by freezing pretrained weights and training only a small number of additional parameters.

---

## Core Idea

Instead of updating the original weight matrix:

```text
W
```

LoRA learns a low-rank update:

```text
ΔW
```

such that:

```text
W' = W + ΔW
```

Rather than learning ΔW directly, LoRA decomposes it into two smaller matrices:

```text
ΔW = BA
```

Where:

```text
B ∈ R(d × r)
A ∈ R(r × k)
```

and:

```text
r << min(d, k)
```

This dramatically reduces the number of trainable parameters.

---

## Example

Original Weight Matrix:

```text
4096 × 4096
```

Total Parameters:

```text
16,777,216
```

Using LoRA with:

```text
r = 16
```

Trainable Parameters:

```text
4096 × 16
+
16 × 4096
=
131,072
```

Trainable Percentage:

```text
131,072 / 16,777,216
≈ 0.78%
```

---

## Project Structure

```text
week13-lora-from-scratch/
│
├── src/
│   ├── lora_linear.py
│   ├── test_zero_init.py
│   └── count_parameters.py
│
├── README.md
└── requirements.txt
```

---

## LoRALinear Implementation

The custom LoRA layer contains:

### Frozen Base Weight

```python
self.weight.requires_grad = False
```

The original pretrained weights remain unchanged during training.

---

### Matrix A

```python
self.A
```

Shape:

```text
(r, in_features)
```

Initialized using small random values.

---

### Matrix B

```python
self.B
```

Shape:

```text
(out_features, r)
```

Initialized to zeros.

---

### Scaling Factor

```python
self.scaling = alpha / r
```

This controls the strength of the LoRA update.

---

## Forward Pass

Standard Linear Layer:

```text
y = Wx
```

LoRA Layer:

```text
y = Wx + (α/r)BAx
```

Implementation:

```python
base = x @ self.weight.T

lora = x @ self.A.T
lora = lora @ self.B.T

output = base + self.scaling * lora
```

---

## Why Initialize B to Zero?

LoRA initializes:

```text
A -> Random
B -> Zero
```

Therefore:

```text
BA = 0
```

Initially:

```text
ΔW = 0
```

Which means:

```text
W' = W
```

The model starts as an exact copy of the pretrained model and gradually learns task-specific updates during training.

---

## Zero Initialization Test

The following verification was performed:

```python
base = x @ layer.weight.T
output = layer(x)

torch.allclose(base, output)
```

Result:

```text
True
```

This confirms that the LoRA layer behaves identically to the frozen base layer before training begins.

---

## Parameter Verification

```text
weight torch.Size([64, 128]) False
A      torch.Size([16, 128]) True
B      torch.Size([64, 16]) True
```

Observations:

* Base weights are frozen
* Matrix A is trainable
* Matrix B is trainable

Only LoRA parameters receive updates during training.

---

## Key Concepts Learned

### Full Fine-Tuning

Updating all model parameters during training.

### Intrinsic Dimensionality Hypothesis

Task-specific updates often lie in a much smaller subspace than the full parameter space.

### Low-Rank Adaptation

Approximate weight updates using low-rank matrices.

### LoRA Scaling

Uses:

```text
α/r
```

to control update magnitude.

### Merging

After training:

```text
W' = W + BA
```

The LoRA update can be merged into the original weights, resulting in no additional inference cost.

### rsLoRA

Uses:

```text
α/√r
```

instead of:

```text
α/r
```

for improved stability at higher ranks.

### DoRA

Extends LoRA by adapting both weight magnitude and direction.

---

## Technologies Used

* Python
* PyTorch

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Tests

### Zero Initialization Test

```bash
python src/test_zero_init.py
```

### Parameter Count

```bash
python src/count_parameters.py
```

---

## Results

Successfully implemented:

* LoRA mathematics from scratch
* Low-rank matrix decomposition
* Frozen base weights
* Alpha scaling
* Zero initialization
* Parameter-efficient adaptation
* Forward pass computation

---

## Next Step

Week 13 Day 3 focuses on QLoRA (Quantized LoRA), which combines LoRA with 4-bit quantization to fine-tune billion-parameter language models on limited GPU memory.
