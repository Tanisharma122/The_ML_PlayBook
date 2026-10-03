# LLM Section · 9. Quantization

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisite:** 05_qlora.md gave you a first taste (NF4). This file covers quantization in general.

## 1. What is it?
**Quantization** reduces the numerical precision of a model's numbers (weights, and sometimes activations) from high-precision formats (like 16-bit floats) to lower-precision ones (like 8-bit or 4-bit integers).

Goal: **smaller model, less memory, often faster inference**, with a small loss in accuracy.

```
FP32 (32 bits) → FP16/BF16 (16 bits) → INT8 (8 bits) → INT4 (4 bits)
```

## 2. Why it matters for LLMs
- LLMs are huge. A 7B model in FP16 needs about 14 GB just for weights.
- Quantizing to 4-bit brings that to roughly 3.5 to 4 GB, so models run on laptops, consumer GPUs and edge devices.
- Lower memory also means less data to move, which often speeds up generation (LLM inference is frequently memory-bandwidth bound).
- Cheaper serving, so more users per GPU.

| Precision | Bytes per parameter | 7B model weights (approx.) |
|---|---|---|
| FP32 | 4 | ~28 GB |
| FP16 / BF16 | 2 | ~14 GB |
| INT8 | 1 | ~7 GB |
| INT4 | 0.5 | ~3.5 GB |

(Weights only. Activations, KV cache and overhead add more.)

## 3. How it works (the math)
**Affine / linear quantization** maps a float `x` to an integer `q`:
```
q = round( x / scale ) + zero_point
x ≈ (q - zero_point) · scale          ← dequantization
scale = (x_max - x_min) / (2^bits - 1)
```
- **Symmetric:** zero_point = 0, range is centered on zero (simple and fast).
- **Asymmetric:** uses a zero_point to cover ranges not centered on zero.

The difference between the original and dequantized value is the **quantization error**.

### Granularity (how many values share one scale)
| Level | Meaning | Trade-off |
|---|---|---|
| Per-tensor | One scale for the whole matrix | Simple, least accurate |
| Per-channel | One scale per row/column | Better accuracy |
| Per-group / block | One scale per small group (e.g. 64 or 128 values) | Common for 4-bit LLMs, accurate with small overhead |

## 4. Types of quantization
- **PTQ (Post-Training Quantization):** quantize an already trained model, with no or little retraining. Fast and cheap. Most LLM quantization is PTQ.
- **QAT (Quantization-Aware Training):** simulate quantization during training so the model learns to handle it. Better accuracy, more costly.
- **Weight-only quantization:** only weights are quantized (activations stay 16-bit). Popular for LLMs (e.g. W4A16).
- **Weight + activation quantization:** both are quantized (e.g. W8A8) for faster integer compute, but activations have outliers that make it harder.
- **KV-cache quantization:** compress the cached keys/values to save memory on long contexts (links to your Context Window file).

## 5. Popular methods and formats (names to recognize)
| Method / format | Idea |
|---|---|
| **LLM.int8()** (bitsandbytes) | 8-bit with special handling of outlier features |
| **NF4 / bitsandbytes 4-bit** | 4-bit NormalFloat used in QLoRA |
| **GPTQ** | PTQ that quantizes layer by layer using second-order information to minimize error |
| **AWQ** | Activation-aware: protects the most important weights based on activation patterns |
| **SmoothQuant** | Shifts activation outliers into weights to enable W8A8 |
| **GGUF** (llama.cpp) | File format and k-quant schemes for running models on CPU/consumer hardware |

## 6. The challenge: outliers
A few weights or activation channels have very large values. A single scale then wastes precision on the normal values. Methods like per-group scales, outlier handling (LLM.int8) and activation-aware scaling (AWQ, SmoothQuant) exist to deal with this.

## 7. Trade-offs
- **Accuracy:** 8-bit is usually near lossless; 4-bit is often fine for many tasks but can drop on hard reasoning, math and code. Below 4-bit loses more.
- **Speed:** depends on hardware and kernel support. Lower bits do not automatically mean faster.
- **Compatibility:** each format needs a runtime that supports it.
- Always **evaluate the quantized model on your own task**.

## 8. Hands-on code
### (a) Understand it by hand (NumPy)
```python
import numpy as np

w = np.random.randn(4, 4).astype(np.float32)        # pretend weights

scale = np.abs(w).max() / 127                        # symmetric INT8
q = np.round(w / scale).astype(np.int8)              # quantize
w_hat = q.astype(np.float32) * scale                 # dequantize

print("max error:", np.abs(w - w_hat).max())
print("memory: float32 =", w.nbytes, "bytes, int8 =", q.nbytes, "bytes")
```
Try it with 4-bit range (-8 to 7) and compare the error.

### (b) Load an LLM in 4-bit (Hugging Face + bitsandbytes)
```python
# pip install transformers accelerate bitsandbytes
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name", quantization_config=bnb, device_map="auto"
)
print(model.get_memory_footprint() / 1e9, "GB")
```
Compare `get_memory_footprint()` for FP16, 8-bit and 4-bit loads of the same model.

### (c) Run a pre-quantized GGUF model locally
Use `llama.cpp` or a tool built on it to run a GGUF model on CPU or a small GPU. Check the current docs for setup.

## 9. Quantization vs related ideas
| Technique | What it shrinks | Needs training? |
|---|---|---|
| Quantization | Bits per number | Usually no (PTQ) |
| Distillation (next roadmap items) | Number of parameters (smaller student model) | Yes |
| Pruning | Removes weights | Usually some |
| LoRA/PEFT | Trainable parameters, not model size | Yes |

These can be combined, e.g. QLoRA = quantization + LoRA.

## 10. Common mistakes
- Assuming 4-bit is always as good as 16-bit. Test on your task.
- Assuming lower bits always mean faster inference.
- Mixing up weight quantization with activation or KV-cache quantization.
- Ignoring that your hardware and runtime must support the chosen format.
- Treating quantization and distillation as the same thing.

## 11. Interview questions
1. What is quantization and why is it useful for LLMs?
2. Write the quantization and dequantization formulas.
3. Symmetric vs asymmetric quantization?
4. PTQ vs QAT: when would you use each?
5. Why does per-group quantization help at 4-bit?
6. What are outliers and why do they make quantization hard?
7. How does GPTQ differ from AWQ, at a high level?
8. Quantization vs distillation vs pruning?
9. How much memory does a 13B model need in FP16 vs 4-bit?

## 12. Checklist for this topic
- [ ] **Learn**: formulas, symmetric/asymmetric, granularity, PTQ vs QAT, weight-only vs W+A, outliers
- [ ] **Project**: load one model in FP16, INT8 and 4-bit; record memory, speed and quality on 20 test prompts in a table
- [ ] **Project (extra)**: implement INT8 quantization of a matrix in NumPy and plot error vs bit-width
- [ ] **Revision**: answer the 9 interview questions; recompute the memory table for a 13B model

## 13. Resources to explore
- Hugging Face docs: quantization overview (bitsandbytes, GPTQ, AWQ)
- Papers: LLM.int8(), GPTQ, AWQ, SmoothQuant, QLoRA
- llama.cpp repository (GGUF and k-quants)
- Maarten Grootendorst's visual guide to quantization (search for it)
