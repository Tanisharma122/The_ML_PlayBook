# LLM Section · 5. QLoRA (Quantized LoRA)

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisite:** 04_lora.md (LoRA basics). Quantization itself has its own roadmap item later.

## 1. What is it?
**QLoRA** = **LoRA on top of a 4-bit quantized base model.**

- The big base model is **frozen and stored in 4-bit** precision, which saves a lot of GPU memory.
- Small **LoRA adapters** are trained in higher precision (16-bit).
- Result: you can fine-tune models that would normally not fit on your GPU, with quality close to 16-bit LoRA.

## 2. The problem it solves
Memory needed during fine-tuning:
```
model weights + gradients + optimizer states + activations
```
Even with LoRA (small gradients and optimizer states), the **base model weights** still take most of the memory.

Rough weight memory for a 7B-parameter model:
| Precision | Bytes per parameter | Approx. weight memory |
|---|---|---|
| FP32 | 4 | ~28 GB |
| FP16 / BF16 | 2 | ~14 GB |
| 4-bit | 0.5 | ~3.5 GB |

(Plus overhead for activations, adapters and optimizer, so real usage is higher.)

## 3. Quick intro to quantization
**Quantization** stores numbers with fewer bits (for example 4-bit instead of 16-bit) to save memory, at the cost of some precision. The weights are grouped into small blocks and each block gets a scaling constant so values can be mapped back approximately.

## 4. The three key ingredients of QLoRA
1. **4-bit NormalFloat (NF4)**
   A 4-bit data type designed for weights that follow a roughly **normal distribution**, which pretrained model weights usually do. It represents them more accurately than plain 4-bit integers.
2. **Double quantization**
   The quantization constants (one per block) are themselves quantized, which saves additional memory.
3. **Paged optimizers**
   Optimizer states can spill to CPU RAM during memory spikes, avoiding out-of-memory crashes.

## 5. How training actually works
```
Frozen base weights (4-bit NF4)
        │  dequantize on the fly to BF16 for compute
        ▼
Forward/backward pass  ──►  gradients flow only into LoRA adapters (BF16/FP16)
        ▼
Only the adapter weights are updated and saved
```
- The 4-bit weights are **never updated**. They are dequantized temporarily for each computation.
- The saved result is just the **LoRA adapter**, the same as in plain LoRA.

## 6. LoRA vs QLoRA
| | LoRA | QLoRA |
|---|---|---|
| Base model precision | 16-bit | **4-bit (NF4)** |
| GPU memory | Medium | **Much lower** |
| Training speed | Faster | Somewhat slower (dequantization overhead) |
| Quality | Strong | Very close to LoRA in the paper's results |
| Best for | Enough GPU memory | Limited GPU (free Colab/Kaggle, consumer GPUs) |

## 7. Hands-on code
```python
# pip install transformers peft trl datasets accelerate bitsandbytes
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NormalFloat4
    bnb_4bit_use_double_quant=True,        # double quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # compute in bf16 (use float16 if bf16 unsupported)
)

model = AutoModelForCausalLM.from_pretrained(
    "your-base-model-name",
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Train with SFTTrainer; use a paged optimizer such as "paged_adamw_8bit"
```

Note: `bitsandbytes` needs a compatible GPU setup (commonly an NVIDIA GPU). Check its current docs for supported hardware before you start.

## 8. Practical tips
- Start with a **small model** (1B to 3B) to learn the workflow, then scale up.
- Use **gradient checkpointing** and small batch sizes with gradient accumulation to save memory.
- Always compare against the base model to confirm the fine-tune actually helps.
- Evaluate with the quantized model the way you will deploy it, or merge carefully (merging adapters into a quantized base needs extra care: check the PEFT docs).

## 9. Common mistakes
- Using the wrong compute dtype for your GPU (bf16 vs fp16).
- Skipping `prepare_model_for_kbit_training`.
- Expecting QLoRA to be faster than LoRA. It saves **memory**, not time.
- Assuming 4-bit means the adapters are 4-bit. The adapters are in higher precision.

## 10. Interview questions
1. What is QLoRA and how is it different from LoRA?
2. What is NF4 and why is it better than plain 4-bit integers for LLM weights?
3. What is double quantization?
4. Are the 4-bit weights updated during training? What gets updated?
5. What are paged optimizers for?
6. Why is QLoRA slower per step than LoRA but still useful?
7. Roughly how much memory does a 7B model need in FP16 vs 4-bit?

## 11. Checklist for this topic
- [ ] **Learn**: quantization basics, NF4, double quantization, paged optimizers, frozen 4-bit base + 16-bit adapters
- [ ] **Project**: QLoRA fine-tune of a small model on free Colab/Kaggle GPU; record peak GPU memory
- [ ] **Project (extra)**: compare LoRA vs QLoRA: memory used, training time, output quality
- [ ] **Revision**: answer the 7 interview questions; redraw the training flow from memory

## 12. Resources to explore
- Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al.)
- Hugging Face blog and docs on 4-bit quantization with bitsandbytes
- Hugging Face PEFT + TRL documentation
