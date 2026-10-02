# LLM Section · 6. PEFT (Parameter-Efficient Fine-Tuning)

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisites:** 03_fine_tuning.md, 04_lora.md, 05_qlora.md

## 1. What is it?
**PEFT** is the **family of techniques** for adapting a large pre-trained model by training only a **small fraction of parameters** (or a few new ones), while keeping most of the model frozen.

Think of it this way:
```
PEFT  (the umbrella idea)
 ├── LoRA          (low-rank updates)
 │    └── QLoRA    (LoRA + 4-bit quantized base)
 ├── Adapters
 ├── Prefix tuning / Prompt tuning / P-tuning
 ├── IA³
 └── others (BitFit, DoRA, AdaLoRA, ...)
```
**LoRA and QLoRA are PEFT methods.** Also, "PEFT" is the name of the Hugging Face **library** (`peft`) that implements many of them.

## 2. Why PEFT exists
Full fine-tuning a billion-parameter model means:
- Huge GPU memory (weights + gradients + optimizer states for *every* parameter)
- A full model copy per task (many GB each)
- Higher risk of catastrophic forgetting

PEFT fixes this:
- Train often **well under 1% to a few %** of parameters
- Store only a small **adapter** per task (MBs, not GBs)
- Keep the base model intact and reusable

## 3. Categories of PEFT methods
| Category | Idea | Examples |
|---|---|---|
| **Reparameterization** | Express the weight update in a compact form | LoRA, QLoRA, DoRA, AdaLoRA |
| **Additive: adapters** | Insert small trainable layers inside the model | Bottleneck Adapters |
| **Additive: soft prompts** | Learn extra **virtual token embeddings**, with no weight changes | Prompt tuning, Prefix tuning, P-tuning |
| **Selective** | Unfreeze only a chosen subset of existing parameters | BitFit (train only bias terms) |
| **Scaling-vector methods** | Learn small vectors that rescale activations | IA³ |

## 4. Method snapshots
- **Adapters:** a small bottleneck layer (down-project → nonlinearity → up-project) added inside each transformer block. Only these layers are trained. May add a little inference latency since they are extra layers.
- **Prompt tuning:** prepend a handful of **learnable embeddings** to the input. Only those embeddings are trained. Very few parameters, works better as models get larger.
- **Prefix tuning:** like prompt tuning, but learnable vectors are added at **every layer** (in the attention keys/values).
- **P-tuning:** learns continuous prompt embeddings, often using a small encoder to produce them.
- **BitFit:** train only the **bias** terms. Extremely light.
- **IA³:** learns vectors that scale keys, values and feed-forward activations. Even fewer parameters than LoRA.
- **LoRA:** low-rank update `ΔW = B·A`, mergeable into the base weights (file 04).

## 5. Quick comparison
| Method | Trainable params | Inference latency | Mergeable into base? | Typical use |
|---|---|---|---|---|
| LoRA | Very low | None after merge | Yes | Default choice for LLMs |
| QLoRA | Very low | None after merge (merging needs care) | Yes, with care | Fine-tuning on small GPUs |
| Adapters | Low | Slight increase | No | Multi-task setups |
| Prompt / prefix tuning | Extremely low | Slight (longer input) | No | Very large models, many tasks |
| IA³ | Extremely low | None after merge | Yes | Few-shot style tuning |
| Full fine-tuning | 100% | None | n/a | Maximum quality, high budget |

(Exact trade-offs vary by model and task, so treat this as a guide, not a rule.)

## 6. Hands-on: the Hugging Face PEFT library
```python
# pip install peft transformers
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PromptTuningConfig, get_peft_model, PeftModel

base = AutoModelForCausalLM.from_pretrained("your-base-model-name")

# --- Option A: LoRA ---
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                 task_type="CAUSAL_LM")

# --- Option B: Prompt tuning (swap the config, same workflow) ---
# cfg = PromptTuningConfig(num_virtual_tokens=20, task_type="CAUSAL_LM")

model = get_peft_model(base, cfg)
model.print_trainable_parameters()

# ... train ...

model.save_pretrained("my-adapter")                 # tiny adapter only

# Later: load the adapter on a fresh copy of the base model
base2 = AutoModelForCausalLM.from_pretrained("your-base-model-name")
model2 = PeftModel.from_pretrained(base2, "my-adapter")
```
The same `get_peft_model` workflow works across methods, so you just swap the config.

**Try this:** apply LoRA and prompt tuning to the same model and task. Compare trainable parameters and results.

## 7. When to use what (practical guide)
- **Default:** LoRA.
- **Not enough GPU memory:** QLoRA.
- **Many tasks on one base model:** keep one adapter per task and swap them.
- **Very large model, tiny budget:** try prompt tuning or IA³.
- **Maximum quality and big budget:** consider full fine-tuning.

Reminder from file 03: for **fresh or changing knowledge**, RAG is usually the better tool than any fine-tuning method.

## 8. Common mistakes
- Treating "PEFT" and "LoRA" as the same thing. LoRA is one method inside PEFT.
- Forgetting the adapter needs the **same base model** to be loaded again.
- Picking a method without checking it supports your model architecture in the PEFT library.
- Expecting PEFT to remove hallucinations or add large amounts of new knowledge.

## 9. Interview questions
1. What is PEFT and why is it needed?
2. Name five PEFT methods and group them by category.
3. How is LoRA different from adapters?
4. How is prompt tuning different from prefix tuning?
5. Which PEFT methods add inference latency and which don't?
6. Is QLoRA a separate method from PEFT? Explain the relationship.
7. How can one base model serve many tasks using PEFT?

## 10. Checklist for this topic
- [ ] **Learn**: PEFT categories; LoRA, adapters, prompt/prefix tuning, IA³, BitFit at idea level
- [ ] **Project**: train two adapters (e.g. different tasks) on one base model and swap between them
- [ ] **Project (extra)**: compare LoRA vs prompt tuning on the same task with a small results table
- [ ] **Revision**: answer the 7 interview questions; draw the PEFT family tree from memory

## 11. Resources to explore
- Hugging Face PEFT documentation and conceptual guides
- Survey: "Parameter-Efficient Fine-Tuning for Large Models" (search arXiv)
- Papers: Adapters (Houlsby et al.), Prefix-Tuning (Li & Liang), Prompt Tuning (Lester et al.), IA³ (Liu et al.)
