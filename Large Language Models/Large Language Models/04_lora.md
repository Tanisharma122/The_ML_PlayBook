# LLM Section · 4. LoRA (Low-Rank Adaptation)

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisite:** 03_fine_tuning.md (what fine-tuning is, full FT vs PEFT)

## 1. What is it?
**LoRA** is a way to fine-tune a large model by training only a **tiny number of new parameters** while the original weights stay **frozen**.

Instead of updating a huge weight matrix `W` directly, LoRA learns a small *update* to it, built from two thin matrices.

## 2. The core idea (know this for interviews)
Full fine-tuning learns an update `ΔW` with the same shape as `W`.

LoRA's insight: the update needed to adapt a model to a new task has **low "intrinsic rank"**, so it can be approximated by the product of two small matrices.

```
W' = W + ΔW        where   ΔW = (alpha / r) · B · A

W : d × k   (frozen, original)
B : d × r   (trainable)
A : r × k   (trainable)
r : rank, much smaller than d and k  (e.g. 8, 16, 32)
```

Forward pass for an input `x`:
```
h = W·x + (alpha / r) · B·(A·x)
```

### Worked example (parameter count)
Take one attention matrix with d = k = 4096.
- Full update: 4096 × 4096 = **16,777,216** parameters
- LoRA with r = 8: (4096 × 8) + (8 × 4096) = **65,536** parameters
- That is about **0.4%** of the original, for each adapted matrix.

## 3. Initialization (a common interview question)
- `A` is initialized with small random values (Gaussian).
- `B` is initialized to **zero**.
- So `B·A = 0` at the start, which means the model begins **exactly equal to the base model** and training moves it gradually.

## 4. Key hyperparameters
| Parameter | Meaning | Typical values |
|---|---|---|
| `r` (rank) | Size of the low-rank matrices. Higher = more capacity, more parameters | 8, 16, 32, 64 |
| `lora_alpha` | Scaling factor. The update is scaled by `alpha / r` | Often 2×r or equal to r |
| `lora_dropout` | Dropout on the LoRA path to reduce overfitting | 0.0 to 0.1 |
| `target_modules` | Which layers get adapters | `q_proj`, `v_proj`, or all linear layers |

Start small (r = 8 or 16). Raise r only if the model underfits.

## 5. Where is LoRA applied?
Usually on the **attention projection layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Many recent setups also target the MLP layers (`gate_proj`, `up_proj`, `down_proj`) for better quality. Layer names differ by model, so check `model.named_modules()`.

## 6. Advantages
- **Far less memory and compute**: optimizer states exist only for the small adapter weights.
- **Tiny checkpoints**: an adapter is often a few MB instead of many GB.
- **No inference slowdown after merging**: `W + ΔW` can be merged into the base weights.
- **Swappable adapters**: one base model, many task-specific adapters (support bot, code helper, translator).
- **Less catastrophic forgetting**: the base weights are untouched.

## 7. Limitations
- May not match full fine-tuning quality on very hard tasks or large domain shifts.
- Needs tuning of `r`, `alpha` and target modules.
- Still needs the full base model in memory (QLoRA reduces this, see file 05).

## 8. Hands-on code
```python
# pip install transformers peft trl datasets accelerate
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("your-base-model-name")

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# example output style: trainable params: ~0.1-1% of all params

# ... train with Trainer / SFTTrainer ...

model.save_pretrained("my-lora-adapter")        # saves ONLY the adapter

# Optional: merge into the base model for deployment
merged = model.merge_and_unload()
```

**Try this:** train the same task with r = 4, 16 and 64. Compare trainable parameter count, training time and output quality.

## 9. Common mistakes
- Wrong `target_modules` names for the model (nothing gets adapted, or an error).
- Setting `r` very high "to be safe", which loses the efficiency benefit.
- Forgetting that `save_pretrained` on a PEFT model saves only the adapter, so you need the base model to load it again.
- Using the wrong chat template or special tokens (see file 01).

## 10. Interview questions
1. Explain LoRA in simple words. What exactly is trained?
2. Why is `B` initialized to zero?
3. What does the rank `r` control, and how do you choose it?
4. What does `alpha` do?
5. Why can LoRA be merged with no inference cost?
6. Which layers would you apply LoRA to, and why?
7. How many parameters does LoRA add for a d × k matrix with rank r?

## 11. Checklist for this topic
- [ ] **Learn**: ΔW = B·A, rank, alpha scaling, zero init, merging
- [ ] **Project**: fine-tune a small open model with LoRA on a tiny custom dataset using free Colab/Kaggle GPU
- [ ] **Project (extra)**: ablation of r = 4 / 16 / 64 with a results table
- [ ] **Revision**: answer the 7 interview questions without notes; redo the parameter-count calculation

## 12. Resources to explore
- Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.)
- Hugging Face PEFT documentation (LoRA guide)
- Hugging Face LLM course, fine-tuning chapters
