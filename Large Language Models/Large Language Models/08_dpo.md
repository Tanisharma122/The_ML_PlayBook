# LLM Section · 8. DPO (Direct Preference Optimization)

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisite:** 07_rlhf.md (preference data, reward model, KL penalty)

## 1. What is it?
**DPO** aligns an LLM with human preferences **directly from preference pairs**, without training a separate reward model and without an RL loop (no PPO).

Core claim of the DPO paper: the RLHF objective can be rewritten so the policy itself acts as an **implicit reward model**, which turns alignment into a simple classification-style loss.

```
RLHF: preference data → reward model → PPO (RL) → aligned model
DPO : preference data ───────────────────────────→ aligned model
```

## 2. The data it needs
Each example has three parts:
```json
{
  "prompt":   "Explain what a context window is.",
  "chosen":   "A context window is the maximum number of tokens ...",
  "rejected": "idk, it's like a window or something."
}
```
`chosen` is the preferred answer, `rejected` is the worse one.

## 3. The DPO loss
For a prompt `x`, chosen `y_w`, rejected `y_l`, policy `π_θ` (being trained) and reference `π_ref` (frozen):

```
L_DPO = - log σ( β · [ (log π_θ(y_w|x) - log π_ref(y_w|x))
                      - (log π_θ(y_l|x) - log π_ref(y_l|x)) ] )
```

In plain words:
- Compare how much the policy has **increased** the probability of the chosen answer (relative to the reference) vs the rejected answer.
- The loss rewards the model for raising chosen answers and lowering rejected ones, **relative to the reference**.
- `β` controls how strongly the model is kept near the reference (similar role to the KL penalty in RLHF). Typical value: around 0.1.

## 4. Typical pipeline
1. Start with a pre-trained base model.
2. Do **SFT** on instruction data → SFT model.
3. Use a copy of the SFT model as the frozen **reference model**.
4. Train with the DPO loss on preference pairs.

## 5. RLHF vs DPO
| | RLHF (PPO) | DPO |
|---|---|---|
| Reward model | Yes, separate | **No** (implicit) |
| RL loop | Yes | **No** |
| Models in memory | Policy, reference, reward, value | Policy + reference |
| Stability / simplicity | Complex, can be unstable | **Simpler, more stable** |
| Data | Preferences (to train RM) + prompts for sampling | Preference pairs only |
| Online sampling from current policy | Yes | No (offline, uses fixed dataset) |
| Cost | Higher | Lower |

## 6. Limitations
- **Offline:** learns from a fixed dataset, not from the model's own fresh outputs, so data quality and coverage matter a lot.
- **Sensitive to data quality:** noisy or inconsistent preferences hurt.
- **Can lower the likelihood of both** chosen and rejected answers in some cases, which may degrade fluency.
- **Length bias:** may drift toward longer or shorter outputs depending on the data.
- **Needs a good SFT starting point.**

## 7. Variants worth knowing (names only for now)
**IPO**, **KTO** (works with simple good/bad labels instead of pairs), **ORPO** (combines SFT and preference optimization), **SimPO** (reference-free). Look these up after you are comfortable with DPO.

## 8. Hands-on code (Hugging Face TRL)
```python
# pip install trl transformers datasets peft
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

model_name = "your-sft-model-name"                 # start from an SFT model
model = AutoModelForCausalLM.from_pretrained(model_name)
tok = AutoTokenizer.from_pretrained(model_name)

# dataset must have: prompt, chosen, rejected
train_ds = load_dataset("json", data_files="preferences.json")["train"]

args = DPOConfig(
    output_dir="dpo-out",
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,        # DPO uses small learning rates
    num_train_epochs=1,
)

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    processing_class=tok,      # older TRL versions call this "tokenizer"
)
trainer.train()
```
Notes:
- TRL's API changes between versions, so confirm argument names in the current docs.
- If no `ref_model` is passed, TRL builds the reference for you. With **LoRA/PEFT** the reference can be the base model with the adapter disabled, which saves memory. This also lets you combine **QLoRA + DPO** on a small GPU.

A pure-PyTorch exercise to understand the loss:
```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # inputs: log-probs of chosen/rejected under policy and reference
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).mean()

print(dpo_loss(torch.tensor([-10.0]), torch.tensor([-12.0]),
               torch.tensor([-11.0]), torch.tensor([-11.0])))
```

## 9. Common mistakes
- Starting DPO from a base model without SFT.
- Using a large learning rate (DPO needs small values).
- Poor-quality or inconsistent preference pairs.
- Confusing `chosen`/`rejected` columns or applying the wrong chat template.
- Not comparing against the SFT model afterwards.

## 10. Interview questions
1. What is DPO and what does it remove compared to RLHF?
2. Write the DPO loss and explain each term.
3. What is the role of `β`?
4. Why is a reference model needed?
5. What does "implicit reward model" mean?
6. Name two weaknesses of DPO.
7. When would you still choose PPO-based RLHF over DPO?
8. How can you run DPO on a small GPU?

## 11. Checklist for this topic
- [ ] **Learn**: preference pairs, DPO loss, β, reference model, RLHF vs DPO
- [ ] **Project**: DPO fine-tune of a small SFT model on a small preference dataset (use LoRA to save memory)
- [ ] **Project (extra)**: build a 100-pair preference dataset yourself and compare before/after outputs
- [ ] **Revision**: answer the 8 interview questions; derive the intuition of the loss in your own words

## 12. Resources to explore
- Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al.)
- Hugging Face TRL: DPOTrainer documentation
- Hugging Face alignment-handbook (recipes for SFT + DPO)
