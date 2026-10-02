# LLM Section · 3. Fine-Tuning

> Roadmap status: Learn ☐ Project ☐ Revision ☐

## 1. What is it?
**Fine-tuning** means continuing to train an already pre-trained LLM on a smaller, task-specific dataset so its **weights are updated** and it learns a new behaviour, style, format or domain.

```
Pre-training (huge raw text) → Base model → Fine-tuning (small curated data) → Specialized model
```
Pre-training teaches the model language and general knowledge. Fine-tuning teaches it **how to behave** for your use case.

## 2. Fine-tuning vs Prompting vs RAG
| | Changes weights? | Best for | Cost |
|---|---|---|---|
| **Prompting** | No | Quick tasks, formatting, few-shot examples | Lowest |
| **RAG** | No | Fresh or private **knowledge/facts**, citations | Low–medium |
| **Fine-tuning** | **Yes** | Consistent **style, format, behaviour**, specialised tasks | Higher |

**Rule of thumb:** use RAG when the model lacks *knowledge*; use fine-tuning when it lacks the right *behaviour*. Always try prompting and RAG first. They are cheaper and easier to update.

## 3. Types of fine-tuning
1. **Full fine-tuning**: update all parameters. Best quality ceiling but needs lots of GPU memory and risks overwriting prior knowledge.
2. **PEFT (Parameter-Efficient Fine-Tuning)**: freeze the base model and train only a small number of extra parameters. Much cheaper.
   - **LoRA (Low-Rank Adaptation)**: instead of updating a big weight matrix W, learn two small low-rank matrices A and B so that the update is ΔW = B·A. Only A and B are trained (often <1% of parameters). Adapters can be merged into the base model or swapped.
   - **QLoRA**: LoRA applied on top of a **4-bit quantized** base model. This lets you fine-tune large models on a single consumer or free-tier GPU.
3. **SFT (Supervised Fine-Tuning)**: train on (prompt → ideal response) pairs. The most common first step.
4. **Preference tuning**: align the model with human preferences.
   - **RLHF**: train a reward model from human rankings, then optimize the LLM with reinforcement learning.
   - **DPO**: skips the reward model and optimizes directly on preferred-vs-rejected pairs. Simpler and more stable.

(RLHF, DPO, LoRA, QLoRA and PEFT are all separate items later in your LLM section, so this file is the overview.)

## 4. The fine-tuning workflow
1. **Define the goal** (e.g. answer in a fixed JSON format, a customer-support tone, domain terminology).
2. **Collect and clean data**: quality beats quantity; hundreds to a few thousand good examples can be enough for SFT.
3. **Format** data in the model's chat template (instruction / input / output).
4. **Split** into train / validation / test sets.
5. **Choose method**: LoRA/QLoRA for most cases.
6. **Train** and watch training vs validation loss.
7. **Evaluate** on held-out examples and compare with the base model.
8. **Deploy**: merge or load the adapter; monitor.

## 5. Data format example (SFT)
```json
{"messages": [
  {"role": "system", "content": "You are a helpful campus assistant."},
  {"role": "user", "content": "What are the library timings?"},
  {"role": "assistant", "content": "The library is open 8 AM to 6 PM, Monday to Saturday."}
]}
```

## 6. Hands-on code (LoRA with Hugging Face PEFT)
```python
# pip install transformers peft trl datasets bitsandbytes accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "your-base-model-name"          # pick a small open model to start
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_cfg = LoraConfig(
    r=8,                  # rank of the low-rank matrices
    lora_alpha=16,        # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # attention layers to adapt (names vary by model)
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()           # shows the tiny % being trained
# Then train with trl's SFTTrainer on your dataset.
```
For QLoRA, load the base model in 4-bit using `BitsAndBytesConfig(load_in_4bit=True, ...)` before applying LoRA.

## 7. Risks and pitfalls
- **Catastrophic forgetting**: the model gets worse at things outside your data.
- **Overfitting**: memorizes the training set. Watch the validation loss.
- **Bad data in, bad model out**: inconsistent or wrong examples get learned faithfully.
- **Fine-tuning doesn't reliably add new facts**: it is better for style/behaviour; use RAG for knowledge that changes.
- **Hallucination can persist**: tuning doesn't remove it.
- Wrong chat template or special tokens can quietly ruin results (links to tokenization, file 01).

## 8. How to evaluate
- Task metrics (accuracy, F1, exact match, format validity).
- Side-by-side comparison with the base model on the same test prompts.
- Human review of a sample of outputs.
- Check for regressions on general abilities.

## 9. Interview questions
1. What is fine-tuning and how does it differ from pre-training?
2. Fine-tuning vs RAG vs prompt engineering: when would you pick each?
3. Explain LoRA. Why does low-rank work and what does `r` control?
4. What does QLoRA add on top of LoRA?
5. What is catastrophic forgetting and how can you reduce it?
6. RLHF vs DPO: what is the difference?
7. How much data do you need and how do you check data quality?

## 10. Checklist for this topic
- [ ] **Learn**: full FT vs PEFT, LoRA idea (ΔW = B·A), SFT vs preference tuning, FT vs RAG
- [ ] **Project**: fine-tune a small open model with LoRA/QLoRA on a tiny custom dataset (e.g. Q&A in a campus-assistant style) using free Colab/Kaggle GPU
- [ ] **Project (extra)**: compare base vs fine-tuned outputs on 20 test prompts
- [ ] **Revision**: answer the 7 interview questions and sketch the workflow from memory

## 11. Resources to explore
- Hugging Face PEFT and TRL documentation
- Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
- Paper: "QLoRA: Efficient Finetuning of Quantized LLMs"
- Hugging Face LLM course (fine-tuning chapters)
