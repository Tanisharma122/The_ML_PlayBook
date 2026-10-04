# LLM Section · 10. Distillation (Knowledge Distillation)

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisites:** 03_fine_tuning.md, 09_quantization.md (so you can compare the two shrinking methods)

## 1. What is it?
**Distillation** trains a **small "student" model** to imitate a **large "teacher" model**, so the student keeps much of the teacher's ability at a fraction of the size, cost and latency.

```
Big teacher model ──(its outputs / probabilities)──► trains ──► Small student model
```
The student learns from the teacher's behaviour, not only from the original labeled data.

## 2. Why it works: "soft labels"
A normal label says: *"the answer is class A"* (a hard label).
A teacher's output says: *"A: 70%, B: 20%, C: 9%, D: 1%"* (soft labels).

Those soft probabilities carry extra information: B is more similar to A than D is. Hinton called this **"dark knowledge"**. The student learns the relationships between answers, not just the winner, so it learns more from each example.

## 3. Temperature (key concept)
Soft labels are produced with a **temperature T** in the softmax:
```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```
- T = 1: normal probabilities (often very peaked).
- T > 1: flatter distribution, which exposes the teacher's "second and third choices".

Both teacher and student use the same T during distillation.

## 4. The classic distillation loss
```
L = α · CE(student, true label)  +  (1 - α) · T² · KL( teacher_soft || student_soft )
```
- **CE term:** learn from the real labels (hard loss).
- **KL term:** match the teacher's softened distribution (soft loss).
- **T²:** keeps gradient sizes comparable when T changes.
- **α:** balance between the two (e.g. 0.5).

## 5. Types of distillation
| Type | What the student learns from |
|---|---|
| **Response-based (logit) distillation** | Teacher's output probabilities |
| **Feature-based** | Teacher's intermediate layer activations (hidden states) |
| **Relation-based** | Relationships between layers or between examples |

### Distillation for LLMs (what you will see in practice)
- **White-box:** you have the teacher's weights and logits, so you can match full probability distributions.
- **Black-box:** you only get the teacher's **text outputs** (for example through an API). You generate answers with the teacher and fine-tune the student on them. This is **sequence-level distillation**, and in practice it is basically SFT on teacher-generated data.
- **Reasoning distillation:** train the student on the teacher's step-by-step reasoning traces so it learns to reason, not just to answer.
- **Synthetic data pipelines:** teacher generates instructions and responses, a filter keeps the good ones, and the student trains on them.

## 6. Known examples
- **DistilBERT:** a smaller BERT trained by distillation. Its paper reports about 40% fewer parameters and about 60% faster inference while keeping most (about 97%) of BERT's language-understanding performance.
- Many small open LLMs are trained or improved using outputs from larger models.

## 7. Distillation vs quantization vs pruning vs LoRA
| Technique | What changes | Needs training? | Result |
|---|---|---|---|
| **Distillation** | A **new, smaller model** learns from a big one | Yes (a full training run) | Fewer parameters |
| **Quantization** | Same model, fewer **bits** per number | Usually no | Smaller memory footprint |
| **Pruning** | Removes weights or structures | Often some | Sparser or smaller model |
| **LoRA / PEFT** | Adds a few trainable parameters | Yes (small) | Adaptation, not shrinking |

These combine well: distill a small model, then quantize it for deployment.

## 8. Hands-on code (PyTorch distillation loss)
```python
import torch
import torch.nn.functional as F

def distill_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # hard loss: student vs true labels
    hard = F.cross_entropy(student_logits, labels)
    # soft loss: student vs teacher's softened distribution
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard + (1 - alpha) * soft

# quick test with fake data
s = torch.randn(4, 5, requires_grad=True)   # student logits (batch 4, 5 classes)
t = torch.randn(4, 5)                       # teacher logits
y = torch.tensor([0, 2, 1, 4])
print(distill_loss(s, t, y))
```
For text generation, the same idea applies per token over the vocabulary.

**Try this:** train a small classifier with and without the teacher's soft labels and compare accuracy.

## 9. Limits and risks
- The student rarely fully matches the teacher, especially on hard reasoning.
- The student **inherits the teacher's mistakes and biases**.
- Distillation needs a good teacher and lots of good teacher outputs (compute cost).
- **Check licenses and terms of service** before using a proprietary model's outputs to train another model. Some providers restrict this.
- Evaluate the student on your own task, not only on general benchmarks.

## 10. Common mistakes
- Confusing distillation with quantization (one makes a new smaller model, the other compresses numbers).
- Forgetting the T² factor or using different temperatures for teacher and student.
- Distilling from a weak teacher or unfiltered synthetic data.
- Calling any fine-tuning on model-generated text "distillation" without noting it is the black-box version.

## 11. Interview questions
1. What is knowledge distillation and why does it work?
2. What are soft labels and "dark knowledge"?
3. What does temperature do in the softmax?
4. Write the distillation loss and explain each term.
5. White-box vs black-box distillation for LLMs?
6. Distillation vs quantization vs pruning: when would you use each?
7. What is sequence-level distillation?
8. What risks come with distilling from another model's outputs?

## 12. Checklist for this topic
- [ ] **Learn**: teacher/student, soft labels, temperature, the combined loss, white-box vs black-box
- [ ] **Project**: distill a small classifier or text model from a larger one and report size, speed and accuracy
- [ ] **Project (extra)**: build a tiny black-box pipeline: teacher writes answers, you fine-tune a small model on them
- [ ] **Revision**: answer the 8 interview questions without notes

## 13. Resources to explore
- Paper: "Distilling the Knowledge in a Neural Network" (Hinton, Vinyals, Dean)
- Paper: "DistilBERT, a distilled version of BERT"
- Hugging Face docs and blog posts on distillation
