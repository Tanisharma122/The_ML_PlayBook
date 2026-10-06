# LLM Section · 14. Speculative Decoding

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisites:** 13_kv_cache.md (prefill vs decode, memory-bound decoding), 01_tokenization.md (shared vocabulary)

## 1. What is it?
**Speculative decoding** speeds up LLM text generation by letting a **small, fast "draft" model guess several tokens ahead**, and then having the **big "target" model check all of those guesses at once**.

The big win: the final text is **the same as what the big model would have produced on its own** (in the standard method the output distribution is exactly preserved), only faster.

```
Normal decoding:        big model → 1 token → big model → 1 token → ...   (slow, one at a time)
Speculative decoding:   small model guesses 4 tokens  →  big model checks all 4 in ONE pass
```

## 2. The problem it solves
In file 13 you learned that **decoding is memory-bandwidth-bound**: for every single token, the GPU must load all the model's weights, but does very little math with them. The hardware is underused.

Key insight: **checking 4 tokens in parallel costs about the same time as generating 1**, because the weights are loaded once either way. So if we can *propose* several tokens cheaply, verification is almost free.

## 3. The algorithm step by step
1. **Draft:** the small model quickly generates `k` candidate tokens (e.g. k = 4), one by one (it is cheap).
2. **Verify:** the big model processes the prompt plus all `k` draft tokens in **one forward pass**, producing its own probability for each position.
3. **Accept or reject, left to right:**
   - Accept the draft token with probability `min(1, p(x) / q(x))`, where `p` is the target model's probability and `q` is the draft model's.
   - On the **first rejection**, throw away that token and everything after it, and **sample a replacement** from a corrected distribution (the normalized `max(0, p − q)`).
4. **Bonus token:** if all `k` drafts are accepted, the big model's last position gives one **extra** token for free.
5. Repeat from the new position.

So each big-model pass yields **at least 1 token** (never worse than normal decoding in token count) and often several.

### Tiny example
```
Prompt: "The capital of France is"
Draft guesses:  " Paris" "," " a" " city"
Target checks:   ✓ Paris   ✓ ,   ✗ " a" (target prefers " the")
Result this pass: " Paris" "," " the"      ← 3 tokens from ONE big-model pass
```

## 4. How much speedup?
Let **α** = the probability that a draft token is accepted, and **k** = number of draft tokens. The expected number of tokens produced per big-model pass is:

```
E[tokens] = (1 − α^(k+1)) / (1 − α)
```
Example: α = 0.8, k = 4 → (1 − 0.8⁵) / 0.2 ≈ **3.36 tokens per big-model pass**.

Real speedup is lower than that number because the draft model also takes time. Published papers report roughly **2 to 3× faster** generation on suitable tasks, with results depending on the models, hardware and task.

## 5. When it works well, and when it doesn't
**Works well**
- The draft model agrees often with the target (high α): predictable text such as code, boilerplate, structured output, summaries, repeated phrases.
- Low temperature / greedy decoding.
- Memory-bound serving with small batch sizes (the usual single-user case).

**Works less well**
- Creative, high-temperature outputs (the draft guesses wrong more often).
- Very large batch sizes, where the GPU is already busy (compute-bound), so there is less spare capacity.
- A draft model that is too big (slow) or too weak (rarely accepted).

## 6. Requirements and costs
- The draft and target must **share the same tokenizer / vocabulary**, so their token probabilities can be compared.
- **Extra memory:** two models are loaded, each with its own **KV cache** (file 13).
- Needs tuning of `k` and the choice of draft model.

## 7. Variants worth knowing (names for now)
| Variant | Idea |
|---|---|
| **Separate draft model** | The classic setup described above |
| **Self-speculative / layer skipping** | The same model drafts using fewer layers |
| **Medusa** | Extra prediction heads on the target model propose several tokens |
| **EAGLE** | A lightweight head predicts features to draft more accurately |
| **Prompt lookup / n-gram drafting** | Draft by copying matching text from the prompt, with no draft model at all (great for editing, quoting and summarizing) |

## 8. Hands-on code
### (a) Simulate the expected speedup
```python
import random

def tokens_per_pass(alpha, k, trials=100_000):
    total = 0
    for _ in range(trials):
        n = 0
        while n < k and random.random() < alpha:   # draft token accepted
            n += 1
        total += n + 1                              # accepted drafts + 1 token from the target
    return total / trials

for a in (0.5, 0.7, 0.8, 0.9):
    print("alpha =", a, "→", round(tokens_per_pass(a, k=4), 2), "tokens per big-model pass")
```
Compare the results with the formula in section 4.

### (b) Try it in Hugging Face (assisted generation)
```python
# pip install transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2-large")             # target tokenizer
target = AutoModelForCausalLM.from_pretrained("gpt2-large")   # big model
draft  = AutoModelForCausalLM.from_pretrained("gpt2")         # small model, SAME tokenizer

inputs = tok("Speculative decoding speeds up", return_tensors="pt")
out = target.generate(**inputs, assistant_model=draft, max_new_tokens=100)
print(tok.decode(out[0]))
```
Time this against `target.generate(**inputs, max_new_tokens=100)` without `assistant_model`. Feature names and defaults can change between library versions, so check the current Hugging Face docs on assisted generation.

**Try this:** run it on code-like text and on creative writing, and compare the speedup.

## 9. Speculative decoding vs related ideas
| Idea | What it speeds up | Changes output? |
|---|---|---|
| **Speculative decoding** | Generation (decode) via parallel verification | **No** (standard method is lossless) |
| **Quantization (09)** | Smaller weights, less memory traffic | Slightly |
| **Distillation (10)** | A smaller model replaces the big one | Yes (different model) |
| **KV cache (13)** | Avoids recomputing past tokens | No |

The draft model is sometimes made by **distilling** the target, which raises the acceptance rate.

## 10. Common mistakes
- Thinking it makes the big model "less accurate". The standard method keeps the target's output distribution exactly.
- Using a draft model with a different tokenizer.
- Expecting big speedups at high temperature or on very busy servers.
- Forgetting the extra memory for the draft model and its cache.
- Mixing it up with **batching** (many users at once) or **beam search**.

## 11. Interview questions
1. What is speculative decoding and what bottleneck does it exploit?
2. Why is verifying k tokens about as cheap as generating one?
3. Explain the accept/reject rule. What happens on the first rejection?
4. Why is the output distribution unchanged?
5. What does the acceptance rate α control? Use the expected-tokens formula.
6. Name three conditions under which the speedup is small.
7. Why must the draft and target share a tokenizer?
8. Medusa/EAGLE vs a separate draft model: what is the trade-off?

## 12. Checklist for this topic
- [ ] **Learn**: draft/verify loop, accept/reject rule, expected-tokens formula, memory-bound decoding
- [ ] **Project**: run assisted generation with a big and small model pair and report speed for 3 different text types
- [ ] **Project (extra)**: run the α simulation, plot tokens-per-pass vs α for k = 2, 4, 8
- [ ] **Revision**: answer the 8 interview questions without notes

## 13. Resources to explore
- Paper: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.)
- Paper: "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al.)
- Papers: Medusa and EAGLE (search the names on arXiv)
- Hugging Face blog and docs on assisted generation
