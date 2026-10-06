# LLM Section · 13. KV Cache

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisites:** 02_context_window.md (why context is expensive), basic idea of self-attention (Query, Key, Value)
> **Read this before file 14** (speculative decoding builds on it).

## 1. What is it?
The **KV cache** stores the **Key (K) and Value (V)** vectors of all tokens the model has already processed, so that during text generation it does **not recompute them** for every new token.

It is the single most important trick that makes LLM text generation fast enough to be usable.

## 2. Quick refresher: attention
For each token, self-attention builds three vectors: **Query (Q), Key (K), Value (V)**.
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```
- The **Query** of the current token is compared with the **Keys** of all earlier tokens, giving attention weights.
- Those weights mix the **Values** of the earlier tokens.

## 3. The problem: generation is one token at a time
LLMs generate **autoregressively**: output token 1, add it to the input, output token 2, and so on.

Without a cache, for every new token the model would re-run the whole sequence so far:
```
Step 1: process  [The]
Step 2: process  [The, cat]                 ← recomputes "The" again
Step 3: process  [The, cat, sat]            ← recomputes "The", "cat" again
...
```
Most of that work is **repeated for no reason**.

## 4. The key observation
Because attention in a decoder is **causal** (a token only looks at earlier tokens), the K and V of an earlier token **never change** when new tokens are added.

So we compute them **once**, store them, and reuse them.

## 5. How it works with the cache
```
Step t (a new token arrives):
  1. Compute Q, K, V for ONLY the new token
  2. Append the new K and V to the cache
  3. Attend: new Q  ×  ALL cached K  →  weights  →  mix ALL cached V
  4. Output the next token
```
Per step, the model processes **one token** instead of the whole sequence. Overall, this removes a huge amount of repeated computation: the work to generate n tokens drops from roughly cubic to roughly quadratic in n for the attention part.

### Two phases of generation
| Phase | What happens | Bottleneck |
|---|---|---|
| **Prefill** | The whole prompt is processed **in parallel**, and the cache is filled | Compute-bound. Determines **time to first token** |
| **Decode** | One new token per step, reading the cache each time | **Memory-bandwidth-bound**. Determines **tokens per second** |

## 6. The cost: memory
The cache is not free. It grows **with every token** and with every user in a batch.

```
KV cache size = 2 (K and V) × layers × KV heads × head dim × sequence length × batch size × bytes per number
```

### Worked example
A model with 32 layers, 32 KV heads, head dim 128 (a Llama-2-7B-like shape), FP16 (2 bytes):
- Per token: 2 × 32 × 32 × 128 × 2 bytes = **524,288 bytes ≈ 0.5 MB**
- For a 4,096-token context: 0.5 MB × 4,096 ≈ **2 GB for ONE sequence**
- 8 users at once: ≈ **16 GB** of cache, on top of the model weights

This is a main reason why **long contexts and many concurrent users get expensive**, and why long-context providers care so much about cache efficiency.

## 7. How the KV cache problem is reduced
| Technique | Idea |
|---|---|
| **MQA / GQA** (Multi-Query / Grouped-Query Attention) | Share K and V across several query heads, so there are **fewer KV heads** and a much smaller cache. Many modern LLMs use GQA |
| **KV cache quantization** | Store K and V in 8-bit or 4-bit instead of 16-bit (links to quantization, file 09) |
| **PagedAttention** (used in vLLM) | Store the cache in small **blocks (pages)** like OS virtual memory, which cuts memory waste and fragmentation and lets more requests share a GPU |
| **Sliding-window attention** | Keep only the last N tokens in the cache |
| **Cache eviction / compression** | Drop or merge less important tokens' entries |
| **Prefix (prompt) caching** | Reuse the cache for a prompt prefix shared across requests, e.g. a long system prompt, so it isn't recomputed |

## 8. Connection to things you already learned
- **Context window (02):** the cache is a big reason the context window is limited and costly.
- **Context engineering (12):** keeping a **stable prefix** first lets **prompt caching** reuse cached work, cutting cost and latency.
- **Quantization (09):** can be applied to the cache, not only the weights.
- **Speculative decoding (14):** both the draft and target models keep their own caches.

## 9. Hands-on code
### (a) Toy version in NumPy: see the cache idea
```python
import numpy as np
d = 8
rng = np.random.default_rng(0)
Wq, Wk, Wv = [rng.standard_normal((d, d)) for _ in range(3)]

def attend(q, K, V):
    scores = (K @ q) / np.sqrt(d)              # compare new query with all cached keys
    w = np.exp(scores - scores.max()); w /= w.sum()
    return w @ V                               # mix cached values

K_cache, V_cache = [], []

def step(x):                                   # x = embedding of ONLY the new token
    K_cache.append(Wk @ x)                     # compute K, V once ...
    V_cache.append(Wv @ x)                     # ... and keep them
    q = Wq @ x
    return attend(q, np.array(K_cache), np.array(V_cache))

for t in range(5):
    out = step(rng.standard_normal(d))
    print(t, "cache length:", len(K_cache))
```

### (b) See the speed difference in a real model
```python
# pip install transformers torch
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "gpt2"                                   # small model, runs on CPU
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
inputs = tok("The KV cache makes generation", return_tensors="pt")

for use_cache in (False, True):
    t0 = time.time()
    model.generate(**inputs, max_new_tokens=200, use_cache=use_cache, do_sample=False)
    print("use_cache =", use_cache, "→", round(time.time() - t0, 2), "seconds")
```
Expect `use_cache=True` to be clearly faster, and the gap to grow as the output gets longer. (Hugging Face turns the cache on by default.)

**Try this:** compute the cache size for a model you use (look up its layers, KV heads, head dim) at 8k and 32k context.

## 10. Common mistakes
- Confusing the **KV cache** (inside one generation, stores K and V) with **prompt caching** (an API feature that reuses it across requests) or with RAG.
- Forgetting that cache memory grows with **batch size** as well as sequence length.
- Assuming the cache speeds up the **prefill**. It mainly speeds up **decoding**.
- Mixing up the cache with the model weights. The cache is temporary, per request.

## 11. Interview questions
1. What is the KV cache and why does it help?
2. Why can we cache K and V but not Q?
3. Prefill vs decode: what happens in each and what is the bottleneck?
4. Write the KV cache size formula and compute it for a given model.
5. Why does GQA/MQA reduce cache memory?
6. What is PagedAttention and what problem does it solve?
7. How does the KV cache relate to the context window limit?
8. KV cache vs prompt (prefix) caching?

## 12. Checklist for this topic
- [ ] **Learn**: attention recap, causal masking, prefill vs decode, cache size formula, MQA/GQA, PagedAttention
- [ ] **Project**: run the `use_cache` True/False timing test at several output lengths and plot time vs tokens
- [ ] **Project (extra)**: write a small table of KV cache size for 3 models at 4k, 32k and 128k context
- [ ] **Revision**: answer the 8 interview questions without notes

## 13. Resources to explore
- Paper: "Attention Is All You Need" (for the attention formula)
- Paper: "Fast Transformer Decoding: One Write-Head is All You Need" (multi-query attention)
- Paper: "GQA: Training Generalized Multi-Query Transformer Models"
- Paper: "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM)
- Hugging Face docs on caching and generation strategies
