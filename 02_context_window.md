# LLM Section · 2. Context Window

> Roadmap status: Learn ☐ Project ☐ Revision ☐

## 1. What is it?
The **context window** is the maximum number of tokens an LLM can process in a single request. It is the model's "working memory."

Everything counts toward it:
```
System prompt + conversation history + documents/RAG chunks + tool outputs + your question + the model's reply
```
The model has no memory beyond this window. If something is not in the context, the model does not know it (unless it was learned during training).

## 2. Key points
- Measured in **tokens**, not words (see file 01).
- Window = **input + output** together. If the window is 128k and your input is 120k, only about 8k is left for the answer.
- Typical sizes today range from ~8k to 1M+ tokens depending on the model. Always check the current docs for the model you use.
- A bigger window does **not** automatically mean better answers.

## 3. Why is it limited?
- **Self-attention cost**: in a standard transformer every token attends to every other token, so compute and memory grow roughly **O(n²)** with sequence length.
- **KV cache memory**: during generation, the model stores Key/Value vectors for every previous token. Longer context → more GPU memory.
- **Positional encoding**: the model is trained on a maximum length; going beyond it degrades quality unless the position scheme is extended (RoPE scaling, ALiBi, YaRN, etc.).

## 4. Problems with long contexts
| Problem | Meaning |
|---|---|
| **Lost in the middle** | Models recall information at the start and end of a long prompt better than the middle |
| **Context rot / dilution** | More irrelevant text makes it harder to find the relevant parts |
| **Cost & latency** | You pay for every input token; long prompts are slower |
| **Distraction** | Contradictory or noisy content can mislead the model |

## 5. What happens when you exceed it?
- The API returns an error, **or**
- Older messages are truncated/dropped (many chat apps do this silently), so the model "forgets" early details.

## 6. Strategies to work within the limit
1. **Chunking + RAG**: store documents in a vector DB, retrieve only the relevant chunks.
2. **Summarization / compaction**: summarize old conversation turns and keep the summary.
3. **Sliding window**: keep only the last N messages.
4. **Prompt trimming**: remove boilerplate, duplicate text, unneeded fields.
5. **Prompt caching**: reuse a repeated long prefix to cut cost and latency.
6. **Put key instructions and the question at the start or end**, not buried in the middle.
7. **Context engineering**: deliberately deciding *what* goes into the window (instructions, memory, retrieved data, tools). This is its own item on your roadmap, so a good bridge topic.

## 7. Hands-on code
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text): 
    return len(enc.encode(text))

CONTEXT_LIMIT = 8000
RESERVED_FOR_OUTPUT = 1000

def fits(prompt):
    return count_tokens(prompt) <= CONTEXT_LIMIT - RESERVED_FOR_OUTPUT

# Simple sliding-window chat history
def trim_history(messages, max_tokens):
    total, kept = 0, []
    for m in reversed(messages):                 # newest first
        t = count_tokens(m["content"])
        if total + t > max_tokens:
            break
        kept.append(m); total += t
    return list(reversed(kept))
```

**Try this:** paste a long article, count its tokens, and check what fraction of an 8k window it uses.

## 8. Context window vs related ideas
| Concept | Difference |
|---|---|
| Context window | Temporary, per-request, limited in size |
| Model weights (training/fine-tuning) | Permanent knowledge baked into parameters |
| RAG | A method of *filling* the window with relevant retrieved data |
| Memory (agents) | External storage that selectively re-enters the window |

## 9. Common mistakes
- Assuming the model remembers earlier chats automatically.
- Stuffing the whole document in "because the window is big" instead of retrieving what is needed.
- Forgetting to reserve space for the output.
- Burying the key instruction in the middle of a huge prompt.

## 10. Interview questions
1. What is a context window and what counts toward it?
2. Why does attention make long contexts expensive (O(n²))?
3. What is the "lost in the middle" problem?
4. How do you handle a document longer than the context window?
5. Long-context model vs RAG: when to use which?
6. How does the KV cache relate to context length?

## 11. Checklist for this topic
- [ ] **Learn**: tokens → window, O(n²) attention, KV cache, lost in the middle
- [ ] **Project**: build a chatbot that keeps history within a token budget (sliding window + summary of old turns)
- [ ] **Project (extra)**: test "needle in a haystack": hide a fact at different positions in a long prompt and see when the model misses it
- [ ] **Revision**: explain the 6 interview questions in your own words

## 12. Resources to explore
- Paper: "Lost in the Middle: How Language Models Use Long Contexts"
- Paper: "Attention Is All You Need" (for the O(n²) idea)
- Docs on prompt caching and context management from your LLM provider
