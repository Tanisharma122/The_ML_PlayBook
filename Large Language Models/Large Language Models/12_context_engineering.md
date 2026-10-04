# LLM Section · 12. Context Engineering

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisites:** 01_tokenization.md, 02_context_window.md, 11_hallucination.md

## 1. What is it?
**Context engineering** is the practice of deciding **what information goes into the model's context window, in what form, and at what moment**, so the model has exactly what it needs to do the task well.

Simple version:
> The model only "knows" what is in its context. Context engineering is designing that context on purpose.

## 2. Prompt engineering vs context engineering
| | Prompt engineering | Context engineering |
|---|---|---|
| Focus | Wording of the instruction | The **whole set of information** the model sees |
| Scope | One prompt | A system that assembles context for every call |
| Includes | Instructions, examples | Instructions + memory + retrieved data + tools + history + format |
| Matters most for | Single-shot tasks | Agents, RAG apps and long-running workflows |

Prompt engineering is **one part** of context engineering.

## 3. What goes into the context (the ingredients)
```
┌──────────────────────────────────────────────┐
│ System instructions (role, rules, tone)      │
│ Tool definitions (what the model can call)   │
│ Few-shot examples                            │
│ Long-term memory (facts saved across chats)  │
│ Retrieved knowledge (RAG chunks)             │
│ Conversation history / summary               │
│ Tool outputs (results of earlier calls)      │
│ Output format / schema                       │
│ The current user request                     │
└──────────────────────────────────────────────┘
          all of it must fit in the context window
```

## 4. Why it matters
- The context window is **limited and costly** (file 02): every token is paid for.
- **More is not better.** Irrelevant text distracts the model and can reduce accuracy (the "lost in the middle" and context dilution problems).
- Good context **reduces hallucinations** (file 11) by giving the model reliable facts to use.
- For agents that run many steps, the context grows quickly, so it must be managed actively.

## 5. Core techniques
A useful way to organize them (a common framing in agent engineering): **write, select, compress, isolate.**

1. **Write (save context outside the window)**
   Store notes, plans and facts in a scratchpad or memory store, so they don't have to stay in the window.
2. **Select (pull in only what's relevant)**
   Use retrieval (RAG), pick only the relevant tools, and load only relevant memories for this request.
3. **Compress (shrink what stays)**
   Summarize old conversation turns, trim long tool outputs, drop duplicates. Keep recent turns in full and compress older ones.
4. **Isolate (split the work)**
   Give sub-tasks to separate agents or calls, each with its own small, focused context, and return only results.

### More practical tips
- **Order matters:** put key instructions at the start and the question near the end, not buried in the middle.
- **Keep a stable prefix:** put unchanging content (system prompt, tool list) first, so **prompt caching** can reuse it and cut cost and latency.
- **Be minimal but sufficient:** the smallest set of high-signal tokens that lets the model succeed.
- **Use clear structure:** labeled sections or XML-style tags make it easier for the model to tell instructions from data.
- **Limit the tool list:** too many similar tools confuses the model.
- **Trim tool results:** return only the fields needed, not whole API responses.

## 6. Common failure modes
| Failure | What happens |
|---|---|
| **Context poisoning** | A wrong fact or hallucination gets into the context and is reused again and again |
| **Context distraction** | A very long context makes the model focus on history instead of the task |
| **Context confusion** | Irrelevant info or too many tools lead to wrong choices |
| **Context clash** | Conflicting pieces of information in the same context |

## 7. Hands-on code (a simple context builder with a token budget)
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
count = lambda s: len(enc.encode(s))

def build_context(system, memory, retrieved_chunks, history, question, budget=6000):
    parts = [("system", system), ("memory", memory)]
    used = sum(count(t) for _, t in parts) + count(question)

    # 1) add retrieved chunks (most relevant first) while budget allows
    for chunk in retrieved_chunks:
        if used + count(chunk) > budget * 0.5:
            break
        parts.append(("context", chunk)); used += count(chunk)

    # 2) add recent history (newest first), then restore the order
    kept = []
    for msg in reversed(history):
        if used + count(msg) > budget:
            break
        kept.append(("history", msg)); used += count(msg)
    parts += list(reversed(kept))

    parts.append(("question", question))
    return "\n\n".join(f"[{name}]\n{text}" for name, text in parts)
```
Upgrade idea: instead of dropping old history, summarize it and add the summary as one short block.

## 8. Where you will use it
- **RAG apps:** deciding which chunks to retrieve and how many.
- **Chatbots:** summarizing old turns to control cost.
- **Agents:** managing tool outputs and long task histories.
- **Automations and APIs:** every call is billed by tokens, so lean context saves money.

## 9. Common mistakes
- Stuffing everything into the prompt "just in case".
- Resending the full chat history every call.
- Returning huge raw tool outputs into the context.
- Treating context engineering as only "better prompt wording".
- Never testing what actually reaches the model (always log the final assembled context).

## 10. Interview questions
1. What is context engineering and how does it differ from prompt engineering?
2. List the components that can appear in an LLM's context.
3. Why is a longer context not always better?
4. Explain write, select, compress and isolate with an example each.
5. What are context poisoning and context distraction?
6. How does prompt caching influence how you order the context?
7. How would you manage context for an agent running 50 tool calls?
8. How does context engineering reduce hallucinations?

## 11. Checklist for this topic
- [ ] **Learn**: ingredients of context, the four techniques, failure modes, link to cost and hallucination
- [ ] **Project**: build a chatbot that keeps recent turns in full, summarizes older turns, and stays under a token budget
- [ ] **Project (extra)**: log the final assembled context for 10 requests and remove everything that wasn't needed
- [ ] **Revision**: answer the 8 interview questions; sketch the context diagram from memory

## 12. Resources to explore
- Your LLM provider's docs on prompt caching, long-context tips and agent design
- Engineering blog posts on context engineering for AI agents (search the term, and favor posts from AI labs and framework authors)
- Frameworks to explore later on your roadmap: LangChain, LangGraph, LlamaIndex, DSPy
