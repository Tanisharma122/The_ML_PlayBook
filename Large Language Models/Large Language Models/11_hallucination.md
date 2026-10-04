# LLM Section · 11. Hallucination

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisites:** 01_tokenization.md, 02_context_window.md (RAG ideas appear again here)

## 1. What is it?
A **hallucination** is when an LLM produces output that is **fluent and confident but false, made up, or not supported by the source it was given.**

Examples:
- Inventing a research paper, a quote or a citation that does not exist.
- Giving a wrong date, number or API function name, with total confidence.
- Summarizing a document and adding details that are not in it.

The danger: hallucinations **sound right**, so people trust them.

## 2. Why it happens (the core idea)
An LLM is trained to **predict the next likely token**, not to check whether a statement is true. It optimizes for *plausible* text. When it lacks the knowledge, it still produces plausible-looking text instead of saying "I don't know."

### Main causes
| Cause | Explanation |
|---|---|
| **Training objective** | Next-token prediction rewards fluency, not truth |
| **Gaps and noise in training data** | Missing, outdated, wrong or biased information |
| **Knowledge cutoff** | The model knows nothing after its training data ends |
| **Overconfidence** | Models are often poorly calibrated and rarely abstain |
| **Decoding randomness** | Higher temperature and sampling can pick less likely, wrong tokens |
| **Weak or missing context** | No grounding material in the prompt, or relevant text is buried in a long context |
| **Ambiguous prompts** | The model guesses the intent and fills in details |
| **Error snowballing** | One wrong token or fact leads the rest of the answer to build on it |

## 3. Types of hallucination
- **Factuality hallucination:** the output contradicts real-world facts (wrong capital, fake law).
- **Faithfulness hallucination:** the output contradicts or goes beyond the **given source** (a summary adds facts not in the document).

Another common split:
- **Intrinsic:** contradicts the provided input.
- **Extrinsic:** adds claims that can't be verified from the input.

## 4. How to reduce hallucinations
No method removes them completely, so combine several layers.

1. **Ground the model with RAG:** retrieve trusted documents and put them in the context (see roadmap: RAG, Vector Database).
2. **Instruct it to stay in the sources:** "Answer only using the context below. If the answer is not there, say you don't know."
3. **Ask for citations or quotes** from the provided text, then check them.
4. **Lower the temperature** for factual tasks.
5. **Use tools:** calculators, search, databases, code execution instead of recalling from memory.
6. **Constrain the output:** structured formats and schema validation catch invalid fields.
7. **Verify with a second pass:** a second model call (or the same one) checks claims against the sources.
8. **Self-consistency:** ask several times; answers that disagree are a warning sign.
9. **Better training:** fine-tuning and preference tuning (RLHF/DPO) that reward admitting uncertainty.
10. **Human review** for high-stakes uses (medical, legal, financial).
11. **Good context hygiene:** put relevant, clean information in the context (see file 12).

## 5. How to measure and detect
- **Faithfulness / groundedness:** is every claim supported by the provided context?
- **Citation accuracy:** do the cited sources really say that?
- **Factual accuracy** on question sets with known answers. TruthfulQA is a well-known benchmark for imitative falsehoods.
- **Human evaluation** on a sample of outputs.
- **Evaluation** is its own roadmap item under Generative AI, so you will go deeper there.

## 6. Hands-on code (grounded prompt + simple consistency check)
```python
# call_llm(prompt, temperature) is a placeholder: use any LLM API you have.

GROUNDED_PROMPT = """You are a careful assistant.
Answer ONLY using the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided context."
Quote the sentence from the context that supports your answer.

Context:
{context}

Question: {question}
"""

def answer(context, question):
    return call_llm(GROUNDED_PROMPT.format(context=context, question=question),
                    temperature=0)

def self_consistency(question, n=5):
    answers = [call_llm(question, temperature=0.8) for _ in range(n)]
    # If answers disagree a lot, treat the result as unreliable.
    return answers
```

**Try this:** ask a model about a made-up library function or a fake paper title. See if it invents details, then add the grounded prompt and compare.

## 7. Common mistakes
- Assuming a confident tone means a correct answer.
- Thinking RAG removes hallucinations. It reduces them, but the model can still ignore or misread the context, and retrieval can fetch the wrong chunks.
- Using high temperature for factual tasks.
- Trusting citations without opening them.
- Believing a bigger model never hallucinates.

## 8. Interview questions
1. What is a hallucination and why do LLMs hallucinate?
2. Factuality vs faithfulness hallucination?
3. Intrinsic vs extrinsic hallucination?
4. How does RAG help, and how can RAG systems still hallucinate?
5. How does temperature affect hallucination?
6. Name five mitigation techniques and the trade-offs of each.
7. How would you measure hallucination in a production system?
8. Why can't we just train hallucinations away?

## 9. Checklist for this topic
- [ ] **Learn**: causes, the two types, mitigation layers, evaluation ideas
- [ ] **Project**: build a small Q&A bot over a PDF with a grounded prompt that says "I don't know" when the answer is missing
- [ ] **Project (extra)**: write 20 test questions (10 answerable, 10 not) and score how often it wrongly answers
- [ ] **Revision**: answer the 8 interview questions without notes

## 10. Resources to explore
- Survey: "A Survey on Hallucination in Large Language Models" (search arXiv)
- Benchmark: TruthfulQA
- Documentation on grounding and citations from your LLM provider
