# LLM Section · 1. Tokenization

> Roadmap status: Learn ☐ Project ☐ Revision ☐

## 1. What is it?
An LLM cannot read text. It only works with numbers. **Tokenization** converts raw text into a sequence of integer IDs (tokens) that the model can process, and converts the model's output IDs back into text.

```
"Tokenization is fun"  →  ["Token", "ization", " is", " fun"]  →  [3404, 2860, 318, 1257]
```

A token is **not** always a word. It can be a whole word, part of a word, a single character, or punctuation/space.

## 2. Why not just use words or characters?
| Approach | Problem |
|---|---|
| Word-level | Huge vocabulary; unknown words (OOV) like "ChatGPTify" break it |
| Character-level | Tiny vocabulary but sequences get very long; model must learn spelling to get meaning |
| **Subword-level** | Best balance: common words stay whole, rare words split into reusable pieces |

Modern LLMs all use **subword tokenization**.

## 3. Main subword algorithms
1. **BPE (Byte Pair Encoding)**: used by GPT models. Start with characters (or bytes), repeatedly merge the most frequent adjacent pair until the vocabulary reaches the target size.
2. **WordPiece**: used by BERT. Similar to BPE, but merges are chosen by likelihood of the training data rather than raw frequency. Continuation pieces are marked `##`.
3. **SentencePiece (Unigram / BPE)**: used by T5, LLaMA. Works on raw text (treats space as a symbol `▁`), so it is language-independent.
4. **Byte-level BPE**: operates on bytes, so any character in any language can be encoded and nothing is "unknown".

### BPE in 4 steps (know this for interviews)
1. Start with a vocabulary of single characters.
2. Count all adjacent pairs in the training corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until you reach the vocabulary size (e.g. 32k, 50k, 100k+).

## 4. The full pipeline
```
Raw text → Normalization → Pre-tokenization (split on spaces/punct) → Subword split → Token IDs → Embedding lookup → Model
```
- **Vocabulary**: the fixed list of all tokens (e.g. GPT-2 has 50,257).
- **Special tokens**: `<bos>`, `<eos>`, `<pad>`, `<unk>`, `[CLS]`, `[SEP]`, chat-role markers.
- **Embedding**: each token ID maps to a learned vector, which is the model's real input.

## 5. Hands-on code
```python
# pip install tiktoken transformers
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
text = "Tokenization is fun"
ids = enc.encode(text)
print(ids)                                  # list of ints
print([enc.decode([i]) for i in ids])       # see each token's text
print(len(ids), "tokens")
```

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tok.tokenize("Unbelievably good"))    # WordPiece: ['un', '##bel', ...]
print(tok("Hello world"))                   # input_ids + attention_mask
```

**Try this:** tokenize the same sentence in English, Hindi and Gujarati. Compare token counts.

## 6. Why it matters in practice
- **Cost & limits**: APIs bill per token and context limits are in tokens. Rule of thumb: 1 token ≈ 4 characters ≈ ¾ English word.
- **Non-English text** (Hindi, Gujarati, etc.) often needs *more* tokens per word, so it is costlier and fills the context faster.
- **Odd behaviours** come from tokenization: poor at counting letters in a word, reversing strings, exact arithmetic on long numbers, and spelling tasks, because the model sees chunks, not letters.
- **Whitespace and case matter**: `"hello"`, `" hello"` and `"Hello"` are different tokens.
- A tokenizer must match its model. Never use one model's tokenizer with another.

## 7. Common mistakes
- Counting words instead of tokens when estimating cost or context.
- Forgetting special tokens when fine-tuning (wrong chat template → bad results).
- Assuming a model "sees" letters.

## 8. Interview questions
1. Why do LLMs use subword tokenization instead of word-level?
2. Explain how BPE builds its vocabulary.
3. Difference between BPE, WordPiece and SentencePiece?
4. Why do LLMs struggle with counting letters in "strawberry"?
5. How does vocabulary size affect the model (embedding size, sequence length, speed)?
6. What happens with a word the tokenizer has never seen?

## 9. Checklist for this topic
- [ ] **Learn**: understand BPE, WordPiece, SentencePiece; know what special tokens do
- [ ] **Project**: write a tiny BPE tokenizer from scratch in Python (train on a small text, encode/decode)
- [ ] **Project (extra)**: compare token counts for English vs Hindi vs Gujarati with `tiktoken`
- [ ] **Revision**: answer the 6 interview questions out loud without notes

## 10. Resources to explore
- Andrej Karpathy: "Let's build the GPT Tokenizer" (YouTube)
- Hugging Face course: Tokenizers chapter
- OpenAI `tiktoken` GitHub repo
