# LLM Section · 7. RLHF (Reinforcement Learning from Human Feedback)

> Roadmap status: Learn ☐ Project ☐ Revision ☐
> **Prerequisite:** 03_fine_tuning.md (SFT). Basic idea of reinforcement learning helps (agent, reward, policy).

## 1. What is it?
**RLHF** is a method to align an LLM with **human preferences** (helpful, honest, harmless) by using human feedback as a reward signal.

Why SFT alone is not enough:
- SFT teaches the model to **imitate** example answers.
- But "good" is hard to write down as a loss function. It is much easier for humans to **compare** two answers and say which one is better.
- RLHF turns those comparisons into a training signal.

## 2. The three stages (know this for interviews)
```
Stage 1: SFT           → teach the model to follow instructions
Stage 2: Reward Model  → learn to score answers the way humans rank them
Stage 3: RL (PPO)      → optimize the LLM to produce high-reward answers
```

### Stage 1: Supervised Fine-Tuning (SFT)
Fine-tune the pre-trained model on high-quality (prompt → response) pairs. This gives a decent starting policy, called the **SFT model** (also the **reference model** later).

### Stage 2: Train a Reward Model (RM)
1. For each prompt, sample several responses from the SFT model.
2. Human labelers **rank** them (or pick the better of two).
3. Train a reward model (usually the LLM with a scalar output head) to give higher scores to preferred responses.

Preference loss (Bradley-Terry style), for a chosen answer `y_w` and rejected answer `y_l`:
```
L = - log σ( r(x, y_w) - r(x, y_l) )
```
It pushes the reward of the chosen answer above the rejected one.

### Stage 3: Optimize the policy with RL (PPO)
The LLM is the **policy**. For a prompt, it generates a response, the reward model scores it, and PPO updates the LLM to increase the expected reward.

A **KL penalty** keeps the model close to the reference (SFT) model:
```
maximize   E[ r(x, y) ]  -  β · KL( π_policy || π_reference )
```
Without the KL term, the model drifts into weird text that fools the reward model.

## 3. Key terms
| Term | Meaning |
|---|---|
| Policy | The LLM being trained |
| Reference model | Frozen copy (usually the SFT model) used for the KL penalty |
| Reward model | Predicts a scalar "human preference" score |
| PPO | Proximal Policy Optimization, the RL algorithm classically used |
| KL penalty | Keeps the policy from straying too far from the reference |
| Value model (critic) | Estimates expected reward, used by PPO to reduce variance |

## 4. Problems with RLHF
- **Complex and unstable:** several models and an RL loop to tune.
- **Memory heavy:** policy, reference, reward and value models may all be needed at once.
- **Reward hacking:** the model finds outputs that score high but are not actually good (e.g. overly long or flattering answers).
- **Expensive and noisy labels:** human preference data is costly and annotators disagree.
- **Bias:** the model learns the labelers' preferences, which may not be universal.

## 5. Related ideas
- **RLAIF / Constitutional AI:** use an AI model (guided by written principles) to produce preference feedback instead of only humans.
- **DPO:** a simpler alternative that skips the reward model and RL loop (next file, 08).
- **GRPO and similar methods:** RL variants that reduce the need for a separate value model.
- **Best-of-N / rejection sampling:** sample many answers, keep the highest-scoring one for further training.

## 6. Hands-on (concept-level code)
Training a full PPO pipeline needs a GPU and careful setup. The Hugging Face **TRL** library provides the building blocks.

```python
# pip install trl transformers datasets
# Stage 2: reward model training (sketch)
from trl import RewardTrainer, RewardConfig
# dataset needs columns like: "chosen" and "rejected" (text for preferred / rejected answers)

# trainer = RewardTrainer(model=reward_model, args=RewardConfig(...),
#                         train_dataset=pref_dataset, processing_class=tokenizer)
# trainer.train()

# Stage 3: PPO / other RL trainers are in TRL as well.
# APIs change between versions, so follow the current TRL docs.
```

A small, doable exercise: write the reward-model loss yourself.
```python
import torch
import torch.nn.functional as F

r_chosen = torch.tensor([1.2, 0.3, 2.0])      # scores for preferred answers
r_rejected = torch.tensor([0.4, 0.5, 1.1])    # scores for rejected answers

loss = -F.logsigmoid(r_chosen - r_rejected).mean()
print(loss)   # smaller when chosen scores exceed rejected scores
```

## 7. Common mistakes
- Thinking RLHF teaches new facts. It mainly shapes **behaviour and style**.
- Forgetting the KL penalty, which leads to reward hacking.
- Confusing the **reward model** (scores answers) with the **policy** (generates answers).
- Skipping SFT and starting RL from a raw base model.

## 8. Interview questions
1. Explain the three stages of RLHF.
2. Why use human *comparisons* instead of human-written scores?
3. Write the reward-model loss and explain it.
4. What does the KL penalty do and why is it needed?
5. What is reward hacking? Give an example.
6. Which models must be in memory during PPO-based RLHF?
7. How does RLAIF differ from RLHF?
8. RLHF vs DPO: pros and cons of each?

## 9. Checklist for this topic
- [ ] **Learn**: SFT → reward model → PPO with KL penalty; reward hacking; RLAIF
- [ ] **Project**: implement the pairwise reward loss and train a tiny reward model on a small preference dataset
- [ ] **Project (extra)**: draw the full RLHF pipeline diagram for your repo README
- [ ] **Revision**: answer the 8 interview questions without notes

## 10. Resources to explore
- Paper: "Training language models to follow instructions with human feedback" (InstructGPT)
- Paper: "Fine-Tuning Language Models from Human Preferences" (Ziegler et al.)
- Paper: "Constitutional AI: Harmlessness from AI Feedback"
- Hugging Face TRL documentation
