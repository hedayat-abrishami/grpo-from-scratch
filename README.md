# The effect of pure clip bound strategies on GRPO. Four clip rules ablation on Qwen2.5-0.5B

## TL;DR

## Motivation and hypothesis

Most of the innovations in this space come from Proximal Policy Optimization (PPO) and its lineage. PPO introduced the clipped surrogate objective, a simplified and cleaner version of Trust Region Policy Optimization (TRPO). Many studies have focused on credit assignment (per token and per sequence) and on the clip bounds. My personal interest in this repo is the clip bounds. Many papers in the PPO lineage introduce multiple innovations at once, such as modifying the clip bound and adding dynamic sampling. As an example, DAPO introduces clip-higher with dynamic sampling. Here, I isolate only the clip bounds and the rest remain GRPO approach: I report the performance of different clipping strategies and show how a small LLM, Qwen2.5-0.5B, responds to each strategy during fine-tuning. 

I conclude that GRPO improves Qwen2.5-0.5B on GSM8K, a math question-answering dataset. I also show that the clipping strategy makes a difference in the model's performance.

## Background

The birth of these reasoning models aren't verifiable rewards it was the original InstructGPT that that used Proximal Policy Optimization (PPO) as its solution to alignment problem where the reward function wasn't verfiable but it was human preferred answers. This is much scalable when we do have a verfiable reward. One of the promiment works in this domain is DeepSeek R1 where it creates a verfiable function for math solution. They use an RL technice called Group Relative Policy Optimization (GRPPO). GRPO saves 

## The four clip rules

## Setup

## Results

### Accuracy

### Entropy

### What each rule clipped

### Stability

### Scorecard against the pre-registered criteria

## Discussion: why entropy didn't collapse

## Systems profile

## Implementation notes

A from-scratch implementation of **Group Relative Policy Optimization** (GRPO,
[Shao et al. 2024](https://arxiv.org/abs/2402.03300)) in plain PyTorch, trained on GSM8K. No RL library
is used: `transformers` loads the models and runs generation, and everything else is written here.

> **Status:** work in progress.

### Layout

```
grpo/
  utils.py       device selection, seeding
  data.py        GSM8K loading, prompt formatting
setup.sh         environment setup
requirements.txt
```

## Limitations and next steps

## Reproduce

```bash
source setup.sh
```

This creates `.venv`, installs `requirements.txt`, and prints the torch version and the detected device
(CUDA, MPS or CPU). It works on macOS (Apple Silicon) and on Linux GPU machines, and needs Python 3.10+.

## Appendix

## References

- Shao et al. 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*:
  [2402.03300](https://arxiv.org/abs/2402.03300)
