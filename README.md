# GRPO from scratch

A from-scratch implementation of **Group Relative Policy Optimization** (GRPO,
[Shao et al. 2024](https://arxiv.org/abs/2402.03300)) in plain PyTorch, trained on GSM8K. No RL library
is used: `transformers` loads the models and runs generation, and everything else is written here.

> **Status:** work in progress.

## Setup

```bash
source setup.sh
```

This creates `.venv`, installs `requirements.txt`, and prints the torch version and the detected device
(CUDA, MPS or CPU). It works on macOS (Apple Silicon) and on Linux GPU machines, and needs Python 3.10+.

## Layout

```
grpo/
  utils.py       device selection, seeding
  data.py        GSM8K loading, prompt formatting
setup.sh         environment setup
requirements.txt
```

## References

- Shao et al. 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*:
  [2402.03300](https://arxiv.org/abs/2402.03300)
