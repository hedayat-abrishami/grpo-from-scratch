# GRPO from scratch

A from-scratch implementation of **Group Relative Policy Optimization** (GRPO,
[Shao et al. 2024](https://arxiv.org/abs/2402.03300)) in plain PyTorch, trained on GSM8K. It uses no RL
library: `transformers` is used only to load models and to call `generate`.

On top of the baseline, the repo tests one idea: **probability-adaptive clipping**. With a symmetric
clip range, a token the policy currently gives low probability can barely grow in a single update, which
drives entropy collapse. DAPO's Clip-Higher raises the upper bound for every token. This project raises
it only for low-probability tokens:

$$
\epsilon_{\text{high}}(o_{i,t}) = \epsilon_0\big(1 + c\,(1 - \pi_{\text{old}}(o_{i,t}))\big)
$$

Setting $c = 0$ recovers vanilla GRPO.

> **Status:** work in progress. There is no training code or results yet.

## Experiments

| Arm | Clip rule |
|---|---|
| A. GRPO | symmetric, $\epsilon = 0.2$ |
| B. DAPO Clip-Higher | $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$ |
| C. Probability-adaptive | $\epsilon_{\text{low}} = 0.2$, $\epsilon_0 = 0.2$, $c \in \{0.5, 1, 2\}$ |

The clip rule is the only thing that changes between arms. Each arm runs with 3 seeds on
Qwen2.5-0.5B-Instruct. Each run reports:

- GSM8K pass@1 (greedy)
- policy entropy
- clip fraction, bucketed by $\pi_{\text{old}}$
- response length

## Setup

```bash
source setup.sh
```

This creates `.venv`, installs `requirements.txt`, and prints the torch version and the detected device
(CUDA, MPS or CPU). It works on macOS (Apple Silicon) and on Linux GPU machines. It needs Python 3.10 or
newer.

## Hardware

| Stage | Model | Hardware |
|---|---|---|
| Debugging | SmolLM2-135M-Instruct | MacBook M1, 8 GB |
| Experiments | Qwen2.5-0.5B-Instruct | A10 24 GB or A100 |

## Layout

```
grpo/
  data.py        GSM8K loading, prompt formatting
  reward.py      rule-based reward (answer correctness + format)
  sampling.py    group generation, completion masks, old log-probs
  loss.py        group advantages, clip rules, GRPO loss
  train.py       training loop, logging, evaluation
configs/         one config per arm
tests/           reward and loss unit tests
results/         plots and write-up
```

## References

- Shao et al. 2024, *DeepSeekMath*: [2402.03300](https://arxiv.org/abs/2402.03300)
- Yu et al. 2025, *DAPO*: [2503.14476](https://arxiv.org/abs/2503.14476)
- Liu et al. 2025, *Understanding R1-Zero-Like Training (Dr. GRPO)*:
  [2503.20783](https://arxiv.org/abs/2503.20783)
