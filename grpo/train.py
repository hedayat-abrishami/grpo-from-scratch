import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.data import load_gsm8k
from grpo.loss import advantages, grpo_loss
from grpo.reward import reward
from grpo.sampling import sample_group, token_logprobs
from grpo.utils import get_device, seed_everything

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
G = 4
STEPS = 50
MAX_NEW_TOKENS = 256
LR = 1e-6
EPS = 0.2
BETA = 0.04


def main():
    seed_everything(0)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    train = load_gsm8k("train")

    for step in range(STEPS):
        ex = random.choice(train)

        # 1. sample a group
        model.eval()
        with torch.no_grad():
            prompt_ids, completions, mask, truncated = sample_group(model, tok, ex["question"], G, MAX_NEW_TOKENS)
            texts = [tok.decode(completions[i][mask[i]], skip_special_tokens=True) for i in range(G)]
            rewards = torch.tensor([reward(t, ex["answer"]) for t in texts], device=device)
            adv = advantages(rewards)
            old_lp = token_logprobs(model, prompt_ids, completions)
            ref_lp = token_logprobs(ref_model, prompt_ids, completions)

        # 2. one gradient step
        new_lp = token_logprobs(model, prompt_ids, completions)
        max_dev = (torch.exp(new_lp - old_lp) - 1)[mask].abs().max().item()
        assert max_dev < 1e-3, f"rho != 1 before update: {max_dev}"

        loss = grpo_loss(new_lp, old_lp, ref_lp, adv, mask, EPS, BETA)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"step {step:3d} | reward {rewards.mean():.3f} | acc {(rewards >= 1).float().mean():.2f} "
              f"| loss {loss.item():+.4f} | grad {grad_norm:.2f} | len {mask.sum(1).float().mean():.0f}")


if __name__ == "__main__":
    main()
