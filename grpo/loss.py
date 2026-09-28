import torch


def advantages(rewards):
    """Group-relative advantage: one scalar per completion."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def grpo_loss(new_lp, old_lp, ref_lp, adv, mask, eps, beta):
    """Clipped surrogate minus beta * KL to the reference, averaged per completion then over the group."""
    mask = mask.float()
    n_tokens = mask.sum(dim=1)

    ratio = torch.exp(new_lp - old_lp)
    A = adv.unsqueeze(1)
    surr = torch.min(ratio * A, torch.clamp(ratio, 1 - eps, 1 + eps) * A)
    surr_per_answer = (surr * mask).sum(dim=1) / n_tokens

    d = ref_lp - new_lp
    kl = torch.exp(d) - d - 1
    kl_per_answer = (kl * mask).sum(dim=1) / n_tokens

    return -(surr_per_answer - beta * kl_per_answer).mean()
