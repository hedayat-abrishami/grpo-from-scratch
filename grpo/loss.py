import torch


def advantages(rewards):
    """Group-relative advantage: one scalar per completion."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def grpo_loss(new_lp, old_lp, ref_lp, adv, mask, eps, beta, clip="symmetric", eps_high=0.28, c=1.0):
    """Clipped surrogate minus beta * KL to the reference, averaged per completion then over the group."""
    pad = ~mask.bool()
    mask = mask.float()
    n_tokens = mask.sum(dim=1)

    # Zero padded positions *before* exp: masking by multiplication afterwards fails if exp overflows (inf * 0 = nan).
    ratio = torch.exp((new_lp - old_lp).masked_fill(pad, 0.0))
    A = adv.unsqueeze(1)
    lo, hi = clip_bounds(old_lp, clip, eps, eps_high, c)
    lo = torch.as_tensor(lo, dtype=ratio.dtype, device=ratio.device)
    hi = torch.as_tensor(hi, dtype=ratio.dtype, device=ratio.device)
    surr = torch.min(ratio * A, torch.clamp(ratio, lo, hi) * A)
    surr_per_answer = (surr * mask).sum(dim=1) / n_tokens

    d = (ref_lp - new_lp).masked_fill(pad, 0.0)
    kl = torch.exp(d) - d - 1
    kl_per_answer = (kl * mask).sum(dim=1) / n_tokens

    return -(surr_per_answer - beta * kl_per_answer).mean()


def clip_bounds(old_lp, mode, eps, eps_high, c):
    """Lower and upper clip bounds on rho. The only code that differs between the arms."""
    if mode == "symmetric":
        return 1 - eps, 1 + eps
    if mode == "decoupled":
        return 1 - eps, 1 + eps_high
    if mode == "prob_adaptive":
        p_old = old_lp.exp()
        return 1 - eps, 1 + eps * (1 + c * (1 - p_old))
    raise ValueError(f"unknown clip mode {mode}")


BUCKETS = [("lt0.1", 0.0, 0.1), ("0.1-0.5", 0.1, 0.5), ("gt0.5", 0.5, 1.01)]


@torch.no_grad()
def clip_counts(new_lp, old_lp, adv, mask, clip, eps, eps_high, c):
    """Per pi_old bucket: (tokens, upper-clipped, lower-clipped). Clipped = rho beyond the bound the advantage pushes toward."""
    ratio = torch.exp(new_lp - old_lp)
    lo, hi = clip_bounds(old_lp, clip, eps, eps_high, c)
    A = adv.unsqueeze(1)
    upper = (ratio > hi) & (A > 0) & mask
    lower = (ratio < lo) & (A < 0) & mask
    p_old = old_lp.exp()
    counts = {}
    for name, a, b in BUCKETS:
        in_bucket = mask & (p_old >= a) & (p_old < b)
        counts[name] = (in_bucket.sum().item(), (upper & in_bucket).sum().item(), (lower & in_bucket).sum().item())
    return counts