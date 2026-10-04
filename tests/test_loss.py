"""The loss and clip rules: the code the experiment's conclusions depend on."""

import math

import pytest
import torch

from grpo.loss import advantages, clip_bounds, clip_counts, grpo_loss

EPS, BETA = 0.2, 0.04


@pytest.fixture
def batch():
    """G=4 answers x L=6 tokens, with rho != 1 so clipping matters, and padding on two rows."""
    torch.manual_seed(0)
    old_lp = -torch.rand(4, 6) * 4
    new_lp = old_lp + 0.4 * torch.randn(4, 6)
    ref_lp = old_lp + 0.1 * torch.randn(4, 6)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[1, 4:] = False
    mask[3, 2:] = False
    return new_lp, old_lp, ref_lp, adv, mask


def loss(b, **kw):
    new_lp, old_lp, ref_lp, adv, mask = b
    return grpo_loss(new_lp, old_lp, ref_lp, adv, mask, EPS, BETA, **kw)


# --- advantages ---

def test_advantages_are_zero_mean():
    a = advantages(torch.tensor([0.0, 0.1, 1.1, 0.1]))
    assert abs(a.mean().item()) < 1e-6


def test_all_equal_rewards_give_zero_advantage():
    assert torch.all(advantages(torch.tensor([0.1, 0.1, 0.1, 0.1])) == 0)


# --- each arm reduces exactly to plain GRPO when its extra knob is off ---

def test_default_is_symmetric(batch):
    assert torch.equal(loss(batch), loss(batch, clip="symmetric"))


def test_prob_adaptive_with_c0_equals_symmetric(batch):
    assert torch.allclose(loss(batch, clip="prob_adaptive", c=0.0), loss(batch, clip="symmetric"))


def test_decoupled_with_eps_high_eq_eps_equals_symmetric(batch):
    assert torch.allclose(loss(batch, clip="decoupled", eps_high=EPS), loss(batch, clip="symmetric"))


def test_arms_differ_when_rho_moves(batch):
    sym = loss(batch, clip="symmetric")
    assert not torch.allclose(loss(batch, clip="decoupled"), sym)
    assert not torch.allclose(loss(batch, clip="prob_adaptive", c=1.0), sym)


def test_clip_rule_is_irrelevant_when_rho_is_1(batch):
    """On the first pass (K=1) pi_theta == pi_old, so every arm must give the same loss."""
    _, old_lp, ref_lp, adv, mask = batch
    b = (old_lp.clone(), old_lp, ref_lp, adv, mask)
    sym = loss(b, clip="symmetric")
    assert torch.allclose(loss(b, clip="decoupled"), sym)
    assert torch.allclose(loss(b, clip="prob_adaptive", c=2.0), sym)


def test_upper_bound_is_irrelevant_when_every_advantage_is_negative(batch):
    """The upper bound only binds for A > 0, so eps_high changes nothing for bad answers."""
    new_lp, old_lp, ref_lp, _, mask = batch
    b = (new_lp, old_lp, ref_lp, torch.tensor([-1.0, -0.5, -0.2, -1.5]), mask)
    assert torch.allclose(loss(b, clip="symmetric"), loss(b, clip="decoupled", eps_high=10.0))


# --- the bounds themselves ---

def test_prob_adaptive_upper_bound_per_token():
    p_old = torch.tensor([0.05, 0.3, 0.9])
    lo, hi = clip_bounds(p_old.log(), "prob_adaptive", EPS, 0.28, 1.0)
    assert lo == pytest.approx(1 - EPS)
    expected = 1 + EPS * (1 + 1.0 * (1 - p_old))           # rare tokens get more room
    assert torch.allclose(hi, expected)
    assert hi[0] > hi[1] > hi[2]


def test_dcpo_bounds_match_the_paper():
    """DCPO eq. 4 with eps_low = 0.16, eps_high = 0.2: both bounds widen as pi_old falls; ratio capped at 10."""
    q = torch.tensor([1.0, 0.5, 0.1, 1e-6])
    lo, hi = clip_bounds(q.log(), "dcpo", 0.16, 0.2, 1.0)
    assert lo[0].item() == pytest.approx(0.8, abs=1e-6)                          # 0.5 + 0.5*sqrt(1 - 0.64)
    assert hi[0].item() == pytest.approx(0.5 + 0.5 * math.sqrt(1.8), abs=1e-6)   # 1.17: tighter than 1.2 when confident
    assert torch.all(lo[1:] <= lo[:-1]) and torch.all(hi[1:] >= hi[:-1])         # rarer token -> wider bounds
    assert lo[-1].item() == pytest.approx(0.5) and hi[-1].item() == pytest.approx(10.0)


def test_dcpo_changes_the_loss(batch):
    new_lp, old_lp, ref_lp, adv, mask = batch
    dcpo = grpo_loss(new_lp, old_lp, ref_lp, adv, mask, 0.16, BETA, clip="dcpo", eps_high=0.2)
    assert torch.isfinite(dcpo)
    assert not torch.allclose(dcpo, loss(batch, clip="symmetric"))


def test_unknown_clip_mode_raises():
    with pytest.raises(ValueError):
        clip_bounds(torch.zeros(1), "assymetric", EPS, 0.28, 1.0)


# --- masking and KL ---

def test_padding_does_not_affect_loss_or_gradient(batch):
    new_lp, old_lp, ref_lp, adv, mask = batch
    garbage = new_lp.clone()
    garbage[~mask] = 123.0                                    # junk at padded positions
    assert torch.allclose(loss((garbage, old_lp, ref_lp, adv, mask)), loss(batch))

    x = new_lp.clone().requires_grad_()
    grpo_loss(x, old_lp, ref_lp, adv, mask, EPS, BETA).backward()
    assert torch.all(x.grad[~mask] == 0)


def test_kl_is_zero_when_policy_equals_reference(batch):
    new_lp, old_lp, _, _, mask = batch
    zero_adv = torch.zeros(4)
    assert loss((new_lp, old_lp, new_lp, zero_adv, mask)).abs().item() < 1e-7


# --- clip counting (the mechanism metric) ---

def test_clip_counts_known_example():
    """rho = 1.25 on a good answer and 0.75 on a bad one, at p_old = 0.05 / 0.3 / 0.9."""
    old_lp = torch.log(torch.tensor([[0.05, 0.3, 0.9], [0.05, 0.3, 0.9]]))
    new_lp = old_lp.clone()
    new_lp[0] += math.log(1.25)
    new_lp[1] += math.log(0.75)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3, dtype=torch.bool)

    def upper(mode, c=1.0):
        counts = clip_counts(new_lp, old_lp, adv, mask, mode, EPS, 0.28, c)
        return [counts[b][1] for b in ("lt0.1", "0.1-0.5", "gt0.5")]

    assert upper("symmetric") == [1, 1, 1]          # bound 1.20 everywhere
    assert upper("decoupled") == [0, 0, 0]          # bound 1.28 everywhere
    assert upper("prob_adaptive") == [0, 0, 1]      # bounds 1.39 / 1.34 / 1.22: only the confident token clips
    lower = [clip_counts(new_lp, old_lp, adv, mask, "symmetric", EPS, 0.28, 1.0)[b][2]
             for b in ("lt0.1", "0.1-0.5", "gt0.5")]
    assert lower == [1, 1, 1]                       # 0.75 < 0.80 in every bucket
