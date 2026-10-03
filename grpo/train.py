import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from grpo.data import load_gsm8k
from grpo.eval import evaluate
from grpo.logger import CSVLogger, log_samples
from grpo.loss import BUCKETS, advantages, clip_counts, grpo_loss
from grpo.reward import reward
from grpo.sampling import sample_group, token_logprobs
from grpo.utils import get_device, seed_everything


@dataclass
class Config:
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    G: int = 4                      # completions per question
    B: int = 4                      # questions per optimizer step
    K: int = 1                      # optimizer updates per batch of rollouts (1 = plain GRPO)
    steps: int = 12
    max_new_tokens: int = 256
    lr: float = 1e-6
    eps: float = 0.2
    clip: str = "symmetric"         # symmetric | decoupled | prob_adaptive
    eps_high: float = 0.28          # upper clip for "decoupled" (DAPO Clip-Higher)
    c: float = 1.0                  # strength of the per-token upper bound for "prob_adaptive"
    beta: float = 0.04
    seed: int = 0
    device: str = ""                # "" = auto (cuda > mps > cpu)
    run_name: str = "debug"
    ckpt_dir: str = ""              # "" = no checkpoints; logs then go to runs/<run_name>
    ckpt_every: int = 25
    eval_every: int = 50            # 0 = no eval
    eval_n: int = 100               # first N test questions, the same every time
    final_eval_n: int = 1319        # questions in the end-of-run eval (0 = skip); 1319 = the full test set


def parse_args(argv=None):
    """Settings come from, in increasing priority: Config defaults, the --config JSON file, command-line flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="JSON file with Config fields, e.g. configs/arm_a_symmetric.json")
    for f in fields(Config):
        # SUPPRESS: a flag that isn't given doesn't appear at all, so it can't overwrite the JSON value.
        if f.type is bool:
            parser.add_argument(f"--{f.name}", action="store_true", default=argparse.SUPPRESS)
        else:
            parser.add_argument(f"--{f.name}", type=f.type, default=argparse.SUPPRESS)
    flags = vars(parser.parse_args(argv))

    values = {}
    config_file = flags.pop("config")
    if config_file:
        with open(config_file) as f:
            values.update(json.load(f))
    values.update(flags)
    return Config(**values)  # an unknown or misspelled key raises TypeError instead of being ignored


def write_run_info(run_dir, start_step):
    """Which code and environment produced this run. A resumed run gets its own file, in case the code changed."""
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    status = sh("git", "status", "--porcelain")
    info = {
        "started_at_step": start_step,
        "git_commit": sh("git", "rev-parse", "HEAD"),
        "git_dirty": None if status is None else status != "",  # True = uncommitted changes were present
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    suffix = "" if start_step == 0 else f"_resume{start_step}"
    with open(os.path.join(run_dir, f"run_info{suffix}.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(run_dir, f"pip_freeze{suffix}.txt"), "w") as f:
        f.write(sh(sys.executable, "-m", "pip", "freeze") or "")


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = get_device(cfg.device or None)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32).to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    train = load_gsm8k("train")
    eval_set = load_gsm8k("test")[: cfg.eval_n]

    # Resume if this run already has a checkpoint.
    start_step = 0
    ckpt = checkpoint_path(cfg.ckpt_dir, cfg.run_name) if cfg.ckpt_dir else None
    if ckpt and os.path.exists(ckpt):
        start_step = load_checkpoint(ckpt, model, optimizer, device)
        print(f"resumed from {ckpt} at step {start_step}")

    # Logs live next to the checkpoint, so they survive a Colab disconnect along with it.
    run_dir = os.path.join(cfg.ckpt_dir or "runs", cfg.run_name)
    resume = start_step if start_step > 0 else None
    train_log = CSVLogger(os.path.join(run_dir, "metrics.csv"), resume)
    eval_log = CSVLogger(os.path.join(run_dir, "eval.csv"), resume)
    # The fully resolved settings: rerun this exact run with --config <run_dir>/config.json
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    write_run_info(run_dir, start_step)
    print(f"logging to {run_dir}")

    def run_eval(step):
        t0 = time.time()
        metrics, samples = evaluate(model, tok, eval_set, cfg.max_new_tokens)
        print(f"eval @ step {step} | acc {metrics['eval/accuracy']:.3f} | format {metrics['eval/format_rate']:.3f} "
              f"| trunc {metrics['eval/truncated']:.2f} | {time.time() - t0:.0f}s")
        eval_log.log(metrics, step)
        log_samples(os.path.join(run_dir, "eval_samples.jsonl"), step, samples)

    if cfg.eval_every and start_step == 0:
        run_eval(0)

    for step in range(start_step, cfg.steps):
        t0 = time.time()
        all_rewards, all_trunc, all_len, all_kl, all_ent, n_signal, total_loss = [], [], [], [], [], 0, 0.0

        rollouts = []
        for _ in range(cfg.B):
            ex = random.choice(train)

            # 1. sample a group
            model.eval()
            with torch.no_grad():
                prompt_ids, completions, mask, truncated = sample_group(model, tok, ex["question"], cfg.G, cfg.max_new_tokens)
                texts = [tok.decode(completions[i][mask[i]], skip_special_tokens=True) for i in range(cfg.G)]
                rewards = torch.tensor([reward(t, ex["answer"]) for t in texts], device=device)
                adv = advantages(rewards)
                old_lp, ent = token_logprobs(model, prompt_ids, completions, return_entropy=True)
                ref_lp, _ = token_logprobs(ref_model, prompt_ids, completions)
                rollouts.append(dict(prompt_ids=prompt_ids, completions=completions, mask=mask,
                                     adv=adv, old_lp=old_lp, ref_lp=ref_lp))

            all_rewards.append(rewards)
            all_trunc.append(truncated.float())
            all_len.append(mask.sum(1).float())
            n_signal += int(rewards.std() > 1e-6)
            all_ent.append(((ent * mask).sum() / mask.sum()).item())  # mean over real tokens only

        # 2. K optimizer updates on the same rollouts
        clip_tot = {name: [0, 0, 0] for name, _, _ in BUCKETS}  # per pi_old bucket: tokens, upper-clipped, lower-clipped
        for k in range(cfg.K):
            optimizer.zero_grad()
            for ro in rollouts:
                new_lp, _ = token_logprobs(model, ro["prompt_ids"], ro["completions"])
                if k > 0:  # on pass 0 rho == 1, so nothing can be clipped
                    for name, (n, up, low) in clip_counts(new_lp, ro["old_lp"], ro["adv"], ro["mask"],
                                                          cfg.clip, cfg.eps, cfg.eps_high, cfg.c).items():
                        clip_tot[name][0] += n
                        clip_tot[name][1] += up
                        clip_tot[name][2] += low
                if k == 0:  # before the first update the policy still equals pi_old
                    max_dev = (torch.exp(new_lp - ro["old_lp"]) - 1)[ro["mask"]].abs().max().item()
                    assert max_dev < 1e-3, f"rho != 1 before update: {max_dev}"

                loss = grpo_loss(new_lp, ro["old_lp"], ro["ref_lp"], ro["adv"], ro["mask"],
                                 cfg.eps, cfg.beta, cfg.clip, cfg.eps_high, cfg.c) / cfg.B
                loss.backward()
                if k == 0:
                    with torch.no_grad():
                        d = ro["ref_lp"] - new_lp
                        kl = ((torch.exp(d) - d - 1) * ro["mask"]).sum() / ro["mask"].sum()
                    all_kl.append(kl.item())
                    total_loss += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        r = torch.cat(all_rewards)
        metrics = {
            "train/reward": r.mean().item(),
            "train/accuracy": (r >= 1).float().mean().item(),
            "train/format_rate": (r > 0).float().mean().item(),
            "train/loss": total_loss,
            "train/grad_norm": grad_norm.item(),
            "train/length": torch.cat(all_len).mean().item(),
            "train/truncated": torch.cat(all_trunc).mean().item(),
            "train/signal_frac": n_signal / cfg.B,
            "train/kl": sum(all_kl) / len(all_kl),
            "train/entropy": sum(all_ent) / len(all_ent),
            **{f"clip/upper_{name}": up / max(n, 1) for name, (n, up, low) in clip_tot.items()},
            **{f"clip/lower_{name}": low / max(n, 1) for name, (n, up, low) in clip_tot.items()},
            "time/step_s": time.time() - t0,
        }
        print(f"step {step:3d} | reward {metrics['train/reward']:.3f} | acc {metrics['train/accuracy']:.3f} "
              f"| loss {total_loss:+.4f} | grad {metrics['train/grad_norm']:.2f} | len {metrics['train/length']:.0f} "
              f"| trunc {metrics['train/truncated']:.2f} | signal {n_signal}/{cfg.B} | ent {metrics['train/entropy']:.3f} | kl {metrics['train/kl']:.2e} "
              f"| {metrics['time/step_s']:.0f}s")
        done = step + 1
        train_log.log(metrics, done)
        if cfg.eval_every and (done % cfg.eval_every == 0 or done == cfg.steps):
            run_eval(done)
        if ckpt and (done % cfg.ckpt_every == 0 or done == cfg.steps):
            save_checkpoint(ckpt, model, optimizer, done, asdict(cfg))
            print(f"saved checkpoint at step {done}")

    # The number for the results table: once, at the end, on the full test set.
    if cfg.final_eval_n:
        t0 = time.time()
        metrics, samples = evaluate(model, tok, load_gsm8k("test")[: cfg.final_eval_n], cfg.max_new_tokens)
        print(f"final eval on {cfg.final_eval_n} questions | acc {metrics['eval/accuracy']:.3f} "
              f"| format {metrics['eval/format_rate']:.3f} | trunc {metrics['eval/truncated']:.2f} | {time.time() - t0:.0f}s")
        final_csv = os.path.join(run_dir, "eval_final.csv")
        final_samples = os.path.join(run_dir, "eval_final_samples.jsonl")
        for path in (final_csv, final_samples):
            if os.path.exists(path):
                os.remove(path)  # a rerun replaces the result instead of adding to it
        CSVLogger(final_csv).log(metrics, cfg.steps)
        log_samples(final_samples, cfg.steps, samples)


if __name__ == "__main__":
    main()
