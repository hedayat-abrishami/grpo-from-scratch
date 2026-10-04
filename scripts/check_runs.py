"""One line per run folder: status, speed, and the numbers worth watching.

Usage:  python scripts/check_runs.py ~/runs            (all runs)
        python scripts/check_runs.py ~/runs rehearsal   (only runs whose name contains "rehearsal")
"""

import csv
import json
import os
import sys


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    root = os.path.expanduser(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"{'run':16s} {'state':7s} {'clip':13s} {'lr':>7s} {'steps':>5s} {'s/step':>6s} {'reward':>6s} "
          f"{'entropy':>7s} {'kl':>8s} {'kl_max':>8s} {'up<0.1':>6s} {'up>0.5':>6s} {'eval':>5s} {'final':>5s}  dirty gpu")
    for name in sorted(os.listdir(root)):
        run = os.path.join(root, name)
        if pattern not in name or not os.path.isdir(run):
            continue
        cfg = read_json(os.path.join(run, "config.json"))
        info = read_json(os.path.join(run, "run_info.json"))
        m = read_csv(os.path.join(run, "metrics.csv"))
        last = m[-10:]                                          # recent behaviour, not the whole run
        ev = read_csv(os.path.join(run, "eval.csv"))
        fin = read_csv(os.path.join(run, "eval_final.csv"))
        state = "done" if os.path.exists(os.path.join(run, ".done")) else "partial"
        kl_max = max((float(r["train/kl_max"]) for r in m if r.get("train/kl_max")), default=float("nan"))
        print(f"{name:16s} {state:7s} {cfg.get('clip', '?'):13s} {cfg.get('lr', float('nan')):7.0e} {len(m):5d} "
              f"{mean(m, 'time/step_s'):6.0f} {mean(last, 'train/reward'):6.3f} {mean(last, 'train/entropy'):7.3f} "
              f"{mean(last, 'train/kl'):8.1e} {kl_max:8.1e} {mean(m, 'clip/upper_lt0.1'):6.3f} "
              f"{mean(m, 'clip/upper_gt0.5'):6.3f} "
              f"{float(ev[-1]['eval/accuracy']) if ev else float('nan'):5.2f} "
              f"{float(fin[-1]['eval/accuracy']) if fin else float('nan'):5.2f}  "
              f"{str(info.get('git_dirty', '?')):5s} {info.get('gpu')}")


if __name__ == "__main__":
    main()
