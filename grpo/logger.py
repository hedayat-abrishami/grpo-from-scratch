"""Append-only CSV metrics, kept next to the checkpoints so they survive Colab disconnects."""

import csv
import json
import os


class CSVLogger:
    def __init__(self, path, resume_step=None):
        """On resume, drop rows logged after the checkpoint, since those steps will be re-run."""
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if resume_step is not None and os.path.exists(path):
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["step"]) <= resume_step]
            self._rewrite(rows)

    def _rewrite(self, rows):
        with open(self.path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def log(self, metrics, step):
        row = {"step": step, **metrics}
        is_new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def log_samples(path, step, samples):
    """One JSON line per sample: the model's actual answers at each eval."""
    with open(path, "a") as f:
        for question, answer, predicted, r, completion in samples:
            f.write(json.dumps({"step": step, "question": question, "answer": answer,
                                "predicted": predicted, "reward": r, "completion": completion}) + "\n")
