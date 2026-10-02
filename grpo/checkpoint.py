"""Save and resume training state: model, optimizer, step, RNG state."""

import os
import random

import numpy as np
import torch


def checkpoint_path(ckpt_dir, run_name):
    return os.path.join(ckpt_dir, run_name, "latest.pt")


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    # RNG states must be CPU ByteTensors, even when the checkpoint was loaded onto the GPU.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, model, optimizer, step, config):
    """Write to a temp file, then rename, so a disconnect mid-save can't corrupt the last good checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "rng": _rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, device):
    """Restore model, optimizer and RNG state in place. Returns the next step to run."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    _set_rng_state(ckpt["rng"])
    return ckpt["step"]
