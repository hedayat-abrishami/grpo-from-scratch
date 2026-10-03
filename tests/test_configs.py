"""The experiment configs: valid, and the arms differ only in the clip settings."""

import glob
import json

import pytest

from grpo.train import Config

ARM_FILES = sorted(glob.glob("configs/arm_*.json"))
CLIP_FIELDS = {"clip", "eps_high", "c"}


def test_arm_configs_exist():
    assert len(ARM_FILES) == 5


@pytest.mark.parametrize("path", sorted(glob.glob("configs/*.json")))
def test_every_config_is_a_valid_config(path):
    Config(**json.load(open(path)))      # a misspelled key raises TypeError


def test_arms_differ_only_in_clip_settings():
    arms = [json.load(open(p)) for p in ARM_FILES]
    keys = set.union(*(set(a) for a in arms))
    differing = {k for k in keys if len({json.dumps(a.get(k)) for a in arms}) > 1}
    assert differing <= CLIP_FIELDS, f"arms also differ in {differing - CLIP_FIELDS}"
