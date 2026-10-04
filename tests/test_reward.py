"""The reward: a wrong reward trains the model toward the wrong thing without any error."""

import pytest

from grpo.data import parse_ground_truth
from grpo.reward import extract_answer, reward


@pytest.mark.parametrize("text, expected", [
    ("so 48 + 24 = 72\n#### 72", "72"),
    ("#### 1,200 clips", "1200"),
    ("#### -5", "-5"),
    ("#### $18.50", "18.50"),
    ("#### 72.", "72"),
    ("#### 48\n#### 48\n#### 4", "48"),        # looping, truncated answer: the FIRST #### counts
    ("#### 36\n#### 48", "36"),                 # two different answers: the first one counts
    ("The answer is 72", None),                 # right number, wrong format
    ("####", None),
    ("#### no number here", None),
    ("", None),
    # Qwen writes Markdown: "####" is also a level-4 heading, which must not be read as the answer
    ("#### Step 2: 3 x 600 = 1800 meters\n...\n#### 540", "540"),
    ("#### Step 1: 3 sprints a day\nSo the total is **540 meters**.", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize("text, truth, expected", [
    ("#### 72", "72", 1.1),
    ("#### 72.0", "72", 1.1),                  # compared as numbers
    ("#### 1,200", "1200", 1.1),
    ("#### -5", "-5", 1.1),
    ("#### 50", "72", 0.1),                    # format only
    ("The answer is 72", "72", 0.0),           # no format, no credit, even with the right number
    ("72 is in the reasoning\n#### 50", "72", 0.1),
    ("", "72", 0.0),
])
def test_reward(text, truth, expected):
    assert reward(text, truth) == pytest.approx(expected)


def test_parse_ground_truth():
    assert parse_ground_truth("48/2 = 24\n#### 72") == "72"
    assert parse_ground_truth("...\n#### 1,200") == "1200"


@pytest.mark.xfail(reason="known gap: a unicode minus (U+2212) is not parsed as a sign, so -5 reads as 5")
def test_unicode_minus():
    assert extract_answer("#### −5") == "-5"
