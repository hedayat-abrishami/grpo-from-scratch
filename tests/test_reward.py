"""The reward: a wrong reward trains the model toward the wrong thing without any error."""

import pytest

from grpo.data import SYSTEM_PROMPT, parse_ground_truth
from grpo.reward import extract_answer, reward


@pytest.mark.parametrize("text, expected", [
    ("so 9 x 2 = 18\n\\boxed{18}", "18"),
    ("the answer is \\(\\boxed{18}\\).", "18"),       # inline math, as Qwen writes it
    ("\\[\n\\boxed{267}\n\\]", "267"),                  # display math
    ("\\boxed{1,200}", "1200"),
    ("\\boxed{-5}", "-5"),
    ("\\boxed{\\$18.50}", "18.50"),
    ("\\boxed{18 \\text{ dollars}}", "18"),            # nested braces
    ("first try \\boxed{3}, actually \\boxed{5}", "5"),  # the LAST box is the answer
    ("\\boxed{}", None),
    ("\\boxed{x}", None),
    ("So the total is **540 meters**.", None),         # right number, wrong format
    ("#### 72", None),                                  # the old format no longer counts
    ("", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize("text, truth, expected", [
    ("\\boxed{72}", "72", 1.1),
    ("\\boxed{72.0}", "72", 1.1),                       # compared as numbers
    ("\\boxed{1,200}", "1200", 1.1),
    ("\\boxed{-5}", "-5", 1.1),
    ("\\boxed{50}", "72", 0.1),                         # format only
    ("The answer is 72", "72", 0.0),                    # no format, no credit, even with the right number
    ("72 is in the reasoning\n\\boxed{50}", "72", 0.1),
    ("", "72", 0.0),
])
def test_reward(text, truth, expected):
    assert reward(text, truth) == pytest.approx(expected)


def test_prompt_asks_for_the_format_the_parser_reads():
    assert "\\boxed{}" in SYSTEM_PROMPT


def test_parse_ground_truth():
    assert parse_ground_truth("48/2 = 24\n#### 72") == "72"
    assert parse_ground_truth("...\n#### 1,200") == "1200"


@pytest.mark.xfail(reason="known gap: a unicode minus (U+2212) is not parsed as a sign, so -5 reads as 5")
def test_unicode_minus():
    assert extract_answer("\\boxed{\u22125}") == "-5"
