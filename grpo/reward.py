import re

# Qwen-Instruct's native answer format: \boxed{...}, allowing one level of nested braces (\boxed{18 \text{ dollars}}).
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def extract_answer(text):
    """Return the first number inside the LAST \\boxed{...}, without commas, or None."""
    boxes = BOXED.findall(text)
    if not boxes:
        return None
    match = NUMBER.search(boxes[-1])
    if match is None:
        return None
    return match.group().replace(",", "")


def reward(text, ground_truth):
    """1.0 for the correct number, plus 0.1 for putting a number in \\boxed{}."""
    pred = extract_answer(text=text)
    if pred is None:
        return 0.0
    r = 0.1
    if float(pred) == float(ground_truth):
        r += 1.0
    return r
