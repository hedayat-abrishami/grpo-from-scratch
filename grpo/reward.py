import re

# "####" immediately followed by the number (optionally "$"). A bare "####" is also a Markdown
# heading ("#### Step 2: ..."), so the number must come straight after it, not anywhere later.
ANSWER = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")


def extract_answer(text):
    """Return the number in the first '#### <number>' as a string without commas, or None."""
    match = ANSWER.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "")


def reward(text, ground_truth):
    """1.0 for the correct number, plus 0.1 for using the '#### <number>' format."""
    pred = extract_answer(text=text)
    if pred is None:
        return 0.0
    r = 0.1
    if float(pred) == float(ground_truth):
        r += 1.0
    return r
