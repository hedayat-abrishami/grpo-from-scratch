import re

NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def extract_answer(text):
    """Return the number after the first '####' as a string without commas, or None."""
    if "####" not in text:
        return None
    after = text.split("####", 1)[1]
    match = NUMBER.search(after)
    if match is None:
        return None
    return match.group().replace(",", "")


def reward(text, ground_truth):
    """1.0 for the correct number, plus 0.1 for using the '#### <number>' format."""
    pred = extract_answer(text=text)
    if pred is None:
        return 0.0
    r = 0.1
    if float(pred) == float(ground_truth):
        r += 1.0
    return r
