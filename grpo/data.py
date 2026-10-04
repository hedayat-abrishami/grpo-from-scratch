from datasets import load_dataset

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "Put the final answer, as a number only, in \\boxed{}."
)


def parse_ground_truth(solution):
    """GSM8K solutions end with '#### <number>'. Return that number as a string, commas removed."""
    return solution.split("####")[-1].strip().replace(",", "")


def load_gsm8k(split):
    """Return a list of {'question', 'answer'} dicts. 'answer' is the final number only."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {"question": ex["question"], "answer": parse_ground_truth(ex["answer"])}
        for ex in ds
    ]


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
