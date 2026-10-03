"""Greedy pass@1 on a fixed set of GSM8K test questions."""

import torch

from grpo.data import build_prompt
from grpo.reward import extract_answer, reward


@torch.no_grad()
def evaluate(model, tokenizer, examples, max_new_tokens, batch_size=16):
    """Return a dict of metrics plus a few sample completions."""
    model.eval()
    n_correct, n_format, n_trunc = 0, 0, 0
    samples = []

    eos_ids = model.generation_config.eos_token_id
    eos_ids = torch.tensor(eos_ids if isinstance(eos_ids, list) else [eos_ids], device=model.device)            
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [build_prompt(tokenizer, ex["question"]) for ex in batch]
        # Left padding so every prompt ends exactly where generation starts.
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )
        completions = out[:, inputs["input_ids"].shape[1] :]

        for ex, comp in zip(batch, completions):
            text = tokenizer.decode(comp, skip_special_tokens=True)
            r = reward(text, ex["answer"])
            n_correct += r >= 1.0
            n_format += r > 0.0
            n_trunc += not torch.isin(comp, eos_ids).any().item()   
            if len(samples) < 5:
                samples.append([ex["question"], ex["answer"], extract_answer(text), r, text])

    n = len(examples)
    return {
        "eval/accuracy": n_correct / n,
        "eval/format_rate": n_format / n,
        "eval/truncated": n_trunc / n,
    }, samples
