import torch
from grpo.data import build_prompt

def sample_group(model, tokenizer, question, G, max_new_tokens):

    device = model.device
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    out = model.generate(**inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        repetition_penalty=1.0,
                        num_return_sequences=G,)
    prompt_len = inputs["input_ids"].shape[1]
    completions = out[:,  prompt_len:]

    is_eos = completions == tokenizer.eos_token_id
    has_eos = is_eos.any(dim=1)
    eos_idx = torch.where(has_eos, is_eos.int().argmax(dim=1), completions.shape[1] - 1)
    positions = torch.arange(completions.shape[1], device=device)
    mask = positions.unsqueeze(0) <= eos_idx.unsqueeze(1)

    return inputs["input_ids"], completions, mask, ~has_eos

def token_logprobs(model, prompt_ids, completions):
    """Log-prob of each completion token under `model`. Shape (G, completion_len)."""
    G = completions.shape[0]
    prompt_len = prompt_ids.shape[1]

    full = torch.cat([prompt_ids.repeat(G, 1), completions], dim=1)
    logits = model(full).logits

    logits = logits[:, prompt_len - 1: -1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(2, completions.unsqueeze(-1)).squeeze(-1)