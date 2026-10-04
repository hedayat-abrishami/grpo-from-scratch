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

    eos_ids = model.generation_config.eos_token_id
    eos_ids = torch.tensor(eos_ids if isinstance(eos_ids, list) else [eos_ids], device=device)
    is_eos = torch.isin(completions, eos_ids)
    has_eos = is_eos.any(dim=1)
    eos_idx = torch.where(has_eos, is_eos.int().argmax(dim=1), completions.shape[1] - 1)
    positions = torch.arange(completions.shape[1], device=device)
    mask = positions.unsqueeze(0) <= eos_idx.unsqueeze(1)

    return inputs["input_ids"], completions, mask, ~has_eos

def token_logprobs(model, prompt_ids, completions, return_entropy=False, chunk=0):
    """Log-prob of each completion token under `model`. Shape (G, completion_len).

    chunk > 0 runs the rows `chunk` at a time and joins the results: same values, lower peak memory.
    Only useful without gradients; with gradients every chunk's graph is kept until backward anyway.
    """
    G, L = completions.shape
    if 0 < chunk < G:
        parts = [token_logprobs(model, prompt_ids, completions[i : i + chunk], return_entropy)
                 for i in range(0, G, chunk)]
        logp = torch.cat([p[0] for p in parts])
        entropy = torch.cat([p[1] for p in parts]) if return_entropy else None
        return logp, entropy

    full = torch.cat([prompt_ids.repeat(G, 1), completions], dim=1)

    logits = model(full, logits_to_keep=L+1).logits[:, :-1].float()
    chosen = logits.gather(2, completions.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits, dim=-1)
    logp = chosen - lse
    if not return_entropy:
        return logp, None
    entropy = lse - (torch.softmax(logits, dim=-1) * logits).sum(-1)  # (G, L), in nats
    return logp, entropy