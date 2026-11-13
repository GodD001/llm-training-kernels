"""Autoregressive sampling with temperature and top-p (nucleus) filtering."""

import torch

from llmtk.model.functional import softmax
from llmtk.model.transformer import TransformerLM


@torch.no_grad()
def decode(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample up to `max_new_tokens` tokens after `prompt`; returns only the generated ids."""
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)  # (t,) -> (1, t)

    x = prompt.to(device)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = model(x)  # (1, cur_len, vocab_size)
        next_logits = logits[0, -1, :]
        probs = softmax(next_logits / temperature, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative_probs > top_p
            # Shift right so the first token that crosses top_p is kept.
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            sorted_probs[remove_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, sorted_probs)

        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break

        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        x = torch.cat([x, next_tok], dim=1)

    return torch.tensor(generated, dtype=torch.long)
