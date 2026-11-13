"""AdamW optimizer (decoupled weight decay)."""

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW with the bias correction folded into the step size.

    Update order: Adam step on theta, then decoupled decay theta -= lr * wd * theta
    (decay uses the base lr, not the bias-corrected alpha_t).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)  # first moment
                    state["v"] = torch.zeros_like(p.data)  # second moment

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                a_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data -= a_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * wd * p.data

        return loss
