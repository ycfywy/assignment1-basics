"""
CS336 Assignment 1 - Section 3: Optimizer and Learning Rate Schedule.

Implements:
- AdamW optimizer
- Cosine learning rate schedule with linear warmup
"""

import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer implementation.

    Decoupled weight decay regularization (Loshchilov & Hutter, 2019).
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # Compute step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Parameter update (without weight decay)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Decoupled weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        it: Current iteration.
        max_learning_rate: Maximum (peak) learning rate.
        min_learning_rate: Minimum (final) learning rate.
        warmup_iters: Number of warmup iterations.
        cosine_cycle_iters: Total number of iterations for cosine cycle.

    Returns:
        Learning rate at the given iteration.
    """
    # Phase 1: Linear warmup
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    # Phase 3: After cosine cycle, constant at min_lr
    if it >= cosine_cycle_iters:
        return min_learning_rate

    # Phase 2: Cosine decay
    # Progress through the cosine phase (0 to 1)
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    # Cosine decay from max to min
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
