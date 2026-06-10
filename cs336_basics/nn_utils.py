"""
CS336 Assignment 1 - Section 3: Neural Network Utilities.

Implements:
- softmax (numerically stable)
- cross_entropy loss
- gradient clipping
"""

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension to apply softmax over.

    Returns:
        Tensor of same shape with softmax applied along dim.
    """
    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(
    inputs: Tensor, targets: Tensor
) -> Tensor:
    """Compute average cross-entropy loss.

    Args:
        inputs: (batch_size, vocab_size) unnormalized logits.
        targets: (batch_size,) integer class indices.

    Returns:
        Scalar tensor with average cross-entropy loss.
    """
    # Numerically stable: log_softmax
    # log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
    x_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - x_max
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    # log_softmax for each example at the target index
    # shifted[i, targets[i]] - log_sum_exp[i]
    log_probs = shifted[torch.arange(inputs.shape[0], device=inputs.device), targets] - log_sum_exp
    return -log_probs.mean()


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """Clip gradients by global L2 norm.

    Args:
        parameters: Iterable of parameters.
        max_l2_norm: Maximum allowed L2 norm.

    Modifies parameter.grad in-place.
    """
    # Collect all gradients
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # Compute global L2 norm
    total_norm = torch.sqrt(sum(torch.sum(g * g) for g in grads))

    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / total_norm
        for g in grads:
            g.mul_(clip_coef)
