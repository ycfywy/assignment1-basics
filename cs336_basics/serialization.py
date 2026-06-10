"""
CS336 Assignment 1 - Section 3: Model Serialization.

Implements:
- save_checkpoint: Save model, optimizer, and iteration state
- load_checkpoint: Restore from checkpoint
"""

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Save model, optimizer state and iteration to a checkpoint file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: Current training iteration.
        out: Path or file-like object to write to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model and optimizer state from a checkpoint file.

    Args:
        src: Path or file-like object to read from.
        model: Model to restore state into.
        optimizer: Optimizer to restore state into.

    Returns:
        The iteration number stored in the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
