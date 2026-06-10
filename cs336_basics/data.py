"""
CS336 Assignment 1 - Section 3: Data Loading Utilities.

Implements:
- get_batch: Sample random batches for language model training
"""

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of input sequences and labels for language modeling.

    For each example in the batch, randomly select a starting index i such that
    dataset[i:i+context_length+1] is valid, then:
        x = dataset[i:i+context_length]
        y = dataset[i+1:i+context_length+1]

    Args:
        dataset: 1D numpy array of integer token IDs.
        batch_size: Number of sequences in the batch.
        context_length: Length of each sequence.
        device: PyTorch device string.

    Returns:
        (x, y) both of shape (batch_size, context_length) as LongTensors on device.
    """
    # Maximum valid starting index
    max_start = len(dataset) - context_length - 1
    # Random starting indices
    start_indices = np.random.randint(0, max_start + 1, size=(batch_size,))

    x = np.stack([dataset[i:i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return x, y
