"""Loss functions."""

import torch


def zncc_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the Zero-mean Normalised Cross-Correlation (ZNCC) loss between two images.
    ZNCC is in the range [-1, 1], where 1 indicates perfect correlation and -1 indicates perfect anti-correlation.
    This implementation computes the negative ZNCC, so that it can be minimised by a gradient-descent optimiser. 

    Args:
        a (torch.Tensor): The rendered image.
        b (torch.Tensor): The target image to compare against.

    Returns:
        torch.Tensor: The mean negative ZNCC loss across the batch.
    """
    # Reshape to (B, N) and convert to float for numerical stability.
    a = a.reshape(a.shape[0], -1).float()
    b = b.reshape(b.shape[0], -1).float()

    # Subtract mean per image to get zero-mean vectors.
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)

    # Compute the batch-averaged negative ZNCC.
    denom = ((a ** 2).sum(1) * (b ** 2).sum(1)).sqrt().clamp(min=1e-8)
    
    return (-(a * b).sum(1) / denom).mean()
