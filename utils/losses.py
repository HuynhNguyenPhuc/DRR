"""Loss functions."""

import torch


def zncc_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the batch-averaged negative Zero-Normalised Cross-Correlation.

    ZNCC ∈ [-1, 1]; this returns the *negated* value so it can be minimised
    by a gradient-descent optimiser.

    Args:
        a (torch.Tensor): Predicted image, shape ``(B, *)``.
        b (torch.Tensor): Reference image, shape ``(B, *)``.

    Returns:
        torch.Tensor: Scalar loss value (differentiable).
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
