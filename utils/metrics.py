"""Image quality metrics."""

import numpy as np
import torch

def compute_metrics(pred: torch.Tensor, ref: torch.Tensor):
    """
    Compute RMSE, PSNR and SSIM between a prediction and a reference image.

    Both tensors are normalised to [0, 1] before comparison.

    Args:
        pred (torch.Tensor): Predicted image tensor (any shape).
        ref (torch.Tensor): Reference image tensor (any shape, same as pred).

    Returns:
        tuple[float, float, float]: ``(rmse, psnr, ssim)``.
        ``ssim`` is ``float("nan")`` when *scikit-image* is not installed.
    """
    # Move to CPU and convert to NumPy for metric computation.
    p = pred.detach().float().cpu().squeeze().numpy()
    r = ref.detach().float().cpu().squeeze().numpy()

    # Normalize to [0, 1] for metric computation.
    eps = 1e-8
    p = (p - p.min()) / (p.max() - p.min() + eps)
    r = (r - r.min()) / (r.max() - r.min() + eps)

    # Compute metrics.
    mse  = float(np.mean((p - r) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-12)))

    # Compute SSIM
    try:
        from skimage.metrics import structural_similarity
        
        ssim = float(structural_similarity(p, r, data_range=1.0))

    except Exception:
        ssim = float("nan")

    return rmse, psnr, ssim
