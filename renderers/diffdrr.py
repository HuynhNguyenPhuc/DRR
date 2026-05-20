"""Renderer factory functions for DiffDRR (Siddon)."""

import torch

from .config import GeometryConfig, DEFAULT_GEOMETRY
from utils.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def build_diffdrr_renderer(
    image_size: int,
    ct_size: int,
    subject,
    voxel_spacing: float,
    device: torch.device,
    geometry: GeometryConfig = None,
):
    """
    Instantiate a ``DRR`` renderer using Siddon's ray-casting method.

    The source-to-detector distance is fixed at 1020 mm (standard chest
    radiography geometry).

    Args:
        image_size (int): Square output image resolution in pixels.
        ct_size (int): Isotropic CT volume side length in voxels.
        subject: torchio ``Subject`` containing the density volume.
        voxel_spacing (float): Isotropic voxel size in mm.
        device (torch.device): Target compute device.

    Returns:
        tuple:
            - **drr** (DRR | None): Ready-to-use DRR model, or ``None`` when
              the import fails.
            - **rot** (torch.Tensor | None): Initial Euler-angle rotation
              ``(1, 3)`` in radians.
            - **xyz** (torch.Tensor | None): Initial translation ``(1, 3)``
              in mm.
    """
    try:
        from diffdrr.drr import DRR
        
    except ImportError as exc:
        logger.warning("[DiffDRR] Import failed: %s", exc)
        return None, None, None

    geo = geometry or DEFAULT_GEOMETRY

    # Calculate Field of View and pixel spacing
    fov  = ct_size * float(voxel_spacing)
    delx = fov / image_size

    drr = DRR(
        subject,
        sdd    = geo.sdd,
        height = image_size,
        delx   = delx,
    ).to(device)

    # Wrap the DiffDRR object to fix the horizontal flip mismatch
    class DRRWrapper(torch.nn.Module):
        def __init__(self, drr_module):
            super().__init__()
            self.drr_module = drr_module

        def forward(self, *args, **kwargs):
            img = self.drr_module(*args, **kwargs)
            return torch.flip(img, dims=[-1])

    drr_wrapper = DRRWrapper(drr)

    # Initial rotation and translation estimates
    # DiffDRR places the source at origin and moves the volume.
    # To match S -> I direction, the volume (at isocenter I) is placed at `SAD * view_dir`.
    rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    xyz = torch.tensor([[
        geo.sad * geo.view_dir_x,
        geo.sad * geo.view_dir_y,
        geo.sad * geo.view_dir_z
    ]], device=device)

    return drr_wrapper, rot, xyz
