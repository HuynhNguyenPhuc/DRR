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
    # DiffDRR places the source at origin and the camera looks along +Y.
    # To support arbitrary view_dir and up_vec, we compute the rotation R_cw
    # from World to DiffDRR Camera space, and the translation T_c.
    import numpy as np
    
    d = np.array([geo.view_dir_x, geo.view_dir_y, geo.view_dir_z], dtype=np.float32)
    d = d / np.linalg.norm(d)
    
    u = np.array([geo.up_vec_x, geo.up_vec_y, geo.up_vec_z], dtype=np.float32)
    u = u / np.linalg.norm(u)
    
    r = np.cross(d, u)
    r = r / np.linalg.norm(r)
    
    # Re-orthogonalize u
    u = np.cross(r, d)
    u = u / np.linalg.norm(u)
    
    # R_cw maps from World to DiffDRR Camera Space (where d maps to +Y, u to +Z)
    R_cw_np = np.stack([r, d, u], axis=0)
    R_cw = torch.from_numpy(R_cw_np).to(device)
    
    try:
        from pytorch3d.transforms import matrix_to_euler_angles
        # DiffDRR uses PyTorch3D transforms (P @ R + T), so we must pass the transpose of R_cw
        rot = matrix_to_euler_angles(R_cw.T, "ZXY").unsqueeze(0)
    except ImportError:
        logger.warning("[DiffDRR] pytorch3d not available for matrix_to_euler_angles, falling back to identity rotation.")
        rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # Compute T_c: we want World isocenter I_w to map to Camera isocenter [0, SAD, 0]
    I_w = torch.tensor([geo.isocenter_x, geo.isocenter_y, geo.isocenter_z], device=device, dtype=torch.float32)
    I_c = torch.tensor([0.0, geo.sad, 0.0], device=device, dtype=torch.float32)
    
    T_c = I_c - torch.matmul(R_cw, I_w)
    xyz = T_c.unsqueeze(0)

    return drr_wrapper, rot, xyz
