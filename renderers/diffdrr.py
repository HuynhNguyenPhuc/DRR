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

    # Wrap the DiffDRR object to fix the horizontal flip mismatch.
    # DiffDRR uses a right-handed camera (cross(right, up) = ray), while medical 
    # DRR conventions use a left-handed camera (right = cross(view, up)). 
    # This means DiffDRR's native output is inherently horizontally mirrored.
    class DRRWrapper(torch.nn.Module):
        def __init__(self, drr_module):
            super().__init__()
            self.drr_module = drr_module

        def forward(self, *args, **kwargs):
            img = self.drr_module(*args, **kwargs)
            return torch.flip(img, dims=[-1])

    drr_wrapper = DRRWrapper(drr)

    # Initial rotation and translation estimates
    import numpy as np
    
    d = np.array([geo.view_dir_x, geo.view_dir_y, geo.view_dir_z], dtype=np.float32)
    d = d / np.linalg.norm(d)
    
    u = np.array([geo.up_vec_x, geo.up_vec_y, geo.up_vec_z], dtype=np.float32)
    u = u / np.linalg.norm(u)
    
    # Re-orthogonalize u
    r = np.cross(d, u)
    r = r / np.linalg.norm(r)
    u = np.cross(r, d)
    
    # DiffDRR's unreoriented base camera is effectively Left-Handed, with:
    # ray=[0,-1,0], up=[0,0,1], right=[-1,0,0]
    # We want ray=d, up=u, right=r.
    # Therefore rotmat must map [0,-1,0]->d, [0,0,1]->u, [-1,0,0]->r.
    # So the columns of rotmat are [-r, -d, u]
    rotmat_np = np.stack([-r, -d, u], axis=1)
    rotmat = torch.from_numpy(rotmat_np).to(device)
    
    try:
        from pytorch3d.transforms import matrix_to_euler_angles
        rot = matrix_to_euler_angles(rotmat, "ZXY").unsqueeze(0)
    except ImportError:
        logger.warning("[DiffDRR] pytorch3d not available, falling back to identity.")
        rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # Compute translation in the camera frame
    # DiffDRR applies camera_center = rotmat @ translation
    # We want camera_center = isocenter - sad * d
    # Therefore translation = rotmat.T @ (isocenter - sad * d)
    I_w = np.array([geo.isocenter_x, geo.isocenter_y, geo.isocenter_z], dtype=np.float32)
    t_x = -np.dot(r, I_w)
    t_y = -np.dot(d, I_w) + geo.sad
    t_z =  np.dot(u, I_w)
    
    xyz = torch.tensor([[t_x, t_y, t_z]], device=device, dtype=torch.float32)

    return drr_wrapper, rot, xyz
