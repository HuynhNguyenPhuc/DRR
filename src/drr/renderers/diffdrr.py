"""Renderer factory functions for DiffDRR with Siddon's ray-casting method."""

import numpy as np
import torch

from .config import GeometryConfig, DEFAULT_GEOMETRY
from drr.utils.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


# Patch size threshold for memory-safe DiffDRR rendering. 
# Above this, the renderer will automatically chunk the output into smaller patches to avoid OOM errors. 
# Below this, it renders in one pass. This is a trade-off between memory usage and speed, as patching incurs some overhead.
_DIFFDRR_PATCH_THRESHOLD_PX = 256


def _diffdrr_memory_kwargs(image_size: int) -> dict:
    """
    Memory-safety kwargs for large DiffDRR renders.

    Args:
        image_size (int): The output resolution in pixels.

    Returns:
        dict: Keyword arguments for DiffDRR's ``DRR`` constructor.
    """
    if image_size > _DIFFDRR_PATCH_THRESHOLD_PX:
        return {
            "patch_size": _DIFFDRR_PATCH_THRESHOLD_PX,
            "checkpoint_gradients": True,
        }
    
    return {
        "patch_size": None, 
        "checkpoint_gradients": False
    }


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

    Args:
        image_size (int): The output resolution in pixels.
        ct_size (int): The CT volume side length in voxels.
        subject: torchio ``Subject`` holding the density volume.
        voxel_spacing (float): The voxel size in mm.
        device (torch.device): The target compute device.
        geometry (GeometryConfig): The X-ray setup; defaults to ``DEFAULT_GEOMETRY``.

    Returns:
        tuple: The DiffDRR ``DRR`` object, the rotation matrix (3x3), and the translation vector (3x1) for the camera.
    """
    # Optional dependency — degrade gracefully instead of crashing the caller.
    try:
        from diffdrr.drr import DRR
    except ImportError as exc:
        logger.warning("[DiffDRR] Import failed: %s", exc)
        return None, None, None

    geo = geometry or DEFAULT_GEOMETRY

    # ── Field of view / pixel spacing ───────────────────────────────────────
    fov  = ct_size * float(voxel_spacing)
    delx = fov / image_size

    drr = DRR(
        subject,
        sdd    = geo.sdd,
        height = image_size,
        delx   = delx,
        **_diffdrr_memory_kwargs(image_size),
    ).to(device)

    # ── Initial rotation estimate ───────────────────────────────────────────
    # DiffDRR's camera coordinate system is Right-Handed, with:
    # ray = -Y, up = +Z, right = -X
    d = np.array([geo.view_dir_x, geo.view_dir_y, geo.view_dir_z], dtype=np.float32)
    d = d / np.linalg.norm(d)

    u = np.array([geo.up_vec_x, geo.up_vec_y, geo.up_vec_z], dtype=np.float32)
    u = u / np.linalg.norm(u)

    r = np.cross(d, u)
    r = r / np.linalg.norm(r)
    u = np.cross(r, d)

    # DiffDRR's unreoriented base camera is effectively Left-Handed, with:
    # ray = +Y, up = +Z, right = +X
    # Therefore, to convert from our RH world coordinates to DiffDRR's LH camera coordinates
    rotmat_np = np.stack([-r, -d, u], axis=1)
    rotmat = torch.from_numpy(rotmat_np).to(device)

    try:
        from pytorch3d.transforms import matrix_to_euler_angles
        rot = matrix_to_euler_angles(rotmat, "ZXY").unsqueeze(0)
    except ImportError:
        logger.warning("[DiffDRR] pytorch3d not available, falling back to identity.")
        rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # ── Initial translation estimate ────────────────────────────────────────
    # DiffDRR applies camera_center = rotmat @ translation
    # We want camera_center = isocenter - sad * d
    # Therefore, translation = rotmat.T @ (isocenter - sad * d)
    I_w = np.array([geo.isocenter_x, geo.isocenter_y, geo.isocenter_z], dtype=np.float32)
    t_x = -np.dot(r, I_w)
    t_y = -np.dot(d, I_w) + geo.sad
    t_z =  np.dot(u, I_w)

    xyz = torch.tensor([[t_x, t_y, t_z]], device=device, dtype=torch.float32)

    return drr, rot, xyz
