"""DeepDRR physics-based renderer."""

from __future__ import annotations

import numpy as np
import torch

from utils.logger import get_logger
from .config import GeometryConfig, DEFAULT_GEOMETRY


# --- Logger --- #
logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# HU conversion utilities
# ═════════════════════════════════════════════════════════════════════════════

def _density_to_hu(volume_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalised density [0, 1] to approximate Hounsfield Units (HU).

    Linear mapping: 0 → -1000 HU (air), 1 → +3000 HU (cortical bone).
    For real CT data, original HU values should be used directly.

    Args:
        volume_tensor (torch.Tensor): Density volume ``(1, 1, D, H, W)``.

    Returns:
        np.ndarray: HU volume ``(D, H, W)`` as ``float32``.
    """
    vol_np = volume_tensor.detach().cpu().squeeze().numpy()  # (D, H, W)
    hu = vol_np * 4_000.0 - 1_000.0
    return hu.astype(np.float32)


def generate_deepdrr_drr(
    volume_tensor: torch.Tensor,
    voxel_spacing: float,
    image_size: int,
    device: torch.device,
    geometry: GeometryConfig | None = None,
) -> torch.Tensor | None:
    """
    Generate a physically realistic DRR using DeepDRR's GPU-accelerated simulator.

    Models beam-hardening and Compton scatter (validated against full Monte Carlo),
    making it a physics-based reference that is orders of magnitude faster than
    stochastic photon-tracking codes. Requires CUDA/Linux; returns None otherwise.

    Args:
        volume_tensor (torch.Tensor): CT density volume ``(1, 1, D, H, W)``
            normalised to ``[0, 1]``.
        voxel_spacing (float): Isotropic voxel spacing in mm.
        image_size (int): Square output image resolution in pixels.
        device (torch.device): Target device; internal GPU compute via cupy.
        geometry (GeometryConfig): X-ray geometry configuration; uses
            DEFAULT_GEOMETRY if None.

    Returns:
        torch.Tensor | None: DRR image ``(1, 1, H, W)`` normalised to ``[0, 1]``,
            or None if DeepDRR package/CUDA is unavailable or rendering fails.
    """
    # 1. Dependency check
    try:
        import deepdrr
        from deepdrr import geo as ddgeo
    except ImportError as exc:
        logger.warning(
            "[DeepDRR] Package not found: %s. "
            "Install with: pip install deepdrr[cuda12x]  (Linux + CUDA required)",
            exc,
        )
        return None

    geo = geometry or DEFAULT_GEOMETRY
    assert volume_tensor.dim() == 5, "Expected (1, 1, D, H, W)"
    D = volume_tensor.shape[2]

    try:
        # 2. Build DeepDRR volume from HU
        hu_values = _density_to_hu(volume_tensor)  # (D, H, W)

        half_mm = D * voxel_spacing / 2.0
        spacing = (float(voxel_spacing),) * 3

        # DeepDRR internally uses SimpleITK, which strictly expects 3D numpy arrays 
        # in (Z, Y, X) dimension order. If we pass (X, Y, Z), it swaps the axes and 
        # rotates the volume. We must transpose it here.
        hu_itk = np.ascontiguousarray(np.transpose(hu_values, (2, 1, 0)))
        
        # Center the volume at the world origin
        origin = ddgeo.point(-half_mm, -half_mm, -half_mm)

        logger.debug("[DeepDRR] Segmenting materials (use_thresholding=True) ...")
        ct_volume = deepdrr.Volume.from_hu(
            hu_values=hu_itk,
            origin=origin,
            spacing=spacing,
            anatomical_coordinate_system="LPS",
            use_thresholding=True,
        )

        # 3. Configure imaging geometry
        fov_mm     = D * voxel_spacing          # field-of-view in mm
        pixel_size = fov_mm / image_size        # mm / pixel

        dd_device = deepdrr.SimpleDevice(
            sensor_height               = image_size,
            sensor_width                = image_size,
            pixel_size                  = pixel_size,
            source_to_detector_distance = geo.sdd,
        )

        # View direction and up-vector in world coordinates.
        # We aligned the volume's center with the world origin (0,0,0).
        # So our geo view and up vectors are directly in DeepDRR's world space.
        view_dir_world = ddgeo.vector(
            geo.view_dir_x, geo.view_dir_y, geo.view_dir_z,
        )
        up_vec_world = ddgeo.vector(
            geo.up_vec_x, geo.up_vec_y, geo.up_vec_z,
        )

        # We must add the config isocenter offset to the volume's physical center in world
        cx = float(ct_volume.center_in_world[0])
        cy = float(ct_volume.center_in_world[1])
        cz = float(ct_volume.center_in_world[2])
        isocenter = ddgeo.point(
            cx + geo.isocenter_x, 
            cy + geo.isocenter_y, 
            cz + geo.isocenter_z
        )

        dd_device.set_view(
            point                  = isocenter,
            direction              = view_dir_world,
            up                     = up_vec_world,
            source_to_point_fraction = geo.sad / geo.sdd,
        )

        # 4. Render
        logger.debug("[DeepDRR] Initialising projector ...")
        with deepdrr.Projector(
            ct_volume,
            device  = dd_device,
            neglog  = True,     # return attenuation (-log T): air=dark, bone=bright
        ) as projector:
            image_np = projector()  # (H, W) float32

        # 5. Normalise and return
        lo, hi = float(image_np.min()), float(image_np.max())
        image_np = (image_np - lo) / (hi - lo + 1e-8)

        drr_tensor = (
            torch.from_numpy(image_np)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        
        # Flip horizontally to match Monte Carlo, DVR, and DiffDRR
        drr_tensor = torch.flip(drr_tensor, dims=[-1])
        
        return drr_tensor

    except Exception as exc:
        logger.error("[DeepDRR] Rendering failed: %s", exc, exc_info=True)
        return None
