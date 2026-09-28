"""DeepDRR physics-based renderer."""

from __future__ import annotations

import numpy as np
import torch

from drr.utils.logger import get_logger
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
    For real CT data, original HU values should be used directly instead.

    Args:
        volume_tensor (torch.Tensor): The CT density volume.

    Returns:
        np.ndarray: The volume in Hounsfield Units, as ``float32``.
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

    Models beam-hardening and Compton scatter, making it a physics-based reference 
    that is orders of magnitude faster than stochastic photon-tracking codes.

    Args:
        volume_tensor (torch.Tensor): The CT density volume, in [0, 1].
        voxel_spacing (float): The voxel size in mm.
        image_size (int): The output resolution (square).
        device (torch.device): The target device (internal GPU work runs via cupy).
        geometry (GeometryConfig): The X-ray setup; defaults to ``DEFAULT_GEOMETRY``.

    Returns:
        torch.Tensor | None: Rendered DRR normalised to [0, 1].
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

    # Clear GPU memory before DeepDRR's cupy backend allocates its own arrays.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        # 2. Build DeepDRR volume from HU
        hu_values = _density_to_hu(volume_tensor)  # (D, H, W)

        half_mm = D * voxel_spacing / 2.0
        spacing = (float(voxel_spacing),) * 3

        # DeepDRR expects ITK-style (x, y, z) order, but our input is (z, y, x).
        hu_itk = np.ascontiguousarray(np.flip(np.transpose(hu_values, (0, 2, 1)), axis=1))
        
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
        # Therefore, our geo view and up vectors are directly in DeepDRR's world space.
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

        # Check if DeepDRR's realized up-vector is antiparallel to the requested up-vector.
        requested_up = np.array([geo.up_vec_x, geo.up_vec_y, geo.up_vec_z], dtype=np.float64)
        view_dir_np  = np.array([geo.view_dir_x, geo.view_dir_y, geo.view_dir_z], dtype=np.float64)
        view_dir_np /= np.linalg.norm(view_dir_np)
        requested_up_perp = requested_up - np.dot(requested_up, view_dir_np) * view_dir_np
        requested_up_perp /= np.linalg.norm(requested_up_perp)
        realized_up = np.array(dd_device.device_from_camera3d @ ddgeo.vector(0, -1, 0))
        realized_up /= np.linalg.norm(realized_up)
        needs_180_correction = np.dot(realized_up, requested_up_perp) < 0

        # 4. Render
        logger.debug("[DeepDRR] Initialising projector ...")
        with deepdrr.Projector(
            ct_volume,
            device  = dd_device,
            neglog  = True,
            max_mesh_hits = 8,
        ) as projector:
            image_np = projector()

        # 5. Post-process
        if needs_180_correction:
            logger.debug(
                "[DeepDRR] Correcting for SimpleDevice.set_view's antiparallel-up "
                "bug (view=%s, up=%s) with a 180-degree in-plane rotation.",
                view_dir_np.tolist(), requested_up.tolist(),
            )
            image_np = image_np[::-1, ::-1]

        # Convert to contiguous Numpy array
        image_np = np.ascontiguousarray(image_np)

        # Convert to Pytorch tensor
        drr_tensor = (
            torch.from_numpy(image_np)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        
        return drr_tensor

    except Exception as exc:
        logger.error("[DeepDRR] Rendering failed: %s", exc, exc_info=True)
        return None
