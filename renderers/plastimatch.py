"""Plastimatch ."""

import os
import tempfile
import subprocess

import numpy as np
import torch

from utils.logger import get_logger
from .config import GeometryConfig, DEFAULT_GEOMETRY

logger = get_logger(__name__)

def generate_plastimatch_drr(
    volume_tensor: torch.Tensor,
    voxel_spacing: float,
    image_size: int,
    device: torch.device,
    geometry: GeometryConfig = None,
) -> torch.Tensor | None:
    """
    Generate a Ground Truth DRR using Plastimatch's exact ray-tracing algorithm.
    Saves the volume to a temporary file, calls the `plastimatch drr` CLI,
    and loads the result back as a tensor.

    Args:
        volume_tensor (torch.Tensor): CT density volume ``(1, 1, D, H, W)``.
        voxel_spacing (float): Voxel size in mm.
        image_size (int): Output DRR resolution (square).
        device (torch.device): Device to place the returned tensor on.

    Returns:
        torch.Tensor | None: The generated DRR tensor ``(1, 1, H, W)``,
        or None if Plastimatch/SimpleITK is unavailable or fails.
    """
    # 1. Check dependencies
    try:
        subprocess.run(["plastimatch", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("[Plastimatch] Executable not found in PATH.")
        raise RuntimeError("Plastimatch executable not found in PATH. It is strictly required for Ground Truth.") from exc

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        logger.error("[Plastimatch] SimpleITK is required but not installed.")
        raise RuntimeError("SimpleITK is required but not installed. Run `pip install SimpleITK`.") from exc

    # 2. Extract dimensions
    assert volume_tensor.dim() == 5, "Expected (1, 1, D, H, W)"
    vol_np = volume_tensor.detach().cpu().squeeze().numpy()  # (D, H, W)

    # 3. Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "ct.mha")
        out_path = os.path.join(tmpdir, "drr.mha")

        # 4. Save CT volume
        image = sitk.GetImageFromArray(vol_np)
        image.SetSpacing([voxel_spacing, voxel_spacing, voxel_spacing])
        sitk.WriteImage(image, in_path)

        # 5. Build plastimatch command
        geo = geometry or DEFAULT_GEOMETRY
        fov = vol_np.shape[0] * voxel_spacing
        delx = fov / image_size

        cmd = [
            "plastimatch", "drr",
            "-t", "float",
            "-a", "exact",           # Exact raytracing for Ground Truth
            "-I", in_path,
            "-O", out_path,
            "-r", f"{image_size} {image_size}",
            "-p", f"{fov} {fov}",
            "-c", f"{geo.isocenter_x} {geo.isocenter_y} {geo.isocenter_z}", # Image center
            "-o", f"{geo.isocenter_x} {geo.isocenter_y} {geo.isocenter_z}", # Isocenter
            "--sid", str(geo.sdd),
            "--sad", str(geo.sad),
            "-n", f"{-geo.view_dir_x} {-geo.view_dir_y} {-geo.view_dir_z}", # Detector normal
            "-v", f"{geo.up_vec_x} {geo.up_vec_y} {geo.up_vec_z}",          # Up vector
        ]

        # 6. Run plastimatch
        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as exc:
            logger.error("[Plastimatch] Command failed:\n%s", exc.stderr)
            raise RuntimeError(f"Plastimatch command failed:\n{exc.stderr}") from exc

        # 7. Load DRR image
        if not os.path.exists(out_path):
            logger.error("[Plastimatch] Output file was not generated.")
            raise FileNotFoundError("Plastimatch output file was not generated.")
        
        drr_img = sitk.ReadImage(out_path)
        drr_np = sitk.GetArrayFromImage(drr_img)  # (1, H, W) or (H, W)
        
        if drr_np.ndim == 2:
            drr_np = drr_np[np.newaxis, np.newaxis, ...]
        elif drr_np.ndim == 3:
            drr_np = drr_np[np.newaxis, ...]

        # Normalize to [0, 1] for comparison
        drr_np = (drr_np - np.min(drr_np)) / (np.max(drr_np) - np.min(drr_np) + 1e-8)
        
        drr_tensor = torch.from_numpy(drr_np).float().to(device)
        return drr_tensor
