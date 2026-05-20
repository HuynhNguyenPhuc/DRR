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
        # Plastimatch appends "0000.pfm" to this prefix -> drr0000.pfm
        out_prefix = os.path.join(tmpdir, "drr")

        # 4. Save CT volume
        # Plastimatch's DRR lookup table expects Hounsfield Units (HU).
        # Convert normalised density [0, 1] → HU: 0 = -1000 (air), 1 = +3000 (bone).
        hu_np = (vol_np * 4_000.0 - 1_000.0).astype(np.float32)
        # vol_np is (X, Y, Z). SimpleITK GetImageFromArray expects (Z, Y, X) order for 3D numpy arrays.
        hu_np = np.transpose(hu_np, (2, 1, 0))
        image = sitk.GetImageFromArray(hu_np)
        image.SetSpacing([voxel_spacing, voxel_spacing, voxel_spacing])
        # Center the volume at the world origin so the isocenter (0,0,0) passes
        # through the volume midpoint — matching DVR, DiffDRR, and MC renderers.
        # vol_np axes are (X, Y, Z), so shape[0] is X dimension.
        half = float(vol_np.shape[0]) * voxel_spacing / 2.0
        image.SetOrigin([-half, -half, -half])
        sitk.WriteImage(image, in_path)

        # 5. Build plastimatch command
        geo = geometry or DEFAULT_GEOMETRY
        fov = vol_np.shape[0] * voxel_spacing
        cmd = [
            "plastimatch", "drr",
            "-i", "exact",           # Exact raytracing algorithm
            "-t", "pfm",             # 32-bit float Portable FloatMap output
            "-I", in_path,
            "-O", out_prefix,        # Prefix only, no extension
            "-r", f"{image_size} {image_size}",
            "-z", f"{fov:.2f} {fov:.2f}",
            "-o", f"{geo.isocenter_x} {geo.isocenter_y} {geo.isocenter_z}", # Isocenter
            "--sid", str(geo.sdd),
            "--sad", str(geo.sad),
            "-n", f"{-geo.view_dir_x} {-geo.view_dir_y} {-geo.view_dir_z}", # Detector normal
            "--vup", f"{geo.up_vec_x} {geo.up_vec_y} {geo.up_vec_z}",       # Up vector
        ]

        # 6. Run plastimatch
        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as exc:
            logger.error("[Plastimatch] Command failed:\n%s", exc.stderr)
            raise RuntimeError(f"Plastimatch command failed:\n{exc.stderr}") from exc

        # 7. Load DRR image — Plastimatch outputs drr0000.pfm
        import glob
        generated_files = sorted(glob.glob(out_prefix + "*.pfm"))
        if not generated_files:
            logger.error("[Plastimatch] Output file was not generated.")
            raise FileNotFoundError("Plastimatch output .pfm file was not generated.")

        out_path = generated_files[0]

        # Read PFM (Portable FloatMap) manually — SimpleITK does not support pfm
        with open(out_path, "rb") as f:
            header = f.readline().decode().strip()   # "PF" (color) or "Pf" (grayscale)
            dims = f.readline().decode().strip()
            scale = float(f.readline().decode().strip())
            cols, rows = map(int, dims.split())
            drr_np = np.frombuffer(f.read(), dtype=np.float32).copy()
            drr_np = drr_np.reshape((rows, cols))
            if scale > 0:  # big-endian
                drr_np = drr_np.byteswap()
            
            # Plastimatch's horizontal axis (normal x vup) points in -X direction.
            # We must flip left-right to match the standard +X right direction.
            # ITK writes PFM top-to-bottom natively, so no vertical flip is needed.
            drr_np = np.fliplr(drr_np)

        drr_np = drr_np[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)

        # Normalize to [0, 1] for comparison
        drr_np = (drr_np - drr_np.min()) / (drr_np.max() - drr_np.min() + 1e-8)

        drr_tensor = torch.from_numpy(drr_np).float().to(device)
        return drr_tensor
