"""Renderer factory function for Plastimatch."""

import glob
import os
import tempfile
import subprocess

import numpy as np
import torch

from drr.data import density_to_hu
from drr.utils.logger import get_logger
from .config import GeometryConfig, DEFAULT_GEOMETRY


# --- Logger --- #
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

    Args:
        volume_tensor (torch.Tensor): The CT density volume.
        voxel_spacing (float): The voxel size in mm.
        image_size (int): The output resolution in pixels.
        device (torch.device): Where to put the returned tensor.
        geometry (GeometryConfig): X-ray setup; defaults to ``DEFAULT_GEOMETRY``.

    Returns:
        torch.Tensor | None: The rendered DRR normalised to [0, 1], or ``None`` if Plastimatch is not installed or fails.
    """
    # 1. Check dependencies (both optional — degrade gracefully instead of raising)
    try:
        subprocess.run(["plastimatch", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("[Plastimatch] Executable not found in PATH.")
        return None

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        logger.error("[Plastimatch] SimpleITK is required but not installed.")
        return None

    # 2. Extract dimensions
    assert volume_tensor.dim() == 5, "Expected (1, 1, D, H, W)"
    vol_np = volume_tensor.detach().cpu().squeeze().numpy()  # (D, H, W)

    # 3. Create temporary directory
    # Plastimatch requires file I/O for both input and output, so we use a temporary directory to avoid cluttering the filesystem.
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
        in_path = os.path.join(tmpdir, "ct.mha")
        # Plastimatch appends "0000.pfm" to this prefix -> drr0000.pfm
        out_prefix = os.path.join(tmpdir, "drr")

        # 4. Save CT volume
        # Plastimatch's DRR lookup table expects Hounsfield Units (HU).
        hu_np = density_to_hu(vol_np)

        # vol_np is (X, Y, Z). SimpleITK GetImageFromArray expects (Z, Y, X) order for 3D numpy arrays.
        hu_np = np.transpose(hu_np, (2, 1, 0))
        image = sitk.GetImageFromArray(hu_np)
        image.SetSpacing([voxel_spacing, voxel_spacing, voxel_spacing])

        # Centre the volume on the world origin: ITK's origin is the centre of the first voxel, so voxel i
        # sits at (i - (n - 1) / 2) * spacing, the same centred grid DVR and DiffDRR use.
        # vol_np axes are (X, Y, Z), so shape[0] is X dimension.
        origin = [-(n - 1) * voxel_spacing / 2.0 for n in vol_np.shape]
        image.SetOrigin(origin)
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
            return None

        # 7. Load DRR image — Plastimatch outputs drr0000.pfm
        generated_files = sorted(glob.glob(out_prefix + "*.pfm"))
        if not generated_files:
            logger.error("[Plastimatch] Output file was not generated.")
            return None

        out_path = generated_files[0]

        # Read PFM (Portable FloatMap) manually — SimpleITK does not support pfm.
        # PFM has a 3-line ASCII header (magic bytes, dimensions, byte-order/scale) followed by raw float32 pixel data.
        try:
            with open(out_path, "rb") as f:
                header = f.readline().decode().strip()   # "PF" (color) or "Pf" (grayscale)
                dims = f.readline().decode().strip()     # "<cols> <rows>"
                scale = float(f.readline().decode().strip())  # sign encodes endianness
                cols, rows = map(int, dims.split())

                endian = '<' if scale < 0 else '>'
                drr_np = np.frombuffer(f.read(), dtype=f"{endian}f4").copy()
                drr_np = drr_np.reshape((rows, cols))

                # Plastimatch's coordinate system (u = vup x n) exactly matches
                # our Monte Carlo setup (right = view x up). No flipping required.
        except (OSError, ValueError) as exc:
            logger.error("[Plastimatch] Failed to parse output .pfm file: %s", exc)
            return None

        drr_np = drr_np[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)

        # Normalize to [0, 1] for comparison
        drr_np = (drr_np - drr_np.min()) / (drr_np.max() - drr_np.min() + 1e-8)

        drr_tensor = torch.from_numpy(drr_np).float().to(device)
        return drr_tensor
