"""
Data loading utilities for CT volumes.

This module provides functions to load CT volumes from NIfTI files, 
use built-in example data from DiffDRR, or generate synthetic phantoms.
"""

import numpy as np
import torch
import torch.nn.functional as F

from utils.logger import get_logger

logger = get_logger(__name__)


def load_ct_volume(ct_path: str | None = None, target_size: int = 128):
    """
    Load or generate a CT density volume.

    Attempts sources in priority order:

    1. NIfTI file at *ct_path* (requires *nibabel*).
    2. DiffDRR built-in example chest CT.
    3. Synthetic spherical phantom (always succeeds).

    The returned tensor is resampled to ``target_size³`` and clamped to
    ``[0, 1]``.

    Args:
        ct_path (str | None): Path to a ``.nii`` / ``.nii.gz`` file.
            Pass ``None`` to skip and fall through to the next source.
        target_size (int): Target isotropic voxel grid side length
            (default: 128).

    Returns:
        tuple:
            - **volume_tensor** (torch.Tensor): Shape ``(1, 1, D, H, W)``.
            - **subject**: torchio ``Subject`` returned by
              ``diffdrr.data.load_example_ct``, or ``None`` when the
              example CT was not used.
            - **voxel_spacing** (float): Isotropic voxel size in mm.
    """
    subject = None
    volume_tensor = None
    voxel_spacing = 2.0

    # ── Source 1: User-supplied NIfTI ────────────────────────────────────────
    if ct_path is not None:
        try:
            import nibabel as nib
            
            img  = nib.load(ct_path)
            data = torch.tensor(img.get_fdata(), dtype=torch.float32)
            
            # Normalize to [0, 1]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            volume_tensor = data.unsqueeze(0).unsqueeze(0)
            
            logger.info("[DATA] Loaded CT from: %s", ct_path)
            
        except Exception as exc:
            logger.warning("[DATA] Could not load %s: %s", ct_path, exc)

    # ── Source 2: DiffDRR Example CT ─────────────────────────────────────────
    if volume_tensor is None:
        try:
            from diffdrr.data import load_example_ct
            
            logger.info("[DATA] Loading DiffDRR example chest CT ...")
            subject = load_example_ct(bone_attenuation_multiplier=1.0)
            dens    = subject.density.data.float()
            volume_tensor = dens.unsqueeze(0)
            
            logger.info("[DATA] Original shape: %s", tuple(volume_tensor.shape))
            
        except Exception as exc:
            logger.warning("[DATA] DiffDRR load_example_ct failed: %s", exc)

    # ── Source 3: Synthetic Phantom ──────────────────────────────────────────
    if volume_tensor is None:
        logger.info("[DATA] Generating synthetic CT phantom ...")
        
        c       = torch.linspace(-1, 1, target_size)
        x, y, z = torch.meshgrid(c, c, c, indexing="ij")
        r       = (x ** 2 + y ** 2 + z ** 2).sqrt()
        
        # Create a simple spherical phantom with two density levels
        vol = (r < 0.40).float() * 0.3 + (r < 0.15).float() * 0.7
        volume_tensor = vol.unsqueeze(0).unsqueeze(0)

    # ── Final Processing: Resample & Clamp ───────────────────────────────────
    if volume_tensor.shape[2] != target_size or volume_tensor.shape[3] != target_size or volume_tensor.shape[4] != target_size:
        logger.info("[DATA] Resampling to %d³ ...", target_size)
        volume_tensor = F.interpolate(
            volume_tensor,
            size=(target_size, target_size, target_size),
            mode="trilinear",
            align_corners=False,
        )
        subject = None  # Force recreation with the new tensor and spacing

    # Clamp the values to [0, 1] range to avoid physically impossible densities
    volume_tensor = volume_tensor.clamp(0.0, 1.0)
    
    logger.info(
        "[DATA] Final volume: %s  range [%.3f, %.3f]",
        tuple(volume_tensor.shape),
        volume_tensor.min().item(),
        volume_tensor.max().item(),
    )
    
    return volume_tensor, subject, voxel_spacing


def make_diffdrr_subject(volume_tensor: torch.Tensor, voxel_spacing: float = 2.0):
    """
    Wrap a raw density tensor in a torchio Subject for DiffDRR.

    Constructs a world-space affine that centres the volume at the origin
    with isotropic spacing *voxel_spacing*.

    Args:
        volume_tensor (torch.Tensor): Shape ``(1, 1, D, H, W)`` or
            ``(1, D, H, W)``.
        voxel_spacing (float): Isotropic voxel size in mm (default: 2.0).

    Returns:
        torchio.Subject | None: Canonicalised subject, or ``None`` when
        *torchio* or *diffdrr* are not installed.
    """
    try:
        import torchio
        from diffdrr.data import canonicalize
    except ImportError:
        return None

    D      = volume_tensor.shape[2]
    sp     = float(voxel_spacing)
    offset = -(D * sp / 2.0)
    affine = np.array(
        [[sp, 0,  0,  offset],
         [0,  sp, 0,  offset],
         [0,  0,  sp, offset],
         [0,  0,  0,  1.0   ]],
        dtype=np.float64,
    )
    dens_4d    = volume_tensor.squeeze(0)
    scalar_img = torchio.ScalarImage(tensor=dens_4d.cpu().numpy(), affine=affine)
    sub        = torchio.Subject(volume=scalar_img)
    # Patch for newer diffdrr where canonicalize expects fiducials to exist
    sub.fiducials = None
    return canonicalize(sub)
