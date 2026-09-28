"""Data loading utilities for CT volumes."""

import numpy as np
import torch
import torch.nn.functional as F

from drr.utils.logger import get_logger


# --- Logger -- #
logger = get_logger(__name__)


def load_ct_volume(ct_path: str | None = None, target_size: int = 128):
    """
    Loads a CT density volume, falling back through three sources — a NIfTI
    file, then DiffDRR's example chest CT, then a synthetic phantom — until
    one succeeds.

    Args:
        ct_path: Path to a NIfTI file to load (requires nibabel), or ``None`` to skip straight to the built-in sources.
        target_size: Cube side length, in voxels, to resample the volume to.

    Returns:
        result: Volume tensor clamped to ``[0, 1]``, a torchio ``Subject``
            (``None`` unless the DiffDRR example CT was used), and voxel
            spacing in mm.
    """
    subject = None
    volume_tensor = None
    voxel_spacing = 2.0
    # Real-world side length (mm), kept so voxel_spacing can be recomputed
    # after resampling below instead of going stale.
    physical_size_mm = None

    # ── Source 1: User-supplied NIfTI ────────────────────────────────────────
    if ct_path is not None:
        try:
            import nibabel as nib

            img   = nib.load(ct_path)
            zooms = img.header.get_zooms()[:3]  # (mm/voxel) per axis
            data  = torch.tensor(img.get_fdata(), dtype=torch.float32)

            # Normalize to [0, 1]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            volume_tensor = data.unsqueeze(0).unsqueeze(0)

            # Downstream code assumes one isotropic spacing — approximate
            # anisotropic scans with their mean per-axis spacing.
            spacing_mean = float(np.mean(zooms))
            if zooms and max(zooms) / min(zooms) > 1.05:
                logger.warning(
                    "[DATA] Anisotropic voxel spacing %s mm — approximating as isotropic %.3f mm.",
                    tuple(round(float(z), 3) for z in zooms), spacing_mean,
                )
            voxel_spacing    = spacing_mean
            physical_size_mm = spacing_mean * float(volume_tensor.shape[2])

            logger.info("[DATA] Loaded CT from: %s  (spacing=%.3fmm)", ct_path, voxel_spacing)

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

            zooms        = tuple(float(s) for s in subject.density.spacing)
            spacing_mean = float(np.mean(zooms))
            if max(zooms) / min(zooms) > 1.05:
                logger.warning(
                    "[DATA] Anisotropic voxel spacing %s mm — approximating as isotropic %.3f mm.",
                    tuple(round(z, 3) for z in zooms), spacing_mean,
                )
            voxel_spacing    = spacing_mean
            physical_size_mm = spacing_mean * float(volume_tensor.shape[2])

            logger.info(
                "[DATA] Original shape: %s  (spacing=%.3fmm)",
                tuple(volume_tensor.shape), voxel_spacing,
            )

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

        # Resampling changes voxel count but not real-world size, so
        # voxel_spacing must scale to match, or every renderer's field of
        # view would use stale spacing.
        if physical_size_mm is not None:
            voxel_spacing = physical_size_mm / target_size
            logger.info("[DATA] Adjusted voxel spacing after resample: %.3fmm", voxel_spacing)

    # Clamp to [0, 1] — densities outside that range aren't physical.
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
    Wraps a raw density tensor in a torchio ``Subject`` DiffDRR can render,
    with a world-space affine that centers the volume at the origin.

    Args:
        volume_tensor: CT density volume to wrap.
        voxel_spacing: Isotropic voxel size, in mm.

    Returns:
        subject: A ready-to-render torchio ``Subject``, or ``None`` if
            torchio/diffdrr aren't installed.
    """
    # torchio/diffdrr are optional — return None instead of raising if missing.
    try:
        import torchio
        from diffdrr.data import canonicalize
    except ImportError:
        return None

    # ── World-space affine centring the volume at the origin ───────────────
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

    dens_4d     = volume_tensor.squeeze(0)
    img_np      = dens_4d.cpu().numpy()
    scalar_img  = torchio.ScalarImage(tensor=img_np, affine=affine)
    # DRR.__init__ reads subject.density.data; the volume is already
    # normalised [0,1], so density == volume.
    density_img = torchio.ScalarImage(tensor=img_np, affine=affine)

    # AP reorient matrix — rotates the C-arm 90° about the x-axis so the
    # detector faces the AP direction.
    reorient = torch.tensor(
        [[1, 0,  0, 0],
         [0, 0, -1, 0],
         [0, 1,  0, 0],
         [0, 0,  0, 1]],
        dtype=torch.float32,
    )

    # torchio.Subject can silently drop non-Image kwargs depending on the
    # version installed — build it with just the two Image entries, then
    # set the rest as plain attributes afterward.
    sub = torchio.Subject(volume=scalar_img, density=density_img)
    sub.mask      = None     # DRR.__init__: `if subject.mask is not None`
    sub.fiducials = None     # canonicalize(): `if subject.fiducials is not None`
    sub.reorient  = reorient # DRR.__init__: passed to Detector
    
    return canonicalize(sub)
