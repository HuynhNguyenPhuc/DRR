"""Data loading utilities for CT volumes."""

import numpy as np
import torch
import torch.nn.functional as F

from drr.utils.logger import get_logger


# --- Logger -- #
logger = get_logger(__name__)


# Shared density encoding: rho = (HU + 1000) / 4000 above the air cutoff, 0 at or below it.
# The cutoff matches Plastimatch's HU-to-attenuation lookup, which returns 0 for HU <= -800,
# so rho stays exactly proportional to the benchmark reference's attenuation.
HU_MIN = -1000.0
HU_MAX = 3000.0
HU_AIR_CUTOFF = -800.0


def hu_to_density(hu: torch.Tensor) -> torch.Tensor:
    """Encode HU as normalised density in ``[0, 1]``."""
    rho = (hu.clamp(max=HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return torch.where(hu > HU_AIR_CUTOFF, rho, torch.zeros_like(rho))


def density_to_hu(rho: np.ndarray) -> np.ndarray:
    """Invert ``hu_to_density``; exact for HU in ``(-800, 3000]``, air maps to -1000."""
    return (rho * (HU_MAX - HU_MIN) + HU_MIN).astype(np.float32)


def _resample_to_cube(grid: torch.Tensor, zooms: tuple, target_size: int, fill: float):
    """
    Resample a (1, 1, X, Y, Z) grid to isotropic voxels of a cube ``target_size`` per side,
    preserving each axis's physical extent and padding the shorter axes with ``fill``.

    Returns:
        tuple: The resampled grid and its isotropic voxel spacing in mm.
    """
    extents = [float(z) * n for z, n in zip(zooms, grid.shape[2:])]
    spacing = max(extents) / target_size
    counts  = [min(target_size, max(1, round(e / spacing))) for e in extents]
    grid = F.interpolate(grid, size=tuple(counts), mode="trilinear", align_corners=False)
    pad = []
    for n in reversed(counts):  # F.pad takes the last axis first
        before = (target_size - n) // 2
        pad += [before, target_size - n - before]
    return F.pad(grid, pad, value=fill), spacing


def load_ct_volume(ct_path: str | None = None, target_size: int = 128):
    """
    Loads a CT density volume, falling back through three sources — a NIfTI
    file, then DiffDRR's example chest CT, then a synthetic phantom — until
    one succeeds.

    Args:
        ct_path: Path to a NIfTI file to load (requires nibabel), or ``None`` to skip straight to the built-in sources.
        target_size: Cube side length, in voxels, to resample the volume to.

    Returns:
        result: Volume tensor clamped to ``[0, 1]``, ``None`` for the subject
            (callers build one with ``make_diffdrr_subject``), and voxel
            spacing in mm.
    """
    subject = None
    volume_tensor = None
    hu_tensor = None
    voxel_spacing = 2.0
    # Real-world side length (mm), kept so voxel_spacing can be recomputed
    # after resampling below instead of going stale.
    # Per-axis voxel size (mm) of the loaded grid, kept so resampling can preserve physical size.
    zooms = None

    # ── Source 1: User-supplied NIfTI ────────────────────────────────────────
    if ct_path is not None:
        try:
            import nibabel as nib

            img   = nib.load(ct_path)
            zooms = tuple(float(z) for z in img.header.get_zooms()[:3])  # (mm/voxel) per axis
            data  = torch.tensor(img.get_fdata(), dtype=torch.float32)

            if data.min() <= -500:
                hu_tensor = data.unsqueeze(0).unsqueeze(0)
            else:
                logger.warning(
                    "[DATA] %s does not look like HU (min %.1f); min-max normalising instead, "
                    "so HU-based renderers will not see true HU.", ct_path, float(data.min()),
                )
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                volume_tensor = data.unsqueeze(0).unsqueeze(0)

            logger.info("[DATA] Loaded CT from: %s  (spacing=%s mm)", ct_path, zooms)

        except Exception as exc:
            logger.warning("[DATA] Could not load %s: %s", ct_path, exc)
            zooms = None

    # ── Source 2: DiffDRR Example CT ─────────────────────────────────────────
    if volume_tensor is None and hu_tensor is None:
        try:
            from diffdrr.data import load_example_ct

            logger.info("[DATA] Loading DiffDRR example chest CT ...")
            example = load_example_ct(bone_attenuation_multiplier=1.0)
            # Take HU rather than DiffDRR's own density, whose min-max scaling would hide the HU scale.
            hu_tensor = example.volume.data.float().unsqueeze(0)
            zooms     = tuple(float(s) for s in example.volume.spacing)

            logger.info(
                "[DATA] Original shape: %s  (spacing=%s mm)", tuple(hu_tensor.shape), zooms,
            )

        except Exception as exc:
            logger.warning("[DATA] DiffDRR load_example_ct failed: %s", exc)

    # ── Source 3: Synthetic Phantom ──────────────────────────────────────────
    if volume_tensor is None and hu_tensor is None:
        logger.info("[DATA] Generating synthetic CT phantom ...")

        c       = torch.linspace(-1, 1, target_size)
        x, y, z = torch.meshgrid(c, c, c, indexing="ij")
        r       = (x ** 2 + y ** 2 + z ** 2).sqrt()

        # Create a simple spherical phantom with two density levels
        vol = (r < 0.40).float() * 0.3 + (r < 0.15).float() * 0.7
        volume_tensor = vol.unsqueeze(0).unsqueeze(0)

    # ── Final Processing: Resample, encode & Clamp ───────────────────────────
    # HU is resampled before encoding so the air cutoff is applied to the final voxels,
    # leaving no interpolated values just above zero for renderers to disagree on.
    is_hu = hu_tensor is not None
    grid  = hu_tensor if is_hu else volume_tensor
    if zooms is not None:
        grid, voxel_spacing = _resample_to_cube(grid, zooms, target_size, fill=HU_MIN if is_hu else 0.0)
        logger.info("[DATA] Resampled to %d³ at %.3f mm isotropic spacing", target_size, voxel_spacing)

    volume_tensor = hu_to_density(grid) if is_hu else grid

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
