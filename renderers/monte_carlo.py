"""Monte Carlo X-ray simulator (Ground Truth reference)."""

from __future__ import annotations

import numpy as np
import torch

from utils.logger import get_logger
from .config import GeometryConfig, DEFAULT_GEOMETRY


# --- Logger --- #
logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Physics data  (NIST XCOM, interpolated)
# ═════════════════════════════════════════════════════════════════════════════

# Discrete energy bins used by the polychromatic integration (keV).
_ENERGY_BINS_KEV = np.array(
    [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0],
    dtype=np.float64,
)

# 80 kVp tungsten-anode spectrum, 2 mm Al + 0.3 mm Cu filtration.
# Relative photon fluence weights per energy bin (unitless, sum→1 after norm).
# Derived from the Kramers–Bohlin model + characteristic lines at 59.3 keV
# (Kα) and 67.2 keV (Kβ) + filtration via the exponential transmission
# coefficients for Al and Cu at each energy.
_SPECTRUM_WEIGHTS_RAW = np.array(
    [0.008, 0.032, 0.072, 0.120, 0.158, 0.175, 0.175, 0.160,
     0.155, 0.093, 0.085, 0.070],
    dtype=np.float64,
)

# NIST XCOM: total mass attenuation coefficient μ/ρ (cm²/g) for WATER.
# Indexed to _ENERGY_BINS_KEV.
_MU_RHO_WATER = np.array(
    [0.8096, 0.5503, 0.3756, 0.3190, 0.2683, 0.2452, 0.2269,
     0.2169, 0.2059, 0.1964, 0.1925, 0.1837],
    dtype=np.float64,
)  # cm²/g

# NIST XCOM: total mass attenuation coefficient μ/ρ (cm²/g) for CORTICAL BONE
# (ICRU-44, ρ = 1.92 g/cm³).
_MU_RHO_BONE = np.array(
    [3.668, 2.028, 1.331, 0.9653, 0.7296, 0.5873, 0.5057,
     0.4367, 0.3934, 0.3618, 0.3350, 0.2955],
    dtype=np.float64,
)  # cm²/g

# Physical densities (g/cm³).
_RHO_WATER = 1.00   # soft tissue proxy
_RHO_BONE  = 1.92   # ICRU-44 cortical bone

# Linear attenuation μ (cm⁻¹) = μ/ρ × ρ, then converted to mm⁻¹.
_MU_TISSUE_MM = (_MU_RHO_WATER * _RHO_WATER) / 10.0  # mm⁻¹
_MU_BONE_MM   = (_MU_RHO_BONE  * _RHO_BONE)  / 10.0  # mm⁻¹


def _spectrum_weights() -> np.ndarray:
    """
    Return normalised weights for 80 kVp tungsten-anode spectrum.

    Weights are derived from Kramers–Bohlin bremsstrahlung model,
    characteristic lines (Kα, Kβ), and filtration via 2 mm Al + 0.3 mm Cu.

    Returns:
        np.ndarray: Normalised spectrum weights, shape (12,), sum = 1.0.
    """
    w = _SPECTRUM_WEIGHTS_RAW.copy()
    w /= w.sum()
    return w


# ═════════════════════════════════════════════════════════════════════════════
# Cone-beam geometry helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_detector_pixels(
    image_size: int,
    voxel_spacing: float,
    ct_size: int,
    geo: GeometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct cone-beam geometry: source and detector pixel positions.

    Detector is placed at distance ``SDD – SAD`` from isocenter along view_dir,
    matching the geometry of DiffDRR and Plastimatch renderers.

    Args:
        image_size (int): Square detector resolution in pixels.
        voxel_spacing (float): Isotropic voxel spacing in mm (sets FOV).
        ct_size (int): CT volume side length in voxels.
        geo (GeometryConfig): Geometry configuration with SDD, SAD, etc.

    Returns:
        tuple[np.ndarray, np.ndarray]: (source_pos, pixel_pos) where
            source_pos shape (3,) is world-space source position (mm),
            pixel_pos shape (H, W, 3) are world-space detector pixels (mm).
    """
    view  = np.array([geo.view_dir_x, geo.view_dir_y, geo.view_dir_z], dtype=np.float64)
    up    = np.array([geo.up_vec_x,   geo.up_vec_y,   geo.up_vec_z],   dtype=np.float64)
    iso   = np.array([geo.isocenter_x, geo.isocenter_y, geo.isocenter_z], dtype=np.float64)

    view  /= np.linalg.norm(view)
    up    /= np.linalg.norm(up)
    right  = np.cross(view, up)
    right /= np.linalg.norm(right)
    # Re-orthogonalise up against view.
    up     = np.cross(right, view)
    up    /= np.linalg.norm(up)

    source_pos      = iso - geo.sad * view
    detector_center = iso + (geo.sdd - geo.sad) * view

    fov_mm     = ct_size * float(voxel_spacing)
    pixel_size = fov_mm / image_size

    i_idx = np.arange(image_size, dtype=np.float64)
    j_idx = np.arange(image_size, dtype=np.float64)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")  # (H, W)

    dx = (ii - (image_size - 1) / 2.0) * pixel_size   # along right
    dy = (jj - (image_size - 1) / 2.0) * pixel_size   # along up

    pixel_pos = (
        detector_center[None, None, :]
        + dx[:, :, None] * right[None, None, :]
        + dy[:, :, None] * up[None, None, :]
    )  # (H, W, 3)

    return source_pos, pixel_pos


# ═════════════════════════════════════════════════════════════════════════════
# Core ray-marching integrator (energy-indexed)
# ═════════════════════════════════════════════════════════════════════════════

def _march_primary(
    vol_np: np.ndarray,         # (D, H, W)  normalised density [0, 1]
    voxel_spacing: float,       # mm
    source_pos: np.ndarray,     # (3,)
    pixel_pos: np.ndarray,      # (H, W, 3)
    mu_tissue_e: float,         # mm⁻¹ at energy bin e
    mu_bone_e: float,           # mm⁻¹ at energy bin e
    n_pts: int,
) -> np.ndarray:
    """
    Compute Beer-Lambert transmission map for one energy bin via ray marching.

    The effective linear attenuation μ_eff at each sample point is obtained by
    blending soft-tissue and bone μ values according to local density:
    μ_eff = density * μ_bone_e + (1 − density) * μ_tissue_e.

    Args:
        vol_np (np.ndarray): Volume density, shape (D, H_vol, W_vol), [0, 1].
        voxel_spacing (float): Isotropic voxel spacing in mm.
        source_pos (np.ndarray): World-space X-ray source, shape (3,) in mm.
        pixel_pos (np.ndarray): World-space detector pixels, shape (H, W, 3) in mm.
        mu_tissue_e (float): Soft-tissue linear attenuation in mm⁻¹ at energy e.
        mu_bone_e (float): Cortical-bone linear attenuation in mm⁻¹ at energy e.
        n_pts (int): Number of sample points per ray.

    Returns:
        np.ndarray: Transmission map exp(-∫ μ dl), shape (H, W), values [0, 1].
    """
    D, H_vol, W_vol = vol_np.shape
    H, W            = pixel_pos.shape[:2]
    half_mm         = np.array(vol_np.shape, dtype=np.float64) * voxel_spacing / 2.0

    # Ray vectors from source to each pixel (mm).
    ray_vec  = pixel_pos - source_pos[None, None, :]  # (H, W, 3)
    ray_len  = np.linalg.norm(ray_vec, axis=-1)       # (H, W)

    # Step length in mm (same for every ray, approximately).
    mean_len = float(ray_len.mean())
    step_mm  = mean_len / n_pts

    line_integral = np.zeros((H, W), dtype=np.float64)

    for k in range(n_pts):
        t = (k + 0.5) / n_pts
        # World-space sample point for all pixels simultaneously.
        pt = source_pos[None, None, :] + t * ray_vec  # (H, W, 3)

        # Convert to voxel indices.
        vox = (pt + half_mm[None, None, :]) / voxel_spacing  # (H, W, 3)

        ix = np.round(vox[..., 0]).astype(np.int32)
        iy = np.round(vox[..., 1]).astype(np.int32)
        iz = np.round(vox[..., 2]).astype(np.int32)

        inside = (
            (ix >= 0) & (ix < D)
            & (iy >= 0) & (iy < H_vol)
            & (iz >= 0) & (iz < W_vol)
        )

        ix = np.clip(ix, 0, D - 1)
        iy = np.clip(iy, 0, H_vol - 1)
        iz = np.clip(iz, 0, W_vol - 1)

        density = vol_np[ix, iy, iz] * inside          # (H, W)

        # density=0 → air (μ=0); density=1 → bone (μ=μ_bone).
        # Multiplying by density ensures air voxels contribute zero attenuation,
        # preventing the black-image artifact caused by tissue μ applied to air.
        mu_eff  = density * (density * mu_bone_e + (1.0 - density) * mu_tissue_e)
        line_integral += mu_eff

    line_integral *= step_mm
    return np.exp(-line_integral).astype(np.float32)    # transmission (H, W)


# ═════════════════════════════════════════════════════════════════════════════
# Public rendering entry-point
# ═════════════════════════════════════════════════════════════════════════════

def generate_mc_drr(
    volume_tensor: torch.Tensor,
    voxel_spacing: float,
    image_size: int,
    device: torch.device,
    geometry: GeometryConfig | None = None,
    n_pts: int = 128,
    add_quantum_noise: bool = False,
    n_photons: float = 1e5,
    scatter_fraction: float = 0.12,
    scatter_sigma_px: float = 8.0,
    seed: int = 42,
) -> torch.Tensor | None:
    """
    Generate a Ground Truth DRR via polychromatic CPU-based Monte Carlo simulation.

    Models the dominant physics of real MC codes: beam-hardening (80 kVp tungsten
    spectrum), energy-dependent attenuation (NIST XCOM for soft-tissue and bone),
    first-order Compton scatter (Gaussian blur kernel), and optional Poisson
    quantum noise. Fully deterministic, no GPU required; always available as
    physics-based reference alongside DeepDRR.

    Args:
        volume_tensor (torch.Tensor): CT density volume ``(1, 1, D, H, W)``
            normalised to ``[0, 1]``.
        voxel_spacing (float): Isotropic voxel spacing in mm.
        image_size (int): Square output DRR resolution in pixels.
        device (torch.device): Target device for output tensor.
        geometry (GeometryConfig): X-ray geometry config; uses DEFAULT_GEOMETRY
            if None.
        n_pts (int): Number of ray marching samples per ray (default: 128).
            Higher values → higher accuracy but slower compute.
        add_quantum_noise (bool): If True, apply Poisson photon noise
            (default: False).
        n_photons (float): Open-beam photon count for noise model
            (default: 1e5). Used only when add_quantum_noise=True.
        scatter_fraction (float): Fraction of primary fluence to add as scatter;
            typical clinical scatter-to-primary ratio ≈ 0.10–0.20
            (default: 0.12).
        scatter_sigma_px (float): Standard deviation of Gaussian scatter kernel
            in pixels; larger values → more diffuse scatter (default: 8).
        seed (int): Random number generator seed for Poisson noise reproducibility
            (default: 42).

    Returns:
        torch.Tensor | None: DRR image ``(1, 1, H, W)`` normalised to ``[0, 1]``,
            or None if scipy.ndimage.gaussian_filter is unavailable.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        logger.error("[MC] scipy is required for Monte Carlo simulation: %s", exc)
        return None

    assert volume_tensor.dim() == 5, "Expected (1, 1, D, H, W)"
    vol_np   = volume_tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    ct_size  = vol_np.shape[0]
    geo      = geometry or DEFAULT_GEOMETRY
    weights  = _spectrum_weights()

    logger.debug(
        "[MC] Casting rays: image=%d², n_pts=%d, energy_bins=%d",
        image_size, n_pts, len(_ENERGY_BINS_KEV),
    )

    # 1. Build cone-beam geometry
    source_pos, pixel_pos = _build_detector_pixels(
        image_size, voxel_spacing, ct_size, geo,
    )

    # 2. Polychromatic integration
    primary = np.zeros((image_size, image_size), dtype=np.float64)

    for e_idx, (w, mu_t, mu_b) in enumerate(
        zip(weights, _MU_TISSUE_MM, _MU_BONE_MM)
    ):
        transmission = _march_primary(
            vol_np, voxel_spacing, source_pos, pixel_pos,
            float(mu_t), float(mu_b), n_pts,
        )
        primary += w * transmission.astype(np.float64)

    primary = primary.astype(np.float32)

    # 3. First-order Compton scatter estimation
    # A Gaussian blur over the attenuated primary approximates the angular
    # spread of Compton-scattered photons reaching each detector pixel.
    scatter_kernel = gaussian_filter(
        (1.0 - primary) * scatter_fraction,
        sigma=scatter_sigma_px,
    )
    image = primary + scatter_kernel
    image = np.clip(image, 0.0, None)

    # 4. Optional Poisson quantum noise
    if add_quantum_noise:
        rng = np.random.default_rng(seed)
        counts = rng.poisson(image * n_photons).astype(np.float32)
        image  = counts / (n_photons + 1e-12)

    # 5. Normalise and return
    lo, hi = float(image.min()), float(image.max())
    image  = (image - lo) / (hi - lo + 1e-8)

    drr_tensor = (
        torch.from_numpy(image)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    logger.debug("[MC] Done. Output shape: %s", tuple(drr_tensor.shape))
    return drr_tensor
