"""Monte Carlo X-ray simulator."""

from __future__ import annotations

import numpy as np
import torch

from drr.utils.logger import get_logger
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

# Relative photon count per energy bin for an 80 kVp X-ray tube (tungsten anode, 2mm Al + 0.3mm Cu filtration). 
# Not measured — modeled from the standard bremsstrahlung + characteristic-line + filtration equations.
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

# Linear attenuation μ (cm⁻¹) = (μ/ρ) · ρ, then converted to mm⁻¹.
_MU_TISSUE_MM = (_MU_RHO_WATER * _RHO_WATER) / 10.0  # mm⁻¹
_MU_BONE_MM   = (_MU_RHO_BONE  * _RHO_BONE)  / 10.0  # mm⁻¹


def _spectrum_weights() -> np.ndarray:
    """
    Return normalised weights for 80 kVp tungsten-anode spectrum.

    Weights are derived from the Kramers–Bohlin bremsstrahlung model,
    characteristic lines (Kα, Kβ), and 2 mm Al + 0.3 mm Cu filtration.

    Returns:
        np.ndarray: One weight per energy bin, summing to 1.
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

    Detector is placed at distance ``SDD – SAD`` from isocenter along
    `geo.view_dir`, matching DiffDRR/Plastimatch's geometry.

    Args:
        image_size (int): Detector resolution (square).
        voxel_spacing (float): Voxel size in mm; sets the field of view with `ct_size`.
        ct_size (int): CT volume side length in voxels.
        geo (GeometryConfig): X-ray setup (SDD, SAD, isocenter, etc).

    Returns:
        tuple[np.ndarray, np.ndarray]: Source position and detector pixel positions, in mm.
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

    # `ii` is row index (y in image, going down). We want row to map to physical -up.
    # `jj` is col index (x in image, going right). We want col to map to physical right.
    dx = (jj - (image_size - 1) / 2.0) * pixel_size   # along right
    dy = - (ii - (image_size - 1) / 2.0) * pixel_size # along up (negative because rows go down)

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

    At each sample point, μ_eff blends soft-tissue and bone μ by local density: 
    ``μ_eff = density * (density * μ_bone_e + (1 − density) * μ_tissue_e)``.
    The outer `density *` factor (beyond the density/(1-density) blend itself)
    zeroes out μ_eff at density=0 (air) even though μ_tissue_e alone is
    nonzero — see the comment at the call site for why.

    Args:
        vol_np (np.ndarray): The CT density volume, values in [0, 1].
        voxel_spacing (float): Voxel size in mm.
        source_pos (np.ndarray): World-space X-ray source position, in mm.
        pixel_pos (np.ndarray): World-space position of every detector pixel, in mm.
        mu_tissue_e (float): Soft-tissue attenuation at this energy bin (mm⁻¹).
        mu_bone_e (float): Bone attenuation at this energy bin (mm⁻¹).
        n_pts (int): Sample points to march along each ray.

    Returns:
        np.ndarray: Transmission image (fraction of beam getting through), in [0, 1].
    """
    D, H_vol, W_vol = vol_np.shape
    H, W            = pixel_pos.shape[:2]
    half_mm         = np.array(vol_np.shape, dtype=np.float64) * voxel_spacing / 2.0

    # Ray vectors from source to each pixel (mm).
    ray_vec  = pixel_pos - source_pos[None, None, :]  # (H, W, 3)
    ray_len  = np.linalg.norm(ray_vec, axis=-1)       # (H, W)

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

        # Track which samples actually fall inside the volume before clipping,
        # so out-of-bounds rays contribute zero attenuation instead of
        # sampling the nearest edge voxel.
        inside = (
            (ix >= 0) & (ix < D)
            & (iy >= 0) & (iy < H_vol)
            & (iz >= 0) & (iz < W_vol)
        )

        # Clip so the (now-masked) out-of-bounds indices are still valid for
        # the fancy-indexing lookup below.
        ix = np.clip(ix, 0, D - 1)
        iy = np.clip(iy, 0, H_vol - 1)
        iz = np.clip(iz, 0, W_vol - 1)

        density = vol_np[ix, iy, iz] * inside          # (H, W)

        # density=0 → air (μ=0); density=1 → bone (μ=μ_bone).
        # Multiplying by density ensures air voxels contribute zero attenuation,
        # preventing the black-image artifact caused by tissue μ applied to air.
        mu_eff  = density * (density * mu_bone_e + (1.0 - density) * mu_tissue_e)
        line_integral += mu_eff

    line_integral *= (ray_len / n_pts)
    return np.exp(-line_integral).astype(np.float32)    # transmission (H, W)


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point: Monte Carlo DRR generator
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
    Generate a physics-informed reference DRR (NOT ground truth) via deterministic polychromatic ray-marching.

    Args:
        volume_tensor (torch.Tensor): The CT density volume, in [0, 1].
        voxel_spacing (float): The voxel size in mm.
        image_size (int): The output resolution in pixels.
        device (torch.device): Where to put the returned tensor.
        geometry (GeometryConfig): X-ray setup; defaults to ``DEFAULT_GEOMETRY``.
        n_pts (int): The number of samples per ray — more is slower but more accurate.
        add_quantum_noise (bool): Whether to add Poisson photon noise.
        n_photons (float): The number of photons per pixel for the Poisson noise model.
        scatter_fraction (float): The fraction of the primary beam that is Compton-scattered and reaches the detector.
        scatter_sigma_px (float): The blur radius of the scatter, in pixels — larger is more diffuse.
        seed (int): The random seed, for reproducible noise.

    Returns:
        torch.Tensor | None: The rendered DRR normalised to [0, 1].
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
    # A Gaussian blur over the attenuated primary approximates the angular spread of Compton-scattered photons reaching each detector pixel.
    scatter_kernel = gaussian_filter(
        (1.0 - primary) * scatter_fraction,
        sigma=scatter_sigma_px,
    )
    image = primary + scatter_kernel

    # Convert transmission → attenuation: -log(T) so that bone is bright and air is dark.
    image = np.clip(image, 1e-6, 1.0)           # avoid log(0) and log(>1)
    image = -np.log(image).astype(np.float32)   # attenuation integral (≥ 0)

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
