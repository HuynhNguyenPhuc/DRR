"""Monte Carlo X-ray simulator."""

from __future__ import annotations

import numpy as np
import torch

from drr.utils.logger import get_logger
from .config import GeometryConfig, DEFAULT_GEOMETRY


# --- Logger --- #
logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Physics data  (NIST X-ray mass attenuation tables, log-log interpolated)
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

# Total mass attenuation coefficients μ/ρ (cm²/g) from the NIST X-ray mass attenuation tables
# (Hubbell & Seltzer, NIST SRD 126): the tabulated values at 20/30/40/50/60/80 keV, log-log
# interpolated in between. Indexed to _ENERGY_BINS_KEV.
_MU_RHO_WATER = np.array(
    [0.8096, 0.5305, 0.3756, 0.3136, 0.2683, 0.2456, 0.2269,
     0.2157, 0.2059, 0.1995, 0.1937, 0.1837],
    dtype=np.float64,
)  # cm²/g

# Cortical bone (ICRU-44, ρ = 1.92 g/cm³).
_MU_RHO_BONE = np.array(
    [4.001, 2.183, 1.331, 0.9181, 0.6655, 0.5247, 0.4242,
     0.363, 0.3148, 0.286, 0.2616, 0.2229],
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

def _sample_density(
    vol_np: np.ndarray,         # (D, H, W)  normalised density [0, 1]
    voxel_spacing: float,       # mm
    source_pos: np.ndarray,     # (3,)
    pixel_pos: np.ndarray,      # (H, W, 3)
    n_pts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample density at ``n_pts`` evenly spaced points along the part of each ray inside the volume.

    Args:
        vol_np (np.ndarray): The CT density volume, values in [0, 1], centred at the world origin.
        voxel_spacing (float): Voxel size in mm.
        source_pos (np.ndarray): World-space X-ray source position, in mm.
        pixel_pos (np.ndarray): World-space position of every detector pixel, in mm.
        n_pts (int): Sample points per ray.

    Returns:
        tuple[np.ndarray, np.ndarray]: Trilinearly interpolated density, shape ``(n_pts, H, W)``,
        and each ray's step length in mm, shape ``(H, W)``.
    """
    from scipy.ndimage import map_coordinates

    # Voxel i's centre sits at (i - (n - 1) / 2) * spacing, the centred grid every renderer uses;
    # the volume's box spans half a voxel beyond the first and last centres.
    half_mm = (np.array(vol_np.shape, dtype=np.float64) - 1) * voxel_spacing / 2.0
    box_lo  = -half_mm - voxel_spacing / 2.0
    box_hi  =  half_mm + voxel_spacing / 2.0
    ray_vec = pixel_pos - source_pos[None, None, :]  # (H, W, 3)

    # Slab test: parameters t in [0, 1] where each ray enters and leaves the volume's box.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_a = (box_lo - source_pos) / ray_vec
        t_b = (box_hi - source_pos) / ray_vec
    t_enter = np.clip(np.nanmax(np.minimum(t_a, t_b), axis=-1), 0.0, 1.0)
    t_exit  = np.clip(np.nanmin(np.maximum(t_a, t_b), axis=-1), 0.0, 1.0)
    t_exit  = np.maximum(t_exit, t_enter)  # rays that miss the volume get zero length

    frac = (np.arange(n_pts, dtype=np.float64) + 0.5) / n_pts
    t    = t_enter[None] + frac[:, None, None] * (t_exit - t_enter)[None]        # (N, H, W)
    pts  = source_pos + t[..., None] * ray_vec[None]                              # (N, H, W, 3)
    vox  = (pts + half_mm) / voxel_spacing
    density = map_coordinates(
        vol_np, vox.reshape(-1, 3).T, order=1, mode="constant", cval=0.0,
    ).reshape(t.shape).astype(np.float32)

    step_mm = (t_exit - t_enter) * np.linalg.norm(ray_vec, axis=-1) / n_pts
    return density, step_mm


def _transmission(density: np.ndarray, step_mm: np.ndarray, mu_tissue_e: float, mu_bone_e: float) -> np.ndarray:
    """
    Beer-Lambert transmission for one energy bin from pre-sampled densities.

    At each sample, μ_eff blends soft-tissue and bone μ by local density:
    ``μ_eff = density * (density * μ_bone_e + (1 − density) * μ_tissue_e)``.
    The outer ``density *`` factor makes air (density 0) attenuate nothing.

    Returns:
        np.ndarray: Transmission image (fraction of beam getting through), in [0, 1].
    """
    mu_eff = density * (density * mu_bone_e + (1.0 - density) * mu_tissue_e)
    line_integral = mu_eff.sum(axis=0, dtype=np.float64) * step_mm
    return np.exp(-line_integral).astype(np.float32)


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
    scatter_sigma_mm: float = 15.0,
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
        scatter_sigma_mm (float): The scatter blur's standard deviation on the detector, in mm, so it is
            the same physical width at every output resolution.
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

    # Density does not depend on energy, so sample it once and reuse it for every bin.
    density, step_mm = _sample_density(vol_np, voxel_spacing, source_pos, pixel_pos, n_pts)
    for w, mu_t, mu_b in zip(weights, _MU_TISSUE_MM, _MU_BONE_MM):
        primary += w * _transmission(density, step_mm, float(mu_t), float(mu_b)).astype(np.float64)

    primary = primary.astype(np.float32)

    # 3. Scatter approximation (a heuristic, not a transport model)
    # A Gaussian blur over the attenuated primary approximates the angular spread of Compton-scattered photons reaching each detector pixel.
    scatter_kernel = gaussian_filter(
        (1.0 - primary) * scatter_fraction,
        sigma=scatter_sigma_mm / (ct_size * float(voxel_spacing) / image_size),
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
