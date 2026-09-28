"""Deep Volume Rendering."""

import torch
import torch.nn as nn

import numpy as np

try:
    from pytorch3d.structures import Volumes
    from pytorch3d.renderer import (
        VolumeRenderer,
        NDCMultinomialRaysampler,
        EmissionAbsorptionRaymarcher,
    )
    from pytorch3d.renderer.cameras import PerspectiveCameras
    from pytorch3d.renderer.implicit.raymarching import (
        _check_density_bounds,
        _check_raymarcher_inputs,
        _shifted_cumprod,
    )
    _PYTORCH3D_AVAILABLE = True

except ImportError:
    _PYTORCH3D_AVAILABLE = False

from drr.utils.logger import get_logger
from .config import GeometryConfig, DEFAULT_GEOMETRY


# --- Logger --- #
logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Normalization utilities
# ═════════════════════════════════════════════════════════════════════════════

def minimized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale tensor to (0, 1] by dividing by its maximum value."""
    return (x + eps) / (x.max() + eps)


def normalized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalise tensor to [0, 1]."""
    return (x - x.min() + eps) / (x.max() - x.min() + eps)


def standardized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardise tensor to zero mean and unit variance."""
    return (x - x.mean()) / (x.std() + eps)


def _require_pytorch3d():
    """Raise an ImportError if pytorch3d isn't installed."""
    if not _PYTORCH3D_AVAILABLE:
        raise ImportError(
            "pytorch3d is required for the DVR renderer. "
            "Install it from https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Custom Raymarcher
# ═════════════════════════════════════════════════════════════════════════════

class AbsorptionEmissionRaymarcher(EmissionAbsorptionRaymarcher if _PYTORCH3D_AVAILABLE else object):
    """Custom absorption-emission raymarcher for X-Ray volume rendering."""

    def __init__(self, *args, **kwargs):
        # Check for PyTorch3D availability.
        _require_pytorch3d()

        super().__init__(*args, **kwargs)

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Raymarch each ray back-to-front instead of PyTorch3D's default front-to-back.
        It better matches how X-rays actually attenuate.

        Args:
            rays_densities: Density at each sample point along each ray, in ``[0, 1]``.
            rays_features: Feature value at each sample point.
            eps: The tiny constant to avoid dividing by (near-)zero transmittance.

        Returns:
            Tensor: Integrated feature per ray, with final opacity appended as the last channel.
        """
        # Ensure inputs are valid and compatible.
        _check_raymarcher_inputs(
            rays_densities, rays_features, None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )

        # Ensure densities are in [0, 1] for physical plausibility.
        _check_density_bounds(rays_densities)

        rays_densities = rays_densities[..., 0]

        # Reverse direction: compute absorption from the end of the ray backwards.
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities.flip(dims=(-1,)),
            shift=-self.surface_thickness,
        ).flip(dims=(-1,))

        # Compute features and opacities along the ray.
        weights  = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)

        return torch.cat((features, opacities), dim=-1)


# Aliases
ScreenCentricRaymarcher = AbsorptionEmissionRaymarcher
ObjectCentricRaymarcher = EmissionAbsorptionRaymarcher if _PYTORCH3D_AVAILABLE else None


# ═════════════════════════════════════════════════════════════════════════════
# X-Ray Volume Renderers
# ═════════════════════════════════════════════════════════════════════════════

class BaseXRayVolumeRenderer(nn.Module):
    """
    Base class for X-ray volume rendering using PyTorch3D.

    Reference: https://github.com/tmquan/cosmed/blob/main/dvr/renderer.py
    """

    def __init__(
        self,
        image_width: int = 256,
        image_height: int = 256,
        n_pts_per_ray: int = 320,
        min_depth: float = 3.0,
        max_depth: float = 9.0,
        ndc_extent: float = 1.0,
    ):
        """
        Store ray-sampling parameters shared by all subclasses.

        Args:
            image_width: The output image width, in pixels.
            image_height: The output image height, in pixels.
            n_pts_per_ray: The number of samples taken along each ray.
            min_depth: The near clipping distance, in NDC-scaled units.
            max_depth: The far clipping distance, same units as `min_depth`.
            ndc_extent: The half-width of the volume's bounding cube in NDC space.
        """
        # Check for PyTorch3D availability.
        _require_pytorch3d()

        super().__init__()

        self.image_width   = image_width
        self.image_height  = image_height
        self.n_pts_per_ray = n_pts_per_ray
        self.min_depth     = min_depth
        self.max_depth     = max_depth
        self.ndc_extent    = ndc_extent

    def _create_raysampler(self):
        """Return the PyTorch3D raysampler to use. Overridden by subclasses."""
        raise NotImplementedError

    def _create_raymarcher(self):
        """Return the PyTorch3D raymarcher to use. Overridden by subclasses."""
        raise NotImplementedError

    def _setup_renderer(self):
        """Build ``self.renderer`` from the subclass's raysampler/raymarcher."""
        self.renderer = VolumeRenderer(
            raysampler=self._create_raysampler(),
            raymarcher=self._create_raymarcher(),
        )

    def forward(
        self,
        volume: torch.Tensor,
        cameras,
        opacity: torch.Tensor | None = None,
        norm_type: str = "standardized",
        scaling_factor: float = 1.0,
        is_grayscale: bool = True,
        return_bundle: bool = False,
        stratified_sampling: bool = False,
    ) -> torch.Tensor:
        """
        Render X-ray images from the input volume.

        Args:
            volume: The density volume, values in ``[0, 1]``.
            cameras: List of PyTorch3D cameras objects defining the viewpoint(s).
            opacity: The separate opacity volume, if it differs from `volume`.
            norm_type: The output normalisation — ``"minimized"``, ``"normalized"``, ``"standardized"``, or ``None``.
            scaling_factor: The multiplier applied to densities before rendering.
            is_grayscale: Whether to average the RGB channels into one.
            return_bundle: Whether to also return PyTorch3D's rendering bundle.

        Returns:
            The rendered image (grayscale or RGB), optionally followed by the rendering bundle.
        """
        # Expand single-channel volume to 3 feature channels expected by VolumeRenderer
        features  = volume.repeat(1, 3, 1, 1, 1) if volume.shape[1] == 1 else volume

        densities = (
            opacity * scaling_factor
            if opacity is not None
            else torch.ones_like(volume[:, [0]]) * scaling_factor
        )
        
        # PyTorch3D grid_sample maps the last dimension (W) to X, middle (H) to Y, first (D) to Z.
        # Our volume is (X, Y, Z), so we must transpose X and Z to pass to PyTorch3D as (Z, Y, X).
        features = features.permute(0, 1, 4, 3, 2)
        densities = densities.permute(0, 1, 4, 3, 2)

        shape   = max(features.shape[2], features.shape[3])
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=2.0 * float(self.ndc_extent) / shape,
        )

        screen_RGBA, bundle = self.renderer(cameras=cameras, volumes=volumes)
        screen_RGBA = screen_RGBA.permute(0, 3, 1, 2)          # (B, 4, H, W)

        rgb_channels = screen_RGBA[:, :3, :, :]                 # (B, 3, H, W)
        screen_RGB   = (
            rgb_channels.mean(dim=1, keepdim=True)
            if is_grayscale
            else rgb_channels
        )

        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))

        return (screen_RGB, bundle) if return_bundle else screen_RGB


class ScreenCentricXRayVolumeRenderer(BaseXRayVolumeRenderer):
    """Screen-centric renderer — uses EmissionAbsorptionRaymarcher."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_renderer()

    def _create_raysampler(self):
        """Uniform NDC raysampler over the full screen, un-stratified for determinism."""
        return NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False,
        )

    def _create_raymarcher(self):
        """PyTorch3D's stock front-to-back emission-absorption raymarcher."""
        return EmissionAbsorptionRaymarcher()


class ObjectCentricXRayVolumeRenderer(BaseXRayVolumeRenderer):
    """Object-centric renderer — uses AbsorptionEmissionRaymarcher.

    This reverses the integration order so that absorbing structures (e.g., bones) appear bright in the final X-ray image.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_renderer()

    def _create_raysampler(self):
        """Uniform NDC raysampler over the full screen, un-stratified for determinism."""
        return NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False,
        )

    def _create_raymarcher(self):
        """This module's reversed-accumulation raymarcher."""
        return AbsorptionEmissionRaymarcher()


__all__ = [
    "minimized",
    "normalized",
    "standardized",
    "AbsorptionEmissionRaymarcher",
    "ScreenCentricRaymarcher",
    "ObjectCentricRaymarcher",
    "BaseXRayVolumeRenderer",
    "ScreenCentricXRayVolumeRenderer",
    "ObjectCentricXRayVolumeRenderer",
    "build_dvr_renderer",
]

def build_dvr_renderer(image_size: int, n_pts: int, ct_size: int, voxel_spacing: float, device: torch.device, geometry: GeometryConfig = None):
    """
    Instantiate an ``ObjectCentricXRayVolumeRenderer`` with a fixed camera.
    Matches DiffDRR's physical geometry using screen-space PerspectiveCameras.

    Args:
        image_size (int): Output resolution (square).
        n_pts (int): Sample points per ray.
        ct_size (int): CT volume side length in voxels.
        voxel_spacing (float): Voxel size in mm.
        device (torch.device): Target compute device.
        geometry (GeometryConfig): X-ray setup; defaults to ``DEFAULT_GEOMETRY``.

    Returns:
        tuple: Renderer and matching cameras, or ``(None, None)`` if pytorch3d isn't installed.
    """
    # PerspectiveCameras is one of the pytorch3d symbols already guarded at
    # module import time — reuse that flag instead of a separate try/except.
    if not _PYTORCH3D_AVAILABLE:
        logger.warning("[DVR] pytorch3d not available.")
        return None, None

    geo = geometry or DEFAULT_GEOMETRY

    # 1. Canonical physical geometry
    sdd = geo.sdd
    sad = geo.sad
    W = image_size
    H = image_size
    L = ct_size * float(voxel_spacing)  # volume size in mm

    # 2. PyTorch3D NDC scaling
    # The default Volumes created in BaseXRayVolumeRenderer maps the data to [-1, 1],
    # so the physical extent L corresponds to 2.0.
    sdd_ndc = sdd * (2.0 / L)
    sad_ndc = sad * (2.0 / L)

    renderer = ObjectCentricXRayVolumeRenderer(
        image_width   = image_size,
        image_height  = image_size,
        n_pts_per_ray = n_pts,
        min_depth     = max(0.1, sad_ndc - 2.0),
        max_depth     = sad_ndc + 2.0,
        ndc_extent    = 1.0,
    ).to(device)

    # 3. Screen-space focal length in pixels
    sx = L / W
    sy = L / H
    fx = sdd / sx
    fy = sdd / sy
    cx = W / 2.0
    cy = H / 2.0

    # 4. Camera basis mapping
    d = np.array([geo.view_dir_x, geo.view_dir_y, geo.view_dir_z], dtype=np.float32)
    d = d / np.linalg.norm(d)
    
    u = np.array([geo.up_vec_x, geo.up_vec_y, geo.up_vec_z], dtype=np.float32)
    u = u / np.linalg.norm(u)

    # PyTorch3D uses:
    # +X left, +Y down, +Z forward
    z_cam = d
    x_cam = np.cross(u, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    # On NumPy >= 2.0, dividing a float32 array by np.linalg.norm(...) silently
    # upcasts the result to float64, which would make R/T float64 tensors that
    # don't match the float32 tensors used elsewhere. Cast back to float32
    # explicitly here rather than relying on numpy to preserve the dtype.
    R_np = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
    R = torch.from_numpy(R_np).to(device)

    # 5. Source translation
    # Source S = I - SAD * d. In world NDC, SAD becomes sad_ndc
    I_ndc = np.array([geo.isocenter_x, geo.isocenter_y, geo.isocenter_z], dtype=np.float32) * (2.0 / L)
    S = I_ndc - sad_ndc * d
    T_np = (-S @ R_np).astype(np.float32)
    T = torch.from_numpy(T_np).unsqueeze(0).to(device)

    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        R=R.unsqueeze(0),
        T=T,
        in_ndc=False,
        image_size=((H, W),),
        device=device,
    )

    return renderer, cameras

