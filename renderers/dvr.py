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
    from pytorch3d.renderer.implicit.raymarching import (
        _check_density_bounds,
        _check_raymarcher_inputs,
        _shifted_cumprod,
    )
    _PYTORCH3D_AVAILABLE = True

except ImportError:
    _PYTORCH3D_AVAILABLE = False

from utils.logger import get_logger
from .config import GeometryConfig, DEFAULT_GEOMETRY


# --- Logger --- #
logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Normalization utilities
# ═════════════════════════════════════════════════════════════════════════════

def minimized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale tensor to (0, 1] by dividing by its maximum value.
    """
    return (x + eps) / (x.max() + eps)


def normalized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalise tensor to [0, 1].
    """
    return (x - x.min() + eps) / (x.max() - x.min() + eps)


def standardized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardise tensor to zero mean and unit variance.
    """
    return (x - x.mean()) / (x.std() + eps)


def _require_pytorch3d():
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
        Perform reversed absorption-emission raymarching.

        This method computes transmittance starting from the far end of each ray and integrates features toward the camera. 
        Compared to the default PyTorch3D implementation, the accumulation order is reversed to better model X-ray attenuation behavior.

        Args:
            rays_densities:
                Tensor of shape ``(B, R, N, 1)`` containing density or absorption values along each ray, where:

                - ``B`` = batch size
                - ``R`` = number of rays
                - ``N`` = number of sampled points per ray

                Values are expected to lie in the range ``[0, 1]``.

            rays_features:
                Tensor of shape ``(B, R, N, F)`` containing feature vectors associated with each sampled point along the ray, where ``F`` is the feature dimension.

            eps:
                Small numerical stability constant added during cumulative product computation to avoid zero-transmittance instability.

            **kwargs:
                Additional keyword arguments accepted for API compatibility.

        Returns:
            Tensor of shape ``(B, R, F + 1)`` containing:

            - Integrated feature values of shape ``(B, R, F)``
            - Final opacity values of shape ``(B, R, 1)``

            The last channel corresponds to the accumulated opacity for each ray.
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
        _require_pytorch3d()
        super().__init__()

        self.image_width   = image_width
        self.image_height  = image_height
        self.n_pts_per_ray = n_pts_per_ray
        self.min_depth     = min_depth
        self.max_depth     = max_depth
        self.ndc_extent    = ndc_extent

    def _create_raysampler(self):
        raise NotImplementedError

    def _create_raymarcher(self):
        raise NotImplementedError

    def _setup_renderer(self):
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
            volume: ``(B, C, D, H, W)`` density tensor, values in ``[0, 1]``.
            cameras: PyTorch3D cameras object defining the viewpoint(s).
            opacity: Optional ``(B, 1, D, H, W)`` opacity override.
            norm_type: One of ``"minimized"``, ``"normalized"``,
                ``"standardized"``, or ``None``.
            scaling_factor: Multiplier applied to densities.
            is_grayscale: If ``True`` the three RGB channels are averaged
                before returning.
            return_bundle: If ``True`` also return the rendering bundle.
            stratified_sampling: Unused (kept for API compatibility).

        Returns:
            Rendered image ``(B, 1, H, W)`` (grayscale) or ``(B, 3, H, W)``,
            optionally followed by the rendering bundle.
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
    """
    Screen-centric renderer — uses EmissionAbsorptionRaymarcher.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_renderer()

    def _create_raysampler(self):
        return NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False,
        )

    def _create_raymarcher(self):
        return EmissionAbsorptionRaymarcher()


class ObjectCentricXRayVolumeRenderer(BaseXRayVolumeRenderer):
    """
    Object-centric renderer — uses AbsorptionEmissionRaymarcher.

    This reverses the integration order so that absorbing structures
    (e.g., bones) appear bright in the final X-ray image.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_renderer()

    def _create_raysampler(self):
        return NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False,
        )

    def _create_raymarcher(self):
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
        image_size (int): Square output image resolution in pixels.
        n_pts (int): Number of sample points per ray.
        ct_size (int): Isotropic CT volume side length in voxels.
        voxel_spacing (float): Isotropic voxel size in mm.
        device (torch.device): Target compute device.

    Returns:
        tuple:
            - **renderer** (ObjectCentricXRayVolumeRenderer | None): Ready-to-use
              renderer, or ``None`` when the import fails.
            - **cameras** (PerspectiveCameras | None): Matching camera object,
              or ``None`` on failure.
    """
    try:
        from pytorch3d.renderer.cameras import PerspectiveCameras
    except ImportError as exc:
        logger.warning("[DVR] Import failed: %s", exc)
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

    R_np = np.stack([x_cam, y_cam, z_cam], axis=1)
    R = torch.from_numpy(R_np).to(device)

    # 5. Source translation
    # Source S = I - SAD * d. In world NDC, SAD becomes sad_ndc
    I_ndc = np.array([geo.isocenter_x, geo.isocenter_y, geo.isocenter_z], dtype=np.float32) * (2.0 / L)
    S = I_ndc - sad_ndc * d
    T_np = -S @ R_np
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

