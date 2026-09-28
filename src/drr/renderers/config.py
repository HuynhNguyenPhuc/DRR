"""Geometry configuration for DRR renderers."""

from dataclasses import dataclass

@dataclass
class GeometryConfig:
    """
    Shared physical X-ray geometry.
    Each renderer takes one of these and converts it into its own coordinate system.
    """

    # Source-to-Detector Distance in mm
    sdd: float = 1020.0

    # Source-to-Axis (Isocenter) Distance in mm
    sad: float = 850.0
    
    # Isocenter position (mm)
    isocenter_x: float = 0.0
    isocenter_y: float = 0.0
    isocenter_z: float = 0.0

    # Viewing direction (unit vector from source to detector)
    # Default is AP projection: ray travels along +Y axis
    view_dir_x: float = 0.0
    view_dir_y: float = 1.0
    view_dir_z: float = 0.0

    # Up vector for the detector
    # Default is +Z axis
    up_vec_x: float = 0.0
    up_vec_y: float = 0.0
    up_vec_z: float = 1.0

# Global default configuration
DEFAULT_GEOMETRY = GeometryConfig()
