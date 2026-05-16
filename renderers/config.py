"""
Geometry configuration for DRR renderers.

Defines a canonical physical geometry in world coordinates (mm)
that all renderers must map to, ensuring pixel-perfect alignments.
"""

from dataclasses import dataclass

@dataclass
class GeometryConfig:
    sdd: float = 1020.0  # Source-to-Detector Distance in mm
    sad: float = 850.0   # Source-to-Axis (Isocenter) Distance in mm
    
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
