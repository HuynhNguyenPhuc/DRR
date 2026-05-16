"""
Renderers module.
Exports factory functions for creating DVR and DiffDRR renderers.
"""

from .dvr import build_dvr_renderer
from .diffdrr import build_diffdrr_renderer

__all__ = [
    "build_dvr_renderer",
    "build_diffdrr_renderer",
]
