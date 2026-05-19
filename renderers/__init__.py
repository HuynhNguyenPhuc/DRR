"""
Renderers module.
Exports factory functions and rendering helpers for all supported renderers.

Renderers
---------
* **DVR** (PyTorch3D) — GPU volume renderer, differentiable.
* **DiffDRR / Siddon** — GPU ray-caster, fully differentiable (DiffDRR lib).
* **Plastimatch** — CPU exact ray-tracer; Ground Truth (geometric accuracy).
* **Monte Carlo** — CPU polychromatic simulation with scatter; Ground Truth
  (physical accuracy: beam hardening + Compton scatter).
* **DeepDRR** — GPU physics renderer with DL scatter estimation; validated
  against full Monte Carlo (requires ``deepdrr`` package + CUDA/Linux).
"""

from .dvr import build_dvr_renderer
from .diffdrr import build_diffdrr_renderer
from .plastimatch import generate_plastimatch_drr
from .monte_carlo import generate_mc_drr
from .deepdrr import generate_deepdrr_drr

__all__ = [
    "build_dvr_renderer",
    "build_diffdrr_renderer",
    "generate_plastimatch_drr",
    "generate_mc_drr",
    "generate_deepdrr_drr",
]
