# Note on Renderer Inconsistencies

After reviewing the code in `renderers/diffdrr.py`, `renderers/dvr.py`, and `renderers/plastimatch.py`, I have identified several fundamental inconsistencies in how the camera intrinsic/extrinsic parameters and volume coordinates are set up across the three renderers. 

Since `torch`, `pytorch3d`, and `plastimatch` are required to test and verify the geometric alignment, I have documented the discrepancies and the required fixes here.

## 1. Geometric Misalignment

### DiffDRR
- **Source-to-Detector Distance (SDD):** 1020.0 mm
- **Source-to-Axis Distance (SAD):** 850.0 mm
- **Field of View (FOV):** Sets the physical detector size to exactly the size of the CT volume (`ct_size * voxel_spacing`).
- **Viewing Direction:** The source is at the origin `(0,0,0)`, and the object is translated to `(0, 850, 0)`. Therefore, the projection is along the **+Y axis**.

### Plastimatch
- **SDD & SAD:** Matches DiffDRR (`--sdd 1020.0`, `--sad 850.0`).
- **Viewing Direction:** Uses `-n 0 -1 0` (detector normal pointing to -Y) which means the X-ray travels along the **+Y axis**. This perfectly matches DiffDRR.
- **Bug in `plastimatch.py`:** There is an invalid argument `-N "0 1 0"` in the subprocess command. `plastimatch drr` does not have a `-N` option, which will cause the CLI to crash.

### DVR (PyTorch3D)
- **Focal Length & FOV:** Uses `FoVPerspectiveCameras` with the default 60-degree FOV, which does **not** match the 1020mm SDD and the computed detector size in DiffDRR.
- **Coordinate System Scale:** PyTorch3D normalizes the volume coordinates to `[-1, 1]^3`. 
- **Distance:** The camera translation is hardcoded to `T = [[0.0, 0.0, 6.0]]` (Z=6.0 in PyTorch3D's normalized NDC space), which has no mathematical relation to SAD = 850.0 mm.
- **Viewing Direction:** PyTorch3D cameras look down the **+Z axis** by default. Because DiffDRR looks down the +Y axis, DVR is currently rendering a completely different angle of the CT volume (e.g., Axial vs Coronal projection).
- **Missing Parameters:** `build_dvr_renderer` is missing `ct_size` and `voxel_spacing` in its arguments, meaning it currently lacks the physical context needed to match DiffDRR's geometry.

## 2. Proposed Fixes

To achieve a 1:1 pixel-perfect alignment for a fair benchmark, the following changes must be made:

### A. Fix `plastimatch.py`
Remove the `-N 0 1 0` flag. The command should only contain the detector normal `-n` and the up vector `-v`.

### B. Update `build_dvr_renderer` signature
Update `build_dvr_renderer` in `renderers/dvr.py` (and its caller in `benchmarks.py`) to accept `ct_size` and `voxel_spacing`:

```python
def build_dvr_renderer(image_size: int, n_pts: int, ct_size: int, voxel_spacing: float, device: torch.device):
    # ...
```

### C. Calculate Exact PyTorch3D Camera Parameters for DVR
Replace `FoVPerspectiveCameras` with `PerspectiveCameras` and compute the equivalent normalized parameters:

```python
from pytorch3d.renderer.cameras import PerspectiveCameras

# 1. Physical parameters
L = ct_size * voxel_spacing  # Size of the volume in mm
sdd = 1020.0                 # mm
sad = 850.0                  # mm

# 2. PyTorch3D NDC Focal Length
# PyTorch3D maps the volume to [-1, 1], so the detector size in NDC is 2.0.
# f_ndc = 2 * (focal_length_mm) / (sensor_size_mm)
f_ndc = 2.0 * sdd / L

# 3. PyTorch3D Camera Translation
# Map the physical SAD distance to PyTorch3D's [-1, 1] scale.
T_z = sad * (2.0 / L)

# 4. Alignment Rotation
# DiffDRR looks along +Y. PyTorch3D looks along +Z. 
# We need to rotate the camera so it looks along +Y.
# A rotation of -90 degrees around the X-axis maps +Z to +Y.
angle = -torch.pi / 2.0
R = torch.tensor([[
    [1.0, 0.0, 0.0],
    [0.0, np.cos(angle), -np.sin(angle)],
    [0.0, np.sin(angle), np.cos(angle)],
]], device=device)

T = torch.tensor([[0.0, -T_z, 0.0]], device=device) # Translated along Y axis in world space

cameras = PerspectiveCameras(
    focal_length=((f_ndc, f_ndc),),
    R=R,
    T=T,
    device=device
)
```

By applying these mathematical transformations, PyTorch3D will raymarch through the `[-1, 1]` volume with the exact same frustum and angle as DiffDRR's physical geometry.
