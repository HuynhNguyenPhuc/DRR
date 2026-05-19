# DRR Benchmarking Engine 🩻

A unified and high-performance benchmarking suite for **Digitally Reconstructed Radiographs (DRRs)**. 

This repository allows for the precise, standardized evaluation of different DRR generation algorithms for medical imaging research (e.g., CVPR, MICCAI). We ensure strict geometric alignment, photometric consistency, and unbiased testing across multiple state-of-the-art DRR engines.

## 🚀 Supported Renderers

This suite natively wraps and strictly aligns the following rendering engines:

1. **Plastimatch (Ground Truth - Geometric):** CPU-based exact geometric ray-tracing.
2. **Monte Carlo (Ground Truth - Physical):** CPU-based polychromatic simulation including scatter effects.
3. **DiffDRR (Siddon & Trilinear):** Highly optimized, fully differentiable GPU ray-casting.
4. **DVR (PyTorch3D):** Differentiable Volume Rendering via PyTorch3D emission-absorption ray-marching.
5. **DeepDRR:** GPU physics renderer with deep-learning scatter estimation validated against Monte Carlo.

## 📋 Prerequisites

To run this benchmarking suite, you will need:
- **OS:** Windows / Linux (Linux recommended for DeepDRR support)
- **GPU:** NVIDIA GPU with CUDA support for hardware-accelerated rendering.
- **Python:** 3.10+ (using uv for fast dependency management recommended)
- **Plastimatch:** Installed globally on your system.

## 🛠️ Installation & Setup

### 1. Create a Virtual Environment with uv
```bash
# Install uv if you haven't already: pip install uv
uv venv --python 3.10

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install PyTorch & PyTorch3D
Install PyTorch with your respective CUDA version (e.g., CUDA 12.8) using `uv`:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install **PyTorch3D** (required for the DVR engine):
```bash
# Install dependencies
uv pip install fvcore iopath
# Build PyTorch3D from source (requires C++ build tools)
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 3. Install Python Dependencies
Install the remaining packages via `uv pip`:
```bash
uv pip install -r requirements.txt
```

### 4. Install System Dependencies (Plastimatch)
You must have [Plastimatch](https://plastimatch.org/) installed and available in your system's `PATH`.
- **Ubuntu/Debian:** `sudo apt-get install plastimatch`
- **Windows:** Download the installer from the Plastimatch website and add it to your environment variables.

### 5. Install DeepDRR
DeepDRR provides physical accuracy with neural scatter estimation. It is officially supported only on Linux with CUDA.
```bash
# For CUDA 11.x:
uv pip install deepdrr[cuda11x]

# For CUDA 12.x:
uv pip install deepdrr[cuda12x]

# For CUDA 12.8:
uv pip install deepdrr[cuda128]
```

## 🧪 Usage

### 1. Sanity Check
Before running heavy benchmarks, verify that all renderers are geometrically aligned and configured correctly. 
Open and run all cells in `sanity_check.ipynb`.

This notebook will render a single CT volume across all 5 engines and plot them side-by-side to ensure the anatomy (ribs, heart, etc.) is perfectly aligned without scale or polarity issues.

### 2. Running Benchmarks
Use `run_benchmark.py` to evaluate speed, VRAM footprint, image quality (PSNR/SSIM against Ground Truth), and optimization convergence.

```bash
# Run benchmark using the default built-in DiffDRR example CT (or a synthetic phantom)
python run_benchmark.py

# Run benchmark on a specific NIfTI volume
python run_benchmark.py --ct path/to/your/volume.nii.gz

# Customize resolution, optimization iterations, etc.
python run_benchmark.py --ct data/sample.nii.gz --size 128 --resolutions 100 200 300 500 --opt-iters 300
```

The output plots (speed, VRAM, metrics) and JSON summary will be saved to the `./benchmark_results` directory.

## 📐 Geometric Configuration
All renderers are aligned to a standardized physical geometry defined in `renderers/config.py`:
- **Source-to-Detector Distance (SDD):** 1020.0 mm
- **Source-to-Axis Distance (SAD):** 850.0 mm
- **Projection:** AP (Anterior-Posterior) by default.

Changes to this configuration will automatically propagate across all 5 engines to preserve 1:1 mapping.

## 📜 License
This project is part of a research initiative. Please adhere to the licenses of the underlying engines (DiffDRR, PyTorch3D, DeepDRR, Plastimatch) when distributing your work.
