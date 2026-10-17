# Installation and Troubleshooting

*(part of [Documentation Guide](INDEX.md))*

This document gives the full installation procedure for the benchmark, with the platform-specific
notes and known failure modes. For a short path to a first run, see the Quick Start in
[`README.md`](../README.md).

## Contents

1. [Requirements](#1-requirements)
2. [Tested configuration](#2-tested-configuration)
3. [Installation steps](#3-installation-steps)
4. [Troubleshooting](#4-troubleshooting)

---

## 1. Requirements

- **OS:** Linux recommended (required for DeepDRR); Windows works for the other renderers.
- **GPU:** NVIDIA, CUDA-capable.
- **Python:** 3.10 or later; [uv](https://docs.astral.sh/uv/) is recommended for dependency
  management.
- **Plastimatch:** installed system-wide and on `PATH`.

## 2. Tested configuration

All results in [RESULTS.md](RESULTS.md) were measured with the following, not with the newest
versions of each package:

| Component | Version |
|---|---|
| GPU | NVIDIA TITAN RTX (24 GB), driver 580.17, CUDA 12.4 |
| PyTorch | 2.6.0+cu124 |
| PyTorch3D | 0.7.9, built from source |
| DiffDRR | 0.6.1 |
| DeepDRR | 1.2.2 |
| Plastimatch | 1.10.0, installed as a snap |

The PyTorch wheel index used in the example below, `cu128`, is only an example. PyTorch3D is built
against whichever PyTorch is installed, and DeepDRR depends on the CUDA version as well, so pick
one consistent set rather than mixing versions.

## 3. Installation steps

### 3.1 Clone and create a virtual environment

```bash
git clone https://github.com/HuynhNguyenPhuc/DRR.git
cd DRR
pip install uv            # if uv is not installed
uv venv --python 3.10

# Linux / macOS
source .venv/bin/activate
```

On Windows, in Command Prompt:

```bat
.venv\Scripts\activate
```

In PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3.2 PyTorch and PyTorch3D

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

PyTorch3D, needed for DVR, has to be built from source:

```bash
uv pip install fvcore iopath
# --no-build-isolation so the build script can see the PyTorch you just installed
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```

### 3.3 Remaining dependencies

```bash
uv pip install -r requirements.txt
```

This also installs the `drr` package itself in editable mode.

### 3.4 Plastimatch

Plastimatch must be on `PATH`.

- **Ubuntu 22.04 and later:** it is no longer in the apt archives (`apt-get install plastimatch`
  fails with "no installation candidate", even with `universe` enabled). Use
  `sudo snap install plastimatch`.
- **Older Ubuntu / Debian:** `sudo apt-get install plastimatch` still works.
- **Windows:** download the installer from plastimatch.org and add it to `PATH`.

### 3.5 DeepDRR (Linux and CUDA only)

```bash
# CUDA 11.x
uv pip install deepdrr[cuda11x]
# CUDA 12.x
uv pip install deepdrr[cuda12x]
```

### 3.6 Jupyter kernel for the sanity check

```bash
uv pip install ipykernel
python -m ipykernel install --user --name drr_env --display-name "Python (DRR)"
```

## 4. Troubleshooting

**`cannot import name 'cudart' from 'cuda'`.** DeepDRR's `cuda-python` dependency is unbounded, so
a fresh install can pull `cuda-python>=13`, which dropped the `from cuda import cudart` shim that
DeepDRR still imports. Install an older version:

```bash
uv pip install "cuda-python<13"
```

**Plastimatch exits successfully but writes no image.** The snap runs under strict confinement with
its own private `/tmp`, which the host cannot see, while Python's `tempfile.TemporaryDirectory()`
defaults to the host's `/tmp`. This repository avoids it by placing the temporary directory under
`$HOME`, which the snap can access. If you meet a variant of the problem with another install
method, run `snap connections plastimatch` to see which paths the snap can reach. Background in
[DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).

**A renderer is reported as `UNAVAILABLE`.** Each renderer whose dependency is missing is skipped
instead of stopping the benchmark. Check that the corresponding package (PyTorch3D for DVR,
`diffdrr`, `deepdrr`) or executable (`plastimatch`) is installed in the active environment.
