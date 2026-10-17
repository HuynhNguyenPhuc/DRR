# DRR Benchmarking Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DRR Benchmarking Engine is a framework for comparing digitally reconstructed radiograph (DRR)
generation methods. It evaluates rendering speed, GPU memory usage, agreement with a reference
renderer, and 2D/3D registration under a shared imaging geometry.

The benchmark includes five rendering engines with different numerical integration methods and
X-ray image formation models.

<p align="center">
  <img src="assets/results/quality_comparison.png" width="900" alt="The same chest CT rendered by all five engines in the same AP view">
  <br><em>The same CT volume and camera geometry rendered by the five engines (200×200, AP view).</em>
</p>

## Renderers

| Renderer | Method | Differentiable | Role in the benchmark |
|---|---|---|---|
| [Plastimatch](docs/methods/plastimatch.md) | Exact ray–voxel integration (Siddon) | No | Benchmark reference |
| [DiffDRR](docs/methods/diffdrr.md) | Siddon integration, or trilinear sampling | Yes | Differentiable baseline |
| [DVR](docs/methods/dvr.md) | Sampled line integral on PyTorch3D's volume renderer | Yes | Graphics-based baseline |
| [DeepDRR](docs/methods/deepdrr.md) | 0.1 mm ray march through thresholded materials, 90 kVp spectrum | No\* | Physics-based comparison |
| [Monte Carlo](docs/methods/monte_carlo.md) | Deterministic 12-bin polychromatic ray march with a blurred scatter term | No | Spectral comparison |

\*Whether a renderer is differentiable refers to the implementation benchmarked here, not
necessarily to everything its original library can do. "Monte Carlo" is a historical name: the
renderer is deterministic and does not simulate photon transport.

Plastimatch, DiffDRR and DVR target the same single-energy line integral, using different
numerical integration schemes and implementation conventions, so their images are expected to
agree closely. DeepDRR and Monte Carlo model a
tube spectrum, so their pixel values are expected to differ from the single-energy reference even
when the anatomy matches. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) explains why.

## Quick Start

Requirements: Linux with an NVIDIA GPU (DeepDRR requires Linux), Python 3.10+, and
[Plastimatch](https://plastimatch.org) on `PATH`.

```bash
git clone https://github.com/HuynhNguyenPhuc/DRR.git
cd DRR
pip install uv
uv venv --python 3.10 && source .venv/bin/activate

# PyTorch (CUDA 12.8 wheels as an example), then PyTorch3D, which is built from source
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install fvcore iopath
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

uv pip install -r requirements.txt     # also installs the `drr` package
uv pip install deepdrr[cuda12x]        # DeepDRR; Linux + CUDA 12 only

python scripts/run_benchmark.py    # speed, memory, quality, registration, sweeps
```

Plots and a JSON summary are written to `./benchmark_results/<name>/`, where `<name>` is the CT file's
name, or `example_ct` for the default volume. Plastimatch installation, DeepDRR
for CUDA 11, the configuration the results were measured with, and fixes for common problems are
in [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## Usage

### Sanity check

Before trusting any benchmark number, open `notebooks/sanity_check.ipynb` and run all cells. It
renders one CT volume through all five engines and plots them side by side, so you can confirm
that the anatomy lines up with no scale or orientation problems.

### Running benchmarks

`scripts/run_benchmark.py` runs seven benchmarks: speed, GPU memory, image quality (PSNR and SSIM
against the benchmark reference), registration convergence, multi-view consistency,
quality against resolution, and DVR's sample-count ablation.

```bash
# Default: DiffDRR's bundled example CT (or a synthetic phantom if unavailable)
python scripts/run_benchmark.py

# Your own volume: one or more files, or directories searched for .nii/.nii.gz. If no path is
# found, the example CT is used with only a warning, so check the log (see docs/DATA.md §1)
python scripts/run_benchmark.py --ct path/to/your/volume.nii.gz

# Custom resolution and iteration count
python scripts/run_benchmark.py --ct data/sample.nii.gz --size 128 --resolutions 100 200 300 500 --opt-iters 300

# The two sweeps (both default to powers of 2)
python scripts/run_benchmark.py --quality-resolutions 64 128 256 512 --dvr-npts 16 32 64 128 256 512
```

## Sample results

Representative results are summarised below. Full tables, plots, the multi-view, resolution and
sampling sweeps, and the analysis behind them are in [`docs/RESULTS.md`](docs/RESULTS.md).

Image agreement is measured against Plastimatch, which serves as the benchmark reference for the
single-energy line-integral model. The metrics quantify agreement with this reference, not
clinical realism. Registration results are obtained from targets rendered by the same renderer
and do not represent registration performance on real radiographs.
Runtime and memory figures depend on the hardware and configuration; these were measured on the
setup listed in [`docs/RESULTS.md`](docs/RESULTS.md) §1.

| Method | RMSE vs reference (200×200) | SSIM vs reference (200×200) | Speed (256×256) | Registration (150×150, to ZNCC 0.9995) |
|---|---|---|---|---|
| DVR | 0.0077 | 0.9846 | 11.7 ms | 29 iters, 0.56 s, 0.18° left |
| DiffDRR (Siddon) | 0.0000 | 1.0000 | 16.9 ms | 29 iters, 0.44 s, 0.20° left |
| DiffDRR (trilinear) | 0.0077 | 0.9847 | not measured | not measured |
| DeepDRR | 0.0281 | 0.9716 | 94.6 ms | not applicable |
| Monte Carlo | 0.1022 | 0.9464 | not measured | not applicable |

Values are rounded for display; DiffDRR (Siddon)'s unrounded RMSE is below 10⁻⁶. RMSE is computed on images rescaled to [0, 1], so 0.0077 means about 0.8% of the intensity
range. DiffDRR's exact mode reproduces the reference to floating-point precision, and the two
trilinear-sampled renderers to an RMSE of 0.008. DeepDRR and Monte Carlo model a tube spectrum
(and, for Monte Carlo, scatter) that the single-energy reference does not, so their pixel values
differ by design.

## Geometric configuration

All renderers use one shared physical geometry, defined in `src/drr/renderers/config.py`:

- Source-to-detector distance (SDD): 1020.0 mm
- Source-to-axis distance (SAD): 850.0 mm
- Default projection: AP (anterior–posterior)

Changing it propagates to all five renderers. The units, defaults and per-renderer mapping of
every shared parameter are documented in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md). The renderers
share the same imaging geometry, but not every ray path is identical in every view; the details
are in that document.

## Documentation

Start with [`docs/INDEX.md`](docs/INDEX.md), which lists every document and a suggested reading
order. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) explains the physics and the comparison
between the renderers.

## License

This repository is released under the MIT License (see [`LICENSE`](LICENSE)). The underlying
engines (DiffDRR, PyTorch3D, DeepDRR, Plastimatch) have their own licenses, which apply to
their use and redistribution.
