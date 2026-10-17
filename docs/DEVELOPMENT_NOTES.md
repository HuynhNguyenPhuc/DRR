# Development Notes

*(part of [Documentation Guide](INDEX.md))*

This document records the bugs and mistaken assumptions found while building the benchmark. They
are kept apart from [RESULTS.md](RESULTS.md) because they describe the development process, not
the measurements; they are listed here because fixing them moved the numbers reported there
without any physical model changing. Each entry says how the problem showed up, what caused it, and
where it was fixed. The final table maps every entry to the code.

## Contents

1. [Data and coordinate systems](#1-data-and-coordinate-systems)
2. [Rendering and physics](#2-rendering-and-physics)
3. [Runtime and dependencies](#3-runtime-and-dependencies)
4. [Evaluation and registration](#4-evaluation-and-registration)
5. [Mapping to the code](#5-mapping-to-the-code)

---

## 1. Data and coordinate systems

**DeepDRR rendered the wrong field of view.** Its frame did not match the other four. The volume
had been transposed into Plastimatch's axis order, but DeepDRR's volume loader uses a different
order, with one axis reversed. Correcting the transpose, and removing an image transpose-and-mirror
that had been partly compensating, raised its PSNR from 9.72 to 21.09 dB and its SSIM from 0.343 to
0.820.

**DiffDRR was mirrored.** Its images were flipped left to right against the other three. An extra
horizontal flip had been applied on top of a rotation matrix that was already correct — the
failure of this entry. DiffDRR's base camera looks along a different axis, with a different up and
right, from this suite's camera frame, and the rotation matrix built from the look-at basis
([METHODOLOGY.md](METHODOLOGY.md) §2.1) maps one onto the other completely: for the frames and conversion used in this benchmark, the two
differ by a rotation, not a reflection. The extra flip mirrored the anatomy relative to the other four renderers.
Removing the flip, which was present in three places, raised SSIM from about 0.51 to 0.79–0.83.

**DeepDRR's PA view was upside down.** Found only when every renderer was tested on more than the
AP view. The antiparallel-vector bug in DeepDRR's own placement routine
rotated the image by 180° for this view and for the cranial–caudal view. DeepDRR's device-placement
routine decides whether its internal up axis already matches the requested one by testing whether
their cross product is close to zero. That test also passes when the vectors are opposite, which
needs a 180° rotation the routine never applies. A check-and-rotate workaround fixed both (PA: 0.36 → 0.82; cranial–caudal:
0.56 → 0.79) without changing the four views that were already correct.

**The density encoding did not match the benchmark reference, or preserve HU.** The example CT used to
arrive through DiffDRR's own min–max density, which `4000ρ − 1000` does not invert: DeepDRR saw HU
shifted by about −174, and Plastimatch's −800 HU air cutoff fell at about −605 HU of the original
scan, zeroing lung tissue that DVR and DiffDRR still integrated. Fixed by encoding HU with a fixed
rule that includes the same cutoff ([DATA.md](DATA.md) §1.2).

**The volume was twice its real size.** Averaging the three voxel spacings and forcing every axis
to the same voxel count made the 360 × 360 × 332.5 mm scan a 667 mm cube. The linear renderers were
unaffected after normalisation, but Monte Carlo and DeepDRR attenuate in real millimetres: DeepDRR's
median transmission was about 0.1%. Fixed by resampling to isotropic voxels that keep each axis's
extent and padding with air ([DATA.md](DATA.md) §1.3).

**Three renderers placed voxels half a voxel apart.** Projecting a small off-centre cube through
all five renderers showed that DVR and DiffDRR centre voxel $`i`$ at $`(i - (D-1)/2)\,s`$, while
Plastimatch and Monte Carlo placed it at $`(i - D/2)\,s`$ and DeepDRR, whose kernel samples one index
behind its own index-to-world transform, elsewhere again — about a pixel apart on the detector.
Nothing looked misaligned by eye, but the offset is what made DiffDRR's exact mode, which runs the
benchmark reference's own algorithm, score RMSE 0.024 and SSIM 0.907 instead of an exact match. All five
now use the centred grid ([CONVENTIONS.md](CONVENTIONS.md) §4) and agree to within 0.3 pixels.

## 2. Rendering and physics

**DVR's compositing was not what it claimed.** The raymarcher, adapted from an earlier renderer,
computed its transmittance with a reversed shift that made the product 1 everywhere, so it summed
density without weighting — the right quantity, by accident — and without the step length, which
darkened the corners by up to 9% in the old geometry. Rewritten as an explicit line integral with
the step length ([DVR](methods/dvr.md) §3.3). The first version of the rewrite held a step length
per sample and ran out of memory at 1024×1024; it now uses one per ray, which also lowered DVR's
peak memory below its earlier values.

**DeepDRR's logarithm compressed the darkest pixels.** The library's own negative-log option adds
the image's minimum intensity plus 0.01 before taking the logarithm. With the volume at its
inflated size, the median pixel's intensity was only about six times that offset, so the darkest
pixels were strongly compressed. Replaced by $`-\log(I/I_0)`$ with the
open-beam intensity $`I_0`$ ([DeepDRR](methods/deepdrr.md) §3).

**Monte Carlo's bone table and sampling.** Its cortical-bone attenuation table matched NIST at
only one of its twelve energies (at 50 keV it was 0.506 cm²/g against NIST's 0.424). Its ray march
also spread 128 nearest-voxel samples over the whole source-to-detector distance, about 8 mm apart;
once the voxels shrank to 2.8 mm, that produced visible aliasing stripes. Both tables now hold NIST
values, the march samples only the part of each ray inside the volume with trilinear interpolation,
and the scatter blur is specified in millimetres rather than pixels ([Monte Carlo](methods/monte_carlo.md)).

## 3. Runtime and dependencies

**Data types broke registration.** With NumPy 2.0 or later, normalising the view and up vectors
silently promoted 32-bit floats to 64-bit, and the perturbation matrix of the registration benchmark ([RESULTS.md](RESULTS.md) §5) was 64-bit as well.
Together they made the registration's matrix product fail on the GPU, and an overly broad exception
handler hid the error. Fixed by setting 32-bit floats explicitly at both points where NumPy arrays
become tensors.

**Plastimatch silently produced nothing.** The command exited successfully but wrote no image.
Installed as a snap, Plastimatch runs with its own private `/tmp`, so it could not see the
temporary files the benchmark had written to the host's. Fixed by putting the temporary files in
the home directory, which the snap can see.

**DiffDRR's memory grew without bound.** Its exact mode rendered every pixel's ray in one batch,
so memory grew with the pixel count; at 1024×1024 the backward pass needed far more than 24 GB.
Fixed by rendering in 256×256 patches and joining the results, which leaves the image unchanged
and bounds the forward memory ([RESULTS.md](RESULTS.md) §3).

**DeepDRR reported a misleading out-of-memory error.** It failed at 1024×1024 while its own memory
pool held only about 264 MB. DeepDRR allocates through CuPy, while the other GPU renderers use
PyTorch; after DVR and DiffDRR had run, PyTorch was holding several gigabytes it no longer used
but had not released. Fixed by releasing PyTorch's cached memory before DeepDRR renders.

## 4. Evaluation and registration

**Registration declared convergence 4.5° from the answer.** The ZNCC threshold of 0.99 is met
while the pose is still about 4.5° off, and DVR optimised the nine entries of its rotation matrix
with nothing keeping it a rotation. Raised to 0.9995 with the pose error now recorded, and DVR's
rotation reparameterised as an axis-angle offset ([RESULTS.md](RESULTS.md) §5).

Together, the last seven fixes moved the AP-view scores from those of the previous version of this
document (SSIM: DVR 0.936, DiffDRR 0.911, DeepDRR 0.843, Monte Carlo 0.856) to those of [RESULTS.md](RESULTS.md) §4.

## 5. Mapping to the code

Every entry above, in the same order:

| Bug | Where it was fixed |
|---|---|
| DeepDRR field of view | The axis transpose of the HU array in `generate_deepdrr_drr` (`src/drr/renderers/deepdrr.py`) |
| DiffDRR mirrored | The `torch.flip(img, dims=[-1])` calls, removed from `build_diffdrr_renderer` and from `run_quality` and `run_optimization` (`src/drr/benchmarks.py`) |
| DeepDRR upside down | The 180° probe in `generate_deepdrr_drr`, working around `deepdrr.SimpleDevice.set_view()` |
| Density encoding | `hu_to_density`, `density_to_hu` (`src/drr/data.py`), used by `generate_plastimatch_drr` and `generate_deepdrr_drr` |
| Volume size | `_resample_to_cube` (`src/drr/data.py`) |
| Voxel centres | MHA origin in `generate_plastimatch_drr`; `half_mm` in Monte Carlo's `_sample_density`; the `set_view` point in `generate_deepdrr_drr` |
| DVR compositing | `AbsorptionEmissionRaymarcher.forward` (`src/drr/renderers/dvr.py`) |
| DeepDRR logarithm | `neglog=False` and the open-beam computation in `generate_deepdrr_drr` |
| Monte Carlo | `_MU_RHO_WATER`, `_MU_RHO_BONE`, `_sample_density`, `scatter_sigma_mm` (`src/drr/renderers/monte_carlo.py`) |
| Data types | `build_dvr_renderer` (`src/drr/renderers/dvr.py`) and the perturbation matrix in `run_optimization` (`src/drr/benchmarks.py`), both now set `dtype` to 32-bit explicitly |
| Plastimatch silent failure | `tempfile.TemporaryDirectory(dir=os.path.expanduser("~"))` in `generate_plastimatch_drr` |
| DiffDRR memory | `_diffdrr_memory_kwargs` (`src/drr/renderers/diffdrr.py`), passing `patch_size` and `checkpoint_gradients` to `DRR(...)` |
| DeepDRR allocator | `torch.cuda.empty_cache()` at the start of `generate_deepdrr_drr` |
| Registration | `CONVERGENCE`, `_rotation_error_deg` and the axis-angle DVR branch in `run_optimization` (`src/drr/benchmarks.py`) |
