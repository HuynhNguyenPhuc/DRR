"""Benchmark execution tasks."""

import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from drr.utils.logger import get_logger
from drr.renderers import (
    build_dvr_renderer,
    build_diffdrr_renderer,
    generate_plastimatch_drr,
    generate_mc_drr,
    generate_deepdrr_drr,
)
from drr.renderers.config import DEFAULT_GEOMETRY, GeometryConfig
from drr.renderers.diffdrr import _diffdrr_memory_kwargs
from drr.utils import time_fn, peak_vram, zncc_loss, compute_metrics


# --- Logger --- #
logger = get_logger(__name__)


def run_speed(volume_tensor, subject, resolutions, n_pts, ct_size, voxel_spacing, device, warmup, repeats):
    """
    Benchmarks per-frame render latency for DVR, DiffDRR, and DeepDRR across
    a set of resolutions.

    Args:
        volume_tensor: The CT density volume to render, already loaded into memory.
        subject: The torchio ``Subject`` object DiffDRR needs internally; pass ``None`` for the other renderers.
        resolutions: List of square image sizes (in pixels) to time, one full pass per entry.
        n_pts: Number of samples DVR takes along each ray; ignored by the other renderers.
        ct_size: Side length of the CT volume, in voxels (volumes are assumed cubic).
        voxel_spacing: Physical size of one voxel, in millimeters.
        device: The GPU or CPU device the renderers should run on.
        warmup: Number of untimed renders to run first, to let the GPU/caches settle before measuring.
        repeats: Number of timed renders to average together at each resolution.

    Returns:
        results: Mean/std latency (ms) per renderer per resolution, ``None`` where unavailable.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 1 — Rendering Speed")
    logger.info("─" * 64)

    # One sub-dict per renderer, keyed by resolution.
    results = {"resolutions": resolutions, "dvr": {}, "diffdrr": {}, "deepdrr": {}}

    for res in resolutions:
        logger.info("  Resolution: %d×%d", res, res)
        vol = volume_tensor.to(device)

        # ── DVR ──────────────────────────────────────────────────────────────
        renderer, cameras = build_dvr_renderer(res, n_pts, ct_size, voxel_spacing, device)
        if renderer is not None:
            def _dvr():
                return renderer(vol, cameras, norm_type="normalized")

            try:
                mean_ms, std_ms = time_fn(_dvr, device, warmup, repeats)
                logger.info("    DVR     : %7.1f ± %5.1f ms", mean_ms, std_ms)
                results["dvr"][res] = {"mean_ms": mean_ms, "std_ms": std_ms}
            except Exception as exc:
                # Large resolutions can legitimately run out of VRAM — log
                # and move on instead of crashing the whole suite.
                logger.warning("    DVR     : ERROR — %s", exc)
                results["dvr"][res] = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            logger.info("    DVR     : UNAVAILABLE")
            results["dvr"][res] = None

        # ── DiffDRR ──────────────────────────────────────────────────────────
        drr, rot, xyz = build_diffdrr_renderer(res, ct_size, subject, voxel_spacing, device)
        if drr is not None:
            def _diffdrr():
                return drr(rot, xyz, parameterization="euler_angles", convention="ZXY")

            try:
                mean_ms, std_ms = time_fn(_diffdrr, device, warmup, repeats)
                logger.info("    DiffDRR : %7.1f ± %5.1f ms", mean_ms, std_ms)
                results["diffdrr"][res] = {"mean_ms": mean_ms, "std_ms": std_ms}
            except Exception as exc:
                logger.warning("    DiffDRR : ERROR — %s", exc)
                results["diffdrr"][res] = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            logger.info("    DiffDRR : UNAVAILABLE")
            results["diffdrr"][res] = None

        # ── DeepDRR (forward pass only; not differentiable) ─────────────────
        def _deepdrr():
            return generate_deepdrr_drr(volume_tensor, voxel_spacing, res, device)

        # DeepDRR re-renders from scratch every call (no build/render split
        # like DVR and DiffDRR). Do one untimed call first, so a missing
        # install doesn't get timed as a broken render.
        deepdrr_failed = False
        try:
            img_check = _deepdrr()
        except Exception as exc:
            logger.warning("    DeepDRR : ERROR — %s", exc)
            results["deepdrr"][res] = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            img_check = None
            deepdrr_failed = True

        if img_check is not None:
            try:
                mean_ms, std_ms = time_fn(_deepdrr, device, warmup, repeats)
                logger.info("    DeepDRR : %7.1f ± %5.1f ms", mean_ms, std_ms)
                results["deepdrr"][res] = {"mean_ms": mean_ms, "std_ms": std_ms}
            except Exception as exc:
                logger.warning("    DeepDRR : ERROR — %s", exc)
                results["deepdrr"][res] = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        elif not deepdrr_failed:
            logger.info("    DeepDRR : UNAVAILABLE")
            results["deepdrr"][res] = None

    return results


def run_vram(volume_tensor, subject, resolutions, n_pts, ct_size, voxel_spacing, device):
    """
    Benchmarks peak GPU memory usage for DVR, DiffDRR, and DeepDRR across a
    set of resolutions.

    Args:
        volume_tensor: The CT density volume to render, already loaded into memory.
        subject: The torchio ``Subject`` object DiffDRR needs internally; pass ``None`` for the other renderers.
        resolutions: List of square image sizes (in pixels) to measure, one full pass per entry.
        n_pts: Number of samples DVR takes along each ray; ignored by the other renderers.
        ct_size: Side length of the CT volume, in voxels (volumes are assumed cubic).
        voxel_spacing: Physical size of one voxel, in millimeters.
        device: Must be an actual GPU — memory always reads 0 on CPU.

    Returns:
        results: Peak forward/forward+backward memory (MB) per renderer per
            resolution, ``None`` where unavailable. DeepDRR has no
            backward number (not differentiable) and its forward number is
            only an approximation (it renders through ``cupy``, not PyTorch).
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 2 — VRAM Footprint (peak MB)")
    logger.info("─" * 64)

    results = {"resolutions": resolutions, "dvr": {}, "diffdrr": {}, "deepdrr": {}}

    for res in resolutions:
        logger.info("  Resolution: %d×%d", res, res)

        # Needs requires_grad=True so the backward pass has something to measure.
        vol = volume_tensor.to(device).clone().requires_grad_(True)

        # ── DVR ──────────────────────────────────────────────────────────────
        renderer, cameras = build_dvr_renderer(res, n_pts, ct_size, voxel_spacing, device)
        if renderer is not None:
            def _fwd_dvr():
                return renderer(vol, cameras, norm_type="normalized")

            try:
                fwd_mb, bwd_mb = peak_vram(_fwd_dvr, device)
                logger.info(
                    "    DVR     : fwd %7.1f MB  |  fwd+bwd %7.1f MB",
                    fwd_mb, bwd_mb,
                )
                results["dvr"][res] = {"fwd_mb": fwd_mb, "bwd_mb": bwd_mb}
            except Exception as exc:
                logger.warning("    DVR     : ERROR — %s", exc)
                results["dvr"][res] = None
        else:
            logger.info("    DVR     : UNAVAILABLE")
            results["dvr"][res] = None

        # ── DiffDRR ──────────────────────────────────────────────────────────
        drr, rot, xyz = build_diffdrr_renderer(res, ct_size, subject, voxel_spacing, device)
        if drr is not None:
            # rot/xyz need requires_grad=True too, for the same reason.
            rot = rot.clone().requires_grad_(True)
            xyz = xyz.clone().requires_grad_(True)

            def _fwd_diffdrr():
                return drr(rot, xyz, parameterization="euler_angles", convention="ZXY")

            try:
                fwd_mb, bwd_mb = peak_vram(_fwd_diffdrr, device)
                logger.info(
                    "    DiffDRR : fwd %7.1f MB  |  fwd+bwd %7.1f MB",
                    fwd_mb, bwd_mb,
                )
                results["diffdrr"][res] = {"fwd_mb": fwd_mb, "bwd_mb": bwd_mb}
            except Exception as exc:
                logger.warning("    DiffDRR : ERROR — %s", exc)
                results["diffdrr"][res] = None
        else:
            logger.info("    DiffDRR : UNAVAILABLE")
            results["diffdrr"][res] = None

        # ── DeepDRR (renders via cupy, so fwd_mb is only a PyTorch-visible
        # approximation, not its true peak usage) ───────────────────────────
        def _fwd_deepdrr():
            return generate_deepdrr_drr(volume_tensor, voxel_spacing, res, device)

        # Same pre-flight check as run_speed: an untimed call first, to
        # tell "unavailable" apart from a genuine failure.
        img_check = _fwd_deepdrr()
        if img_check is not None:
            try:
                fwd_mb, bwd_mb = peak_vram(_fwd_deepdrr, device)
                logger.info(
                    "    DeepDRR : fwd %7.1f MB  |  (no backward — not differentiable)",
                    fwd_mb,
                )
                results["deepdrr"][res] = {"fwd_mb": fwd_mb, "bwd_mb": None}
            except Exception as exc:
                logger.warning("    DeepDRR : ERROR — %s", exc)
                results["deepdrr"][res] = None
        else:
            logger.info("    DeepDRR : UNAVAILABLE (requires deepdrr + CUDA/Linux)")
            results["deepdrr"][res] = None

    return results


def run_quality(volume_tensor, subject, n_pts, ct_size, voxel_spacing, device, out_dir):
    """
    Renders a fixed scene through all five engines and scores RMSE/PSNR/SSIM
    against Plastimatch, the sole ground-truth renderer in this repo.

    Args:
        volume_tensor: The CT density volume to render, already loaded into memory.
        subject: The torchio ``Subject`` object DiffDRR needs internally; pass ``None`` for the other renderers.
        n_pts: Number of samples DVR takes along each ray; ignored by the other renderers.
        ct_size: Side length of the CT volume, in voxels (volumes are assumed cubic).
        voxel_spacing: Physical size of one voxel, in millimeters.
        device: The GPU or CPU device the renderers should run on.
        out_dir: Folder to save the ``quality_comparison.png`` side-by-side figure into.

    Returns:
        results: RMSE/PSNR/SSIM per renderer, scored against Plastimatch.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 3 — Image Quality (200×200)")
    logger.info("─" * 64)

    RES     = 200
    results = {}
    images  = {}
    vol     = volume_tensor.to(device)

    # ── Ground Truth reference ──────────────────────────────────────────────

    # Plastimatch GT (exact geometric ray-tracing on CPU) — sole Ground Truth.
    img_plastimatch = generate_plastimatch_drr(volume_tensor, voxel_spacing, RES, device)
    if img_plastimatch is not None:
        images["plastimatch_gt"] = img_plastimatch
        logger.info("    Plastimatch (GT)      rendered: shape %s", tuple(img_plastimatch.shape))
    else:
        logger.warning("    Plastimatch (GT)      : FAILED")

    # Monte Carlo — not ground truth, so it's kept out of the "_gt" keys
    # below and scored *against* Plastimatch like everyone else.
    img_mc = generate_mc_drr(volume_tensor, voxel_spacing, RES, device)
    if img_mc is not None:
        images["monte_carlo"] = img_mc
        logger.info("    Monte Carlo           rendered: shape %s", tuple(img_mc.shape))
    else:
        logger.warning("    Monte Carlo           : FAILED")

    # ── Fast / differentiable renderers ─────────────────────────────────────

    renderer, cameras = build_dvr_renderer(RES, n_pts, ct_size, voxel_spacing, device)
    if renderer is not None:
        with torch.no_grad():
            img_dvr = renderer(vol, cameras, norm_type="normalized")
        images["dvr"] = img_dvr
        logger.info("    DVR               rendered: shape %s", tuple(img_dvr.shape))
    else:
        logger.info("    DVR               : UNAVAILABLE — skipping quality check for DVR.")

    drr_siddon, rot, xyz = build_diffdrr_renderer(RES, ct_size, subject, voxel_spacing, device)
    if drr_siddon is not None:
        with torch.no_grad():
            img_siddon = drr_siddon(rot, xyz, parameterization="euler_angles", convention="ZXY")
        images["diffdrr_siddon"] = img_siddon
        logger.info("    DiffDRR (Siddon)  rendered: shape %s", tuple(img_siddon.shape))

    try:
        # diffdrr is optional — a missing install just skips this branch.
        from diffdrr.drr import DRR as DiffDRR_module

        delx = ct_size * voxel_spacing / RES
        drr_tri = DiffDRR_module(
            subject,
            sdd      = DEFAULT_GEOMETRY.sdd,
            height   = RES,
            delx     = delx,
            renderer = "trilinear",
            **_diffdrr_memory_kwargs(RES),
        ).to(device)

        with torch.no_grad():
            # No horizontal flip needed — orientation already matches the
            # other renderers.
            img_trilinear = drr_tri(rot, xyz, parameterization="euler_angles", convention="ZXY")
        images["diffdrr_trilinear"] = img_trilinear
        logger.info("    DiffDRR (Trilinear) rendered: shape %s", tuple(img_trilinear.shape))
    except Exception as exc:
        logger.debug("    DiffDRR trilinear: skipped (%s)", exc)

    # DeepDRR (polychromatic, segmented materials; no scatter in the installed version)
    img_deepdrr = generate_deepdrr_drr(volume_tensor, voxel_spacing, RES, device)
    if img_deepdrr is not None:
        images["deepdrr"] = img_deepdrr
        logger.info("    DeepDRR           rendered: shape %s", tuple(img_deepdrr.shape))
    else:
        logger.info("    DeepDRR           : UNAVAILABLE (requires deepdrr + CUDA/Linux)")

    # ── Metrics vs Ground Truth ──────────────────────────────────────────────
    # Plastimatch is the sole GT; Monte Carlo is scored against it too,
    # under "monte_carlo" rather than a "_gt" key.
    GT_REFS = {
        k: (v, k.replace("_", " ").replace("gt", "GT").title())
        for k, v in images.items()
        if k.endswith("_gt")
    }

    logger.info("")
    for name, img in images.items():
        if name.endswith("_gt"):
            continue   # skip GT-vs-GT comparisons

        for ref_key, (ref_img, ref_label) in GT_REFS.items():
            rmse, psnr, ssim = compute_metrics(img, ref_img)
            label = f"{name} vs {ref_label}"
            logger.info(
                "    %-40s  RMSE=%.4e  PSNR=%6.2f dB  SSIM=%.4f",
                label, rmse, psnr, ssim,
            )
            entry = results.setdefault(name, {})
            entry[ref_key] = {"rmse": rmse, "psnr": psnr, "ssim": ssim, "reference": ref_label}

    # ── Save side-by-side comparison figure ─────────────────────────────────
    if images:
        n    = len(images)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)
        if n == 1:
            axes = [axes]

        for ax, (name, img) in zip(axes, images.items()):
            im_np = img.detach().cpu().squeeze().numpy()
            im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min() + 1e-8)
            ax.imshow(im_np, cmap="gray")
            # Mark ground truth in the title
            title = name.replace("_gt", " [GT]").replace("_", "\n")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        fig.suptitle("Image Quality Comparison (200×200)", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_comparison.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("    Saved → %s", out_dir / "quality_comparison.png")

    return results


def _rotation_error_deg(R: torch.Tensor, R_ref: torch.Tensor) -> float:
    """Angle of the rotation taking ``R_ref`` to ``R``, in degrees (float64, no clamping)."""
    M = (R_ref.transpose(-1, -2) @ R).reshape(3, 3).double()
    sin = torch.stack([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]).norm() / 2
    cos = (M.trace() - 1) / 2
    return float(torch.rad2deg(torch.atan2(sin, cos)))


def run_optimization(volume_tensor, subject, n_pts, ct_size, voxel_spacing, opt_iters, device, out_dir):
    """
    Perturbs the camera pose 15° in pitch (about the world X axis) and runs gradient
    descent (Adam + ZNCC loss) with DVR and DiffDRR back toward the target pose, each
    against a target rendered by the same renderer. Stops at ZNCC >= 0.9995 and records
    the remaining rotation and source-position error.

    Args:
        volume_tensor: The CT density volume to render, already loaded into memory.
        subject: The torchio ``Subject`` object DiffDRR needs internally.
        n_pts: Number of samples DVR takes along each ray.
        ct_size: Side length of the CT volume, in voxels (volumes are assumed cubic).
        voxel_spacing: Physical size of one voxel, in millimeters.
        opt_iters: Maximum optimisation steps to run before giving up on convergence.
        device: The GPU or CPU device the renderers should run on.
        out_dir: Folder to save the ``registration_convergence.png`` loss-curve plot into.

    Returns:
        results: For ``"dvr"``/``"diffdrr"`` — converged iteration, elapsed time, final
            ZNCC, final rotation error (deg) and source-position error (mm), and the
            loss curve; ``None`` where unavailable.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 4 — 2D/3D Registration (Gradient Descent)")
    logger.info("─" * 64)

    OPT_RES     = 150     # lower resolution than the quality checks, for faster iterations
    # ZNCC 0.99 is already reached about 5° from the target; 0.9995 corresponds to under ~0.3°.
    CONVERGENCE = 0.9995
    LR          = 1e-2
    PERTURB_DEG = 15.0
    opt_results = {}
    vol         = volume_tensor.to(device)
    mm_per_ndc  = ct_size * voxel_spacing / 2.0

    perturb_rad = float(PERTURB_DEG * np.pi / 180.0)
    R_perturb = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, np.cos(perturb_rad), -np.sin(perturb_rad)],
        [0.0, np.sin(perturb_rad), np.cos(perturb_rad)],
    ]], dtype=torch.float32, device=device)  # pinned: np.cos/np.sin would make this float64

    # ── DVR registration ─────────────────────────────────────────────────────
    try:
        from pytorch3d.transforms import axis_angle_to_matrix

        logger.info("")
        logger.info("  [DVR] Setting up registration ...")

        # Step 1: render the target from the default pose.
        renderer, cam_gt = build_dvr_renderer(OPT_RES, n_pts, ct_size, voxel_spacing, device)
        if renderer is None:
            raise ImportError("pytorch3d not available")
        with torch.no_grad():
            target_dvr = renderer(vol, cam_gt, norm_type="normalized")

        # Step 2: start from the target pose rotated 15° about X.
        R_gt, T_gt = cam_gt.R.clone(), cam_gt.T.clone()
        R_init = R_perturb @ R_gt

        # Step 3: optimise an axis-angle offset on R_init, so the rotation stays in SO(3).
        omega = torch.zeros(1, 3, device=device, requires_grad=True)
        T_opt = T_gt.clone().requires_grad_(True)
        optimizer      = torch.optim.Adam([omega, T_opt], lr=LR)
        losses_dvr     = []
        converged_iter = opt_iters
        t0             = time.perf_counter()

        # Step 4: gradient-descend back toward the target.
        for it in range(opt_iters):
            optimizer.zero_grad()

            cam_cur = cam_gt.clone()
            cam_cur.R = axis_angle_to_matrix(omega) @ R_init
            cam_cur.T = T_opt

            img_cur = renderer(vol, cam_cur, norm_type="normalized")
            loss    = zncc_loss(img_cur, target_dvr.detach())
            loss.backward()
            optimizer.step()

            zncc_val = -loss.item()
            losses_dvr.append(loss.item())

            if it % 50 == 0 or it < 5:
                logger.debug("    iter %4d  ZNCC=%.4f", it, zncc_val)

            if zncc_val >= CONVERGENCE:
                converged_iter = it + 1
                logger.info("    ✓ DVR converged at iteration %d", converged_iter)
                break

        # Step 5: record results, including the pose error the image metric cannot see.
        elapsed = time.perf_counter() - t0
        with torch.no_grad():
            R_fin = axis_angle_to_matrix(omega) @ R_init
            # PyTorch3D cameras map world to view as X R + T, so the camera centre is -T R^T.
            centre     = -(T_opt @ R_fin.transpose(1, 2))
            centre_gt  = -(T_gt @ R_gt.transpose(1, 2))
            rot_err    = _rotation_error_deg(R_fin, R_gt)
            centre_err = float((centre - centre_gt).norm()) * mm_per_ndc
        opt_results["dvr"] = {
            "converged_iter": converged_iter,
            "elapsed_s": round(elapsed, 2),
            "final_zncc": -losses_dvr[-1],
            "rotation_error_deg": rot_err,
            "source_error_mm": centre_err,
            "loss_curve": losses_dvr,
        }
        logger.info(
            "    DVR  total time: %.1f s  | final ZNCC: %.4f | rotation error %.2f° | source error %.1f mm",
            elapsed, -losses_dvr[-1], rot_err, centre_err,
        )

    except ImportError as exc:
        logger.info("  [DVR] Optimisation skipped: %s", exc)
        opt_results["dvr"] = None
    except Exception as exc:
        # Don't let a DVR-specific failure take down the DiffDRR branch below.
        logger.warning("  [DVR] Optimisation FAILED: %s", exc)
        opt_results["dvr"] = None

    # ── DiffDRR registration ─────────────────────────────────────────────────
    try:
        # diffdrr is optional — a missing install just skips this branch.
        from diffdrr.drr import DRR as DiffDRR_module
        from diffdrr.pose import convert

        logger.info("")
        logger.info("  [DiffDRR] Setting up registration ...")
        delx = ct_size * voxel_spacing / OPT_RES

        # Step 1: render the target from DiffDRR's identity-rotation pose.
        drr     = DiffDRR_module(subject, sdd=DEFAULT_GEOMETRY.sdd, height=OPT_RES, delx=delx).to(device)
        rot_gt  = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        xyz_gt  = torch.tensor([[
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_x,
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_y,
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_z,
        ]], device=device)
        with torch.no_grad():
            target_ddrr = drr(rot_gt, xyz_gt, parameterization="euler_angles", convention="ZXY")
        pose_gt = convert(rot_gt, xyz_gt, parameterization="euler_angles", convention="ZXY").matrix[0]

        # Step 2: perturb the rotation the same way the DVR branch above does.
        # With convention="ZXY", index 1 is the X-axis rotation.
        rot_opt = torch.tensor([[0.0, perturb_rad, 0.0]], device=device, requires_grad=True)
        xyz_opt = xyz_gt.clone().requires_grad_(True)

        # Step 3: set up the Adam optimiser over the perturbed rotation/translation.
        optimizer      = torch.optim.Adam([rot_opt, xyz_opt], lr=LR)
        losses_ddrr    = []
        converged_iter = opt_iters
        t0             = time.perf_counter()

        # Step 4: gradient-descend back toward the target.
        for it in range(opt_iters):
            optimizer.zero_grad()
            img_cur  = drr(rot_opt, xyz_opt, parameterization="euler_angles", convention="ZXY")
            loss     = zncc_loss(img_cur, target_ddrr.detach())
            loss.backward()
            optimizer.step()

            zncc_val = -loss.item()
            losses_ddrr.append(loss.item())

            if it % 50 == 0 or it < 5:
                logger.debug("    iter %4d  ZNCC=%.4f", it, zncc_val)

            if zncc_val >= CONVERGENCE:
                converged_iter = it + 1
                logger.info("    ✓ DiffDRR converged at iteration %d", converged_iter)
                break

        # Step 5: record results.
        elapsed = time.perf_counter() - t0
        with torch.no_grad():
            pose = convert(rot_opt, xyz_opt, parameterization="euler_angles", convention="ZXY").matrix[0]
            rot_err    = _rotation_error_deg(pose[:3, :3], pose_gt[:3, :3])
            centre_err = float((pose[:3, 3] - pose_gt[:3, 3]).norm())
        opt_results["diffdrr"] = {
            "converged_iter": converged_iter,
            "elapsed_s": round(elapsed, 2),
            "final_zncc": -losses_ddrr[-1],
            "rotation_error_deg": rot_err,
            "source_error_mm": centre_err,
            "loss_curve": losses_ddrr,
        }
        logger.info(
            "    DiffDRR total time: %.1f s  | final ZNCC: %.4f | rotation error %.2f° | source error %.1f mm",
            elapsed, -losses_ddrr[-1], rot_err, centre_err,
        )

    except ImportError as exc:
        logger.info("  [DiffDRR] Optimisation skipped: %s", exc)
        opt_results["diffdrr"] = None
    except Exception as exc:
        logger.warning("  [DiffDRR] Optimisation FAILED: %s", exc)
        opt_results["diffdrr"] = None

    # ── Convergence plot ─────────────────────────────────────────────────────
    have_dvr  = opt_results.get("dvr")     is not None
    have_ddrr = opt_results.get("diffdrr") is not None

    if have_dvr or have_ddrr:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

        if have_dvr:
            ax.plot(opt_results["dvr"]["loss_curve"],
                    label="DVR (PyTorch3D)", linewidth=1.5)
        if have_ddrr:
            ax.plot(opt_results["diffdrr"]["loss_curve"],
                    label="DiffDRR (Siddon)", linewidth=1.5)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Neg-ZNCC Loss")
        ax.set_title("Registration Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "registration_convergence.png")
        plt.close(fig)
        logger.info("    Saved → %s", out_dir / "registration_convergence.png")

    return opt_results


def run_multiview(volume_tensor, subject, n_pts, ct_size, voxel_spacing, device, out_dir):
    """
    Re-renders every method across six view/geometry configurations — AP,
    PA, two laterals, an oblique, and a cranial-caudal edge case — and
    scores each against Plastimatch per view.

    Args:
        volume_tensor: The CT density volume to render, already loaded into memory.
        subject: The torchio ``Subject`` object DiffDRR needs internally.
        n_pts: Number of samples DVR takes along each ray.
        ct_size: Side length of the CT volume, in voxels (volumes are assumed cubic).
        voxel_spacing: Physical size of one voxel, in millimeters.
        device: The GPU or CPU device the renderers should run on.
        out_dir: Folder to save the ``multiview_comparison.png`` grid figure into.

    Returns:
        results: RMSE/PSNR/SSIM per renderer per view, ``None`` for a view
            where Plastimatch itself failed to render.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 5 — Multi-View Compatibility")
    logger.info("─" * 64)

    RES    = 200
    vol    = volume_tensor.to(device)
    r_diag = 1.0 / np.sqrt(2.0)   # 45°-diagonal component for the oblique view below

    # Six view/geometry configurations to sweep.
    VIEWS = {
        "AP": GeometryConfig(  # default: front-to-back
            view_dir_x=0.0, view_dir_y=1.0, view_dir_z=0.0,
            up_vec_x=0.0, up_vec_y=0.0, up_vec_z=1.0,
        ),
        "PA": GeometryConfig(  # back-to-front — the AP view direction reversed
            view_dir_x=0.0, view_dir_y=-1.0, view_dir_z=0.0,
            up_vec_x=0.0, up_vec_y=0.0, up_vec_z=1.0,
        ),
        "Lateral +X": GeometryConfig(  # side-on, looking along +X
            view_dir_x=1.0, view_dir_y=0.0, view_dir_z=0.0,
            up_vec_x=0.0, up_vec_y=0.0, up_vec_z=1.0,
        ),
        "Lateral -X": GeometryConfig(  # side-on, opposite laterality
            view_dir_x=-1.0, view_dir_y=0.0, view_dir_z=0.0,
            up_vec_x=0.0, up_vec_y=0.0, up_vec_z=1.0,
        ),
        "Oblique (+offset)": GeometryConfig(  # 45° between AP and lateral, plus an isocenter offset
            view_dir_x=r_diag, view_dir_y=r_diag, view_dir_z=0.0,
            up_vec_x=0.0, up_vec_y=0.0, up_vec_z=1.0,
            isocenter_x=5.0, isocenter_y=5.0, isocenter_z=0.0,
        ),
        "Cranial-Caudal (edge case)": GeometryConfig(  # looks straight down the default up-vector's own axis
            view_dir_x=0.0, view_dir_y=0.0, view_dir_z=-1.0,
            up_vec_x=0.0, up_vec_y=1.0, up_vec_z=0.0,
        ),
    }

    results    = {}
    row_images = []   # [(view_name, {renderer_name: tensor}), ...]

    for view_name, geo in VIEWS.items():
        logger.info("  View: %s", view_name)
        images = {}

        # Each renderer is independent — a missing/failing one just leaves a gap in `images`.
        img_plastimatch = generate_plastimatch_drr(volume_tensor, voxel_spacing, RES, device, geometry=geo)
        if img_plastimatch is not None:
            images["plastimatch_gt"] = img_plastimatch

        img_mc = generate_mc_drr(volume_tensor, voxel_spacing, RES, device, geometry=geo)
        if img_mc is not None:
            images["monte_carlo"] = img_mc

        renderer, cameras = build_dvr_renderer(RES, n_pts, ct_size, voxel_spacing, device, geometry=geo)
        if renderer is not None:
            with torch.no_grad():
                images["dvr"] = renderer(vol, cameras, norm_type="normalized")

        drr_siddon, rot, xyz = build_diffdrr_renderer(RES, ct_size, subject, voxel_spacing, device, geometry=geo)
        if drr_siddon is not None:
            with torch.no_grad():
                images["diffdrr_siddon"] = drr_siddon(rot, xyz, parameterization="euler_angles", convention="ZXY")

        img_deepdrr = generate_deepdrr_drr(volume_tensor, voxel_spacing, RES, device, geometry=geo)
        if img_deepdrr is not None:
            images["deepdrr"] = img_deepdrr

        row_images.append((view_name, images))

        if "plastimatch_gt" not in images:
            logger.warning("    Plastimatch GT unavailable for this view — skipping metrics.")
            results[view_name] = None
            continue

        # Score every other renderer against this view's Plastimatch render.
        ref_img      = images["plastimatch_gt"]
        view_results = {}
        for name, img in images.items():
            if name == "plastimatch_gt":
                continue
            rmse, psnr, ssim = compute_metrics(img, ref_img)
            view_results[name] = {"rmse": rmse, "psnr": psnr, "ssim": ssim}
            logger.info("    %-16s SSIM=%.4f  PSNR=%6.2f dB  RMSE=%.4e", name, ssim, psnr, rmse)
        results[view_name] = view_results

    # ── Summary table: SSIM vs Plastimatch GT, per view ─────────────────────
    renderer_order = ["monte_carlo", "dvr", "diffdrr_siddon", "deepdrr"]
    logger.info("")
    logger.info("  SSIM vs Plastimatch GT, per view:")
    logger.info("    %-28s" + "".join(f"  {r:>14s}" for r in renderer_order), "View")
    for view_name, view_results in results.items():
        if view_results is None:
            continue
        row = "".join(
            f"  {view_results[r]['ssim']:14.4f}" if r in view_results else f"  {'N/A':>14s}"
            for r in renderer_order
        )
        logger.info("    %-28s%s", view_name, row)

    # ── Grid figure: one row per view, one column per renderer ──────────────
    col_order = ["plastimatch_gt", "monte_carlo", "dvr", "diffdrr_siddon", "deepdrr"]
    present   = [c for c in col_order if any(c in imgs for _, imgs in row_images)]

    if row_images and present:
        n_rows, n_cols = len(row_images), len(present)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), dpi=150)
        axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

        for i, (view_name, imgs) in enumerate(row_images):
            for j, rname in enumerate(present):
                ax = axes[i, j]
                if rname in imgs:
                    im_np = imgs[rname].detach().cpu().squeeze().numpy()
                    im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min() + 1e-8)
                    ax.imshow(im_np, cmap="gray")
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if i == 0:
                    ax.set_title(rname.replace("_gt", " [GT]").replace("_", " "), fontsize=10)
                if j == 0:
                    ax.set_ylabel(view_name, fontsize=9)

        fig.suptitle("Multi-View Compatibility (200×200)", fontsize=13, y=1.0)
        fig.tight_layout()
        fig.savefig(out_dir / "multiview_comparison.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("    Saved → %s", out_dir / "multiview_comparison.png")

    return results


def run_resolution_sweep(volume_tensor, subject, n_pts, ct_size, voxel_spacing, device, resolutions, out_dir):
    """
    Runs the same RMSE/PSNR/SSIM-vs-Plastimatch comparison as
    ``run_quality``, but across a list of resolutions instead of one, to
    track how image quality trends with output size.

    Args:
        volume_tensor: The CT density volume to render, already loaded into memory.
        subject: The torchio ``Subject`` object DiffDRR needs internally.
        n_pts: Number of samples DVR takes along each ray.
        ct_size: Side length of the CT volume, in voxels (volumes are assumed cubic).
        voxel_spacing: Physical size of one voxel, in millimeters.
        device: The GPU or CPU device the renderers should run on.
        resolutions: List of square image sizes (in pixels) to test; powers of 2 give the cleanest-looking trend line.
        out_dir: Folder to save the ``quality_vs_resolution.png`` line plot into.

    Returns:
        results: RMSE/PSNR/SSIM per renderer per resolution, ``None`` where
            Plastimatch itself failed to render.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 6 — Quality vs. Resolution")
    logger.info("─" * 64)

    vol     = volume_tensor.to(device)
    results = {}

    for res in resolutions:
        logger.info("  Resolution: %d×%d", res, res)
        images = {}

        # If Plastimatch itself fails, there's nothing to score against — skip this resolution.
        img_plastimatch = generate_plastimatch_drr(volume_tensor, voxel_spacing, res, device)
        if img_plastimatch is None:
            logger.warning("    Plastimatch GT unavailable at this resolution — skipping.")
            results[res] = None
            continue
        images["plastimatch_gt"] = img_plastimatch

        try:
            img_mc = generate_mc_drr(volume_tensor, voxel_spacing, res, device)
            if img_mc is not None:
                images["monte_carlo"] = img_mc
        except Exception as exc:
            logger.warning("    Monte Carlo     : ERROR — %s", exc)

        # Large resolutions can legitimately run out of VRAM — log and skip
        # that renderer instead of crashing the whole sweep.
        try:
            renderer, cameras = build_dvr_renderer(res, n_pts, ct_size, voxel_spacing, device)
            if renderer is not None:
                with torch.no_grad():
                    images["dvr"] = renderer(vol, cameras, norm_type="normalized")
        except Exception as exc:
            logger.warning("    DVR             : ERROR — %s", exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        try:
            drr_siddon, rot, xyz = build_diffdrr_renderer(res, ct_size, subject, voxel_spacing, device)
            if drr_siddon is not None:
                with torch.no_grad():
                    images["diffdrr_siddon"] = drr_siddon(rot, xyz, parameterization="euler_angles", convention="ZXY")
        except Exception as exc:
            logger.warning("    DiffDRR (Siddon): ERROR — %s", exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        try:
            img_deepdrr = generate_deepdrr_drr(volume_tensor, voxel_spacing, res, device)
            if img_deepdrr is not None:
                images["deepdrr"] = img_deepdrr
        except Exception as exc:
            logger.warning("    DeepDRR         : ERROR — %s", exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Score every renderer that succeeded against this resolution's Plastimatch render.
        ref_img     = images["plastimatch_gt"]
        res_results = {}
        for name, img in images.items():
            if name == "plastimatch_gt":
                continue
            rmse, psnr, ssim = compute_metrics(img, ref_img)
            res_results[name] = {"rmse": rmse, "psnr": psnr, "ssim": ssim}
            logger.info("    %-16s SSIM=%.4f  PSNR=%6.2f dB  RMSE=%.4e", name, ssim, psnr, rmse)
        results[res] = res_results

    # ── Line plot: SSIM / PSNR vs resolution, one line per renderer ─────────
    renderer_names = sorted({n for r in results.values() if r for n in r})
    valid_res      = [res for res in resolutions if results.get(res)]

    if valid_res and renderer_names:
        fig, (ax_ssim, ax_psnr) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

        for name in renderer_names:
            xs      = [res for res in valid_res if name in results[res]]
            ssim_ys = [results[res][name]["ssim"] for res in xs]
            psnr_ys = [results[res][name]["psnr"] for res in xs]
            ax_ssim.plot(xs, ssim_ys, marker="o", label=name)
            ax_psnr.plot(xs, psnr_ys, marker="o", label=name)

        for ax, ylabel, title in [
            (ax_ssim, "SSIM vs Plastimatch GT", "Quality vs Resolution — SSIM"),
            (ax_psnr, "PSNR (dB) vs Plastimatch GT", "Quality vs Resolution — PSNR"),
        ]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(valid_res)
            ax.set_xticklabels([str(r) for r in valid_res])
            ax.set_xlabel("Resolution (px)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        ax_ssim.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_vs_resolution.png")
        plt.close(fig)
        logger.info("    Saved → %s", out_dir / "quality_vs_resolution.png")

    return results


def run_dvr_sampling_ablation(volume_tensor, subject, ct_size, voxel_spacing, device, resolution, n_pts_list, warmup, repeats, out_dir):
    """
    Sweeps DVR's samples-per-ray parameter at a fixed resolution to measure
    the cost/accuracy trade-off against render time.

    Args:
        volume_tensor: The CT density volume to render, already loaded into memory.
        subject: Unused — kept only so this function's signature matches the other benchmarks.
        ct_size: Side length of the CT volume, in voxels (volumes are assumed cubic).
        voxel_spacing: Physical size of one voxel, in millimeters.
        device: The GPU or CPU device DVR should run on.
        resolution: Fixed square output size (in pixels) used for the whole sweep.
        n_pts_list: Sample-per-ray counts to try, ideally powers of 2.
        warmup: Number of untimed renders to run before timing each sample count.
        repeats: Number of timed renders to average per sample count.
        out_dir: Folder to save the ``dvr_sampling_ablation.png`` cost-vs-accuracy plot into.

    Returns:
        results: RMSE/PSNR/SSIM plus mean/std render time per sample count
            (DiffDRR excluded — no comparable knob); ``None`` entirely if
            Plastimatch was unavailable.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 7 — DVR Ray-Sample-Count Ablation")
    logger.info("─" * 64)

    vol = volume_tensor.to(device)

    # One Plastimatch render at this resolution is the reference for every n_pts value below.
    img_plastimatch = generate_plastimatch_drr(volume_tensor, voxel_spacing, resolution, device)
    if img_plastimatch is None:
        logger.warning("    Plastimatch GT unavailable — skipping ablation.")
        return None

    results = {}
    for n_pts in n_pts_list:
        renderer, cameras = build_dvr_renderer(resolution, n_pts, ct_size, voxel_spacing, device)
        if renderer is None:
            logger.info("    n_pts=%-5d DVR UNAVAILABLE", n_pts)
            results[n_pts] = None
            continue

        def _dvr():
            return renderer(vol, cameras, norm_type="normalized")

        # One render for quality metrics, then time_fn handles the speed side.
        with torch.no_grad():
            img = _dvr()
        rmse, psnr, ssim = compute_metrics(img, img_plastimatch)
        mean_ms, std_ms  = time_fn(_dvr, device, warmup, repeats)

        results[n_pts] = {
            "rmse": rmse, "psnr": psnr, "ssim": ssim,
            "mean_ms": mean_ms, "std_ms": std_ms,
        }
        logger.info(
            "    n_pts=%-5d SSIM=%.4f  PSNR=%6.2f dB  RMSE=%.4e  |  %6.1f ± %4.1f ms",
            n_pts, ssim, psnr, rmse, mean_ms, std_ms,
        )

    # ── Cost-vs-accuracy plot ────────────────────────────────────────────────
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        xs      = sorted(valid.keys())
        ssim_ys = [valid[n]["ssim"] for n in xs]
        time_ys = [valid[n]["mean_ms"] for n in xs]

        # Dual y-axis: SSIM (quality) on the left, render time (cost) on the right.
        fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=150)
        ax2 = ax1.twinx()
        l1, = ax1.plot(xs, ssim_ys, marker="o", color="tab:blue", label="SSIM vs Plastimatch GT")
        l2, = ax2.plot(xs, time_ys, marker="s", color="tab:orange", label="Render time (ms)")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([str(n) for n in xs])
        ax1.set_xlabel("DVR samples per ray (n_pts)")
        ax1.set_ylabel("SSIM vs Plastimatch GT", color="tab:blue")
        ax2.set_ylabel("Render time (ms)", color="tab:orange")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"DVR Cost vs Accuracy ({resolution}×{resolution})")
        ax1.legend(handles=[l1, l2], loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "dvr_sampling_ablation.png")
        plt.close(fig)
        logger.info("    Saved → %s", out_dir / "dvr_sampling_ablation.png")

    return results
