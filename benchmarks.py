"""
Benchmark execution tasks.

Contains functions to run speed, VRAM footprint, image quality,
and optimization convergence comparisons between the renderers.
"""

import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import get_logger
from renderers import (
    build_dvr_renderer,
    build_diffdrr_renderer,
    generate_plastimatch_drr,
    generate_mc_drr,
    generate_deepdrr_drr,
)
from renderers.config import DEFAULT_GEOMETRY
from utils import time_fn, peak_vram, zncc_loss, compute_metrics

logger = get_logger(__name__)


def run_speed(volume_tensor, subject, resolutions, n_pts, ct_size, voxel_spacing, device, warmup, repeats):
    """
    Measure per-frame rendering latency across image resolutions.

    Args:
        volume_tensor (torch.Tensor): CT density volume ``(1, 1, D, H, W)``.
        subject: torchio ``Subject`` for DiffDRR, or ``None``.
        resolutions (list[int]): Square image sizes to evaluate.
        n_pts (int): DVR sample points per ray.
        ct_size (int): CT volume side length in voxels.
        device (torch.device): Target compute device.
        warmup (int): Warm-up iterations before timing.
        repeats (int): Timed iterations per configuration.

    Returns:
        dict: Nested result dict with keys ``"resolutions"``, ``"dvr"``,
        ``"diffdrr"``.  Each renderer maps resolution → ``{"mean_ms", "std_ms"}``
        or ``None`` when unavailable.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 1 — Rendering Speed")
    logger.info("─" * 64)

    results = {"resolutions": resolutions, "dvr": {}, "diffdrr": {}, "deepdrr": {}}

    for res in resolutions:
        logger.info("  Resolution: %d×%d", res, res)
        vol = volume_tensor.to(device)

        # DVR
        renderer, cameras = build_dvr_renderer(res, n_pts, ct_size, voxel_spacing, device)
        if renderer is not None:
            def _dvr():
                return renderer(vol, cameras, norm_type="normalized")
            mean_ms, std_ms = time_fn(_dvr, device, warmup, repeats)
            logger.info("    DVR     : %7.1f ± %5.1f ms", mean_ms, std_ms)
            results["dvr"][res] = {"mean_ms": mean_ms, "std_ms": std_ms}
        else:
            logger.info("    DVR     : UNAVAILABLE")
            results["dvr"][res] = None

        # DiffDRR
        drr, rot, xyz = build_diffdrr_renderer(res, ct_size, subject, voxel_spacing, device)
        if drr is not None:
            def _diffdrr():
                return drr(rot, xyz, parameterization="euler_angles", convention="ZXY")
            mean_ms, std_ms = time_fn(_diffdrr, device, warmup, repeats)
            logger.info("    DiffDRR : %7.1f ± %5.1f ms", mean_ms, std_ms)
            results["diffdrr"][res] = {"mean_ms": mean_ms, "std_ms": std_ms}
        else:
            logger.info("    DiffDRR : UNAVAILABLE")
            results["diffdrr"][res] = None

        # DeepDRR (forward pass only; not differentiable)
        def _deepdrr():
            return generate_deepdrr_drr(volume_tensor, voxel_spacing, res, device)
        img_check = _deepdrr()
        if img_check is not None:
            mean_ms, std_ms = time_fn(_deepdrr, device, warmup, repeats)
            logger.info("    DeepDRR : %7.1f ± %5.1f ms", mean_ms, std_ms)
            results["deepdrr"][res] = {"mean_ms": mean_ms, "std_ms": std_ms}
        else:
            logger.info("    DeepDRR : UNAVAILABLE")
            results["deepdrr"][res] = None

    return results


def run_vram(volume_tensor, subject, resolutions, n_pts, ct_size, voxel_spacing, device):
    """
    Measure peak GPU memory for forward and forward+backward passes.

    Args:
        volume_tensor (torch.Tensor): CT density volume ``(1, 1, D, H, W)``.
        subject: torchio ``Subject`` for DiffDRR, or ``None``.
        resolutions (list[int]): Square image sizes to evaluate.
        n_pts (int): DVR sample points per ray.
        ct_size (int): CT volume side length in voxels.
        device (torch.device): Must be a CUDA device; VRAM is 0 on CPU.

    Returns:
        dict: Nested result dict with keys ``"resolutions"``, ``"dvr"``,
        ``"diffdrr"``.  Each renderer maps resolution → ``{"fwd_mb", "bwd_mb"}``
        or ``None`` when unavailable.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 2 — VRAM Footprint (peak MB)")
    logger.info("─" * 64)

    results = {"resolutions": resolutions, "dvr": {}, "diffdrr": {}, "deepdrr": {}}

    for res in resolutions:
        logger.info("  Resolution: %d×%d", res, res)
        
        # To accurately measure backward pass VRAM, we need something to require gradients.
        vol = volume_tensor.to(device).clone().requires_grad_(True)

        # DVR
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

        # DiffDRR
        drr, rot, xyz = build_diffdrr_renderer(res, ct_size, subject, voxel_spacing, device)
        if drr is not None:
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

        # DeepDRR (uses cupy internally; torch.cuda memory counters won't reflect
        # cupy allocations, so fwd_mb is an approximation of PyTorch-visible VRAM)
        def _fwd_deepdrr():
            return generate_deepdrr_drr(volume_tensor, voxel_spacing, res, device)
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
    Render a shared scene with each renderer and compare image quality.

    Uses a fixed 200x200 resolution.  Quality metrics (RMSE / PSNR / SSIM)
    are computed against the Plastimatch Ground Truth.

    Args:
        volume_tensor (torch.Tensor): CT density volume ``(1, 1, D, H, W)``.
        subject: torchio ``Subject`` for DiffDRR, or ``None``.
        n_pts (int): DVR sample points per ray.
        ct_size (int): CT volume side length in voxels.
        voxel_spacing (float): Isotropic voxel size in mm.
        device (torch.device): Target compute device.
        out_dir (pathlib.Path): Directory to save ``quality_comparison.png``.

    Returns:
        dict: Maps renderer name → ``{"rmse", "psnr", "ssim", "reference"}``.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 3 — Image Quality (200×200)")
    logger.info("─" * 64)

    RES     = 200
    results = {}
    images  = {}
    vol     = volume_tensor.to(device)

    # ── Ground Truth references ───────────────────────────────────────────────

    # Plastimatch GT (exact geometric ray-tracing on CPU)
    from renderers.plastimatch import generate_plastimatch_drr
    img_plastimatch = generate_plastimatch_drr(volume_tensor, voxel_spacing, RES, device)
    images["plastimatch_gt"] = img_plastimatch
    logger.info("    Plastimatch (GT)      rendered: shape %s", tuple(img_plastimatch.shape))

    # Monte Carlo GT (polychromatic + scatter on CPU)
    img_mc = generate_mc_drr(volume_tensor, voxel_spacing, RES, device)
    if img_mc is not None:
        images["monte_carlo_gt"] = img_mc
        logger.info("    Monte Carlo (GT)      rendered: shape %s", tuple(img_mc.shape))
    else:
        logger.warning("    Monte Carlo (GT)      : FAILED")

    # ── Fast / differentiable renderers ───────────────────────────────────────
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
        from diffdrr.drr import DRR as DiffDRR_module
        delx = ct_size * voxel_spacing / RES
        drr_tri = DiffDRR_module(
            subject,
            sdd      = DEFAULT_GEOMETRY.sdd,
            height   = RES,
            delx     = delx,
            renderer = "trilinear",
        ).to(device)
        with torch.no_grad():
            img_trilinear = drr_tri(rot, xyz, parameterization="euler_angles", convention="ZXY")
            img_trilinear = torch.flip(img_trilinear, dims=[-1])
        images["diffdrr_trilinear"] = img_trilinear
        logger.info("    DiffDRR (Trilinear) rendered: shape %s", tuple(img_trilinear.shape))
    except Exception as exc:
        logger.debug("    DiffDRR trilinear: skipped (%s)", exc)

    # DeepDRR (physics-based, DL scatter, validated against MC)
    img_deepdrr = generate_deepdrr_drr(volume_tensor, voxel_spacing, RES, device)
    if img_deepdrr is not None:
        images["deepdrr"] = img_deepdrr
        logger.info("    DeepDRR           rendered: shape %s", tuple(img_deepdrr.shape))
    else:
        logger.info("    DeepDRR           : UNAVAILABLE (requires deepdrr + CUDA/Linux)")

    # ── Metrics vs both Ground Truth references ───────────────────────────────
    # Primary GT: Plastimatch (geometric accuracy benchmark, as in DiffDRR paper).
    # Secondary GT: Monte Carlo (physical accuracy benchmark).
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

    # Save side-by-side comparison figure
    if images:
        n    = len(images)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)
        if n == 1:
            axes = [axes]
        for ax, (name, img) in zip(axes, images.items()):
            im_np = img.detach().cpu().squeeze().numpy()
            im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min() + 1e-8)
            ax.imshow(im_np, cmap="gray")
            # Add GT marker to title for clarity
            title = name.replace("_gt", " [GT]").replace("_", "\n")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle("Image Quality Comparison (200×200)", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "quality_comparison.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("    Saved → %s", out_dir / "quality_comparison.png")

    return results


def run_optimization(volume_tensor, subject, n_pts, ct_size, voxel_spacing, opt_iters, device, out_dir):
    """
    Benchmark 2D/3D registration convergence via gradient descent.

    Both renderers start from a perturbed pose (±15° yaw) and optimise
    toward a ground-truth target using ZNCC loss and Adam.

    Args:
        volume_tensor (torch.Tensor): CT density volume ``(1, 1, D, H, W)``.
        subject: torchio ``Subject`` for DiffDRR, or ``None``.
        n_pts (int): DVR sample points per ray.
        ct_size (int): CT volume side length in voxels.
        opt_iters (int): Maximum number of optimisation iterations.
        device (torch.device): Target compute device.
        out_dir (pathlib.Path): Directory to save
            ``registration_convergence.png``.

    Returns:
        dict: Maps ``"dvr"`` / ``"diffdrr"`` →
        ``{"converged_iter", "elapsed_s", "final_zncc", "loss_curve"}``
        or ``None`` when the renderer is unavailable.
    """
    logger.info("")
    logger.info("─" * 64)
    logger.info("  BENCHMARK 4 — 2D/3D Registration (Gradient Descent)")
    logger.info("─" * 64)

    OPT_RES     = 150
    CONVERGENCE = -0.99
    LR          = 1e-2
    PERTURB_DEG = 15.0
    opt_results = {}
    vol         = volume_tensor.to(device)

    # ── DVR registration ──────────────────────────────────────────────────────
    try:
        from renderers.dvr import ObjectCentricXRayVolumeRenderer
        from pytorch3d.renderer.cameras import FoVPerspectiveCameras

        logger.info("")
        logger.info("  [DVR] Setting up registration ...")
        renderer, cam_gt = build_dvr_renderer(OPT_RES, n_pts, ct_size, voxel_spacing, device)

        with torch.no_grad():
            target_dvr = renderer(vol, cam_gt, norm_type="normalized")

        # Start from a perturbed rotation
        R_gt = cam_gt.R[0]
        # DiffDRR perturb_rad is around X axis. 
        # For PyTorch3D (where camera looks along +Z and Y is down), 
        # rotation around X axis matches Pitch (which DiffDRR also does).
        perturb_rad = float(PERTURB_DEG * np.pi / 180.0)
        R_perturb = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.0, np.cos(perturb_rad), -np.sin(perturb_rad)],
            [0.0, np.sin(perturb_rad), np.cos(perturb_rad)],
        ]], device=device)
        
        # Apply perturbation
        R_init = R_perturb @ R_gt.unsqueeze(0)
        
        T_opt = cam_gt.T.clone().requires_grad_(True)
        R_opt = R_init.clone().requires_grad_(True)

        optimizer      = torch.optim.Adam([R_opt, T_opt], lr=LR)
        losses_dvr     = []
        converged_iter = opt_iters
        t0             = time.perf_counter()

        for it in range(opt_iters):
            optimizer.zero_grad()
            
            # Reconstruct camera with updated R and T, keeping focal_length and principal_point
            cam_cur = cam_gt.clone()
            cam_cur.R = R_opt
            cam_cur.T = T_opt
            
            img_cur = renderer(vol, cam_cur, norm_type="normalized")
            loss    = zncc_loss(img_cur, target_dvr.detach())
            loss.backward()
            optimizer.step()

            zncc_val = -loss.item()
            losses_dvr.append(loss.item())

            if it % 50 == 0 or it < 5:
                logger.debug("    iter %4d  ZNCC=%.4f", it, zncc_val)

            if zncc_val >= abs(CONVERGENCE):
                converged_iter = it + 1
                logger.info("    ✓ DVR converged at iteration %d", converged_iter)
                break

        elapsed = time.perf_counter() - t0
        opt_results["dvr"] = {
            "converged_iter": converged_iter,
            "elapsed_s": round(elapsed, 2),
            "final_zncc": -losses_dvr[-1],
            "loss_curve": losses_dvr,
        }
        logger.info(
            "    DVR  total time: %.1f s  | final ZNCC: %.4f",
            elapsed, -losses_dvr[-1],
        )

    except ImportError as exc:
        logger.info("  [DVR] Optimisation skipped: %s", exc)
        opt_results["dvr"] = None

    # ── DiffDRR registration ──────────────────────────────────────────────────
    try:
        from diffdrr.drr import DRR as DiffDRR_module

        logger.info("")
        logger.info("  [DiffDRR] Setting up registration ...")
        delx = ct_size * voxel_spacing / OPT_RES

        drr     = DiffDRR_module(subject, sdd=DEFAULT_GEOMETRY.sdd, height=OPT_RES, delx=delx).to(device)
        rot_gt  = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        xyz_gt  = torch.tensor([[
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_x,
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_y,
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_z,
        ]], device=device)
        with torch.no_grad():
            target_ddrr = drr(rot_gt, xyz_gt, parameterization="euler_angles", convention="ZXY")
            target_ddrr = torch.flip(target_ddrr, dims=[-1])

        perturb_rad = float(PERTURB_DEG * np.pi / 180.0)
        rot_opt = torch.tensor([[perturb_rad, 0.0, 0.0]], device=device, requires_grad=True)
        xyz_opt = torch.tensor([[
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_x,
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_y,
            DEFAULT_GEOMETRY.sad * DEFAULT_GEOMETRY.view_dir_z,
        ]], device=device, requires_grad=True)

        optimizer      = torch.optim.Adam([rot_opt, xyz_opt], lr=LR)
        losses_ddrr    = []
        converged_iter = opt_iters
        t0             = time.perf_counter()

        for it in range(opt_iters):
            optimizer.zero_grad()
            img_cur  = drr(rot_opt, xyz_opt, parameterization="euler_angles", convention="ZXY")
            img_cur  = torch.flip(img_cur, dims=[-1])
            loss     = zncc_loss(img_cur, target_ddrr.detach())
            loss.backward()
            optimizer.step()

            zncc_val = -loss.item()
            losses_ddrr.append(loss.item())

            if it % 50 == 0 or it < 5:
                logger.debug("    iter %4d  ZNCC=%.4f", it, zncc_val)

            if zncc_val >= abs(CONVERGENCE):
                converged_iter = it + 1
                logger.info("    ✓ DiffDRR converged at iteration %d", converged_iter)
                break

        elapsed = time.perf_counter() - t0
        opt_results["diffdrr"] = {
            "converged_iter": converged_iter,
            "elapsed_s": round(elapsed, 2),
            "final_zncc": -losses_ddrr[-1],
            "loss_curve": losses_ddrr,
        }
        logger.info(
            "    DiffDRR total time: %.1f s  | final ZNCC: %.4f",
            elapsed, -losses_ddrr[-1],
        )

    except ImportError as exc:
        logger.info("  [DiffDRR] Optimisation skipped: %s", exc)
        opt_results["diffdrr"] = None

    # ── Convergence plot ──────────────────────────────────────────────────────
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
