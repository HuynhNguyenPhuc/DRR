"""
Plotting and terminal summary utilities for benchmark results.

This module provides functions to visualize speed and VRAM measurements,
as well as a utility to print a formatted summary of all benchmarks.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Speed chart ───────────────────────────────────────────────────────────────

def plot_speed(speed_results: dict, out_dir) -> None:
    """
    Save a grouped bar chart of rendering latency vs image resolution.

    Args:
        speed_results (dict): Output from :func:`tasks.run_speed`.
        out_dir (pathlib.Path): Destination directory for
            ``speed_comparison.png``.
    """
    dvr_data    = speed_results.get("dvr",     {})
    ddrr_data   = speed_results.get("diffdrr", {})
    ddrr_data2  = speed_results.get("deepdrr", {})
    res_list    = speed_results.get("resolutions", [])

    renderers = [
        ("DVR (PyTorch3D)",    dvr_data,   "#2196F3"),
        ("DiffDRR (Siddon)",   ddrr_data,  "#FF5722"),
        ("DeepDRR",            ddrr_data2, "#4CAF50"),
    ]

    x     = np.arange(len(res_list))
    n_r   = len(renderers)
    width = 0.8 / n_r
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    for i, (label, data, color) in enumerate(renderers):
        means = [data.get(r, {}).get("mean_ms") if data.get(r) else None for r in res_list]
        stds  = [data.get(r, {}).get("std_ms", 0) if data.get(r) else 0 for r in res_list]
        if not any(v is not None for v in means):
            continue
        offset = (i - (n_r - 1) / 2.0) * width
        vals   = [v if v is not None else 0 for v in means]
        errs   = [e if means[j] is not None else 0 for j, e in enumerate(stds)]
        ax.bar(x + offset, vals, width, yerr=errs,
               label=label, capsize=4, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}×{r}" for r in res_list])
    ax.set_xlabel("Image Resolution")
    ax.set_ylabel("Render Time (ms)")
    ax.set_title("Rendering Speed Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "speed_comparison.png")
    plt.close(fig)
    logger.info("    Saved → %s", out_dir / "speed_comparison.png")


# ── VRAM chart ────────────────────────────────────────────────────────────────

def plot_vram(vram_results: dict, out_dir) -> None:
    """
    Save a dual bar chart of peak VRAM usage (forward and backward passes).

    Args:
        vram_results (dict): Output from :func:`tasks.run_vram`.
        out_dir (pathlib.Path): Destination directory for
            ``vram_comparison.png``.
    """
    dvr_data    = vram_results.get("dvr",     {})
    ddrr_data   = vram_results.get("diffdrr", {})
    ddrr_data2  = vram_results.get("deepdrr", {})
    res_list    = vram_results.get("resolutions", [])

    renderers = [
        ("DVR (PyTorch3D)",  dvr_data,  "#2196F3"),
        ("DiffDRR (Siddon)", ddrr_data, "#FF5722"),
        ("DeepDRR (fwd)",    ddrr_data2, "#4CAF50"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for ax, key, label in zip(
        axes,
        ["fwd_mb", "bwd_mb"],
        ["Forward Pass", "Forward + Backward Pass"],
    ):
        x     = np.arange(len(res_list))
        n_r   = len(renderers)
        width = 0.8 / n_r

        for i, (rname, data, color) in enumerate(renderers):
            vals = [data.get(r, {}).get(key, 0) or 0 if data.get(r) else 0 for r in res_list]
            if not any(v for v in vals):
                continue
            offset = (i - (n_r - 1) / 2.0) * width
            ax.bar(x + offset, vals, width, label=rname, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}×{r}" for r in res_list])
        ax.set_xlabel("Image Resolution")
        ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title(f"VRAM — {label}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "vram_comparison.png")
    plt.close(fig)
    logger.info("    Saved → %s", out_dir / "vram_comparison.png")


# ── Terminal summary ──────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """
    Print a formatted summary table to the console.

    Args:
        results (dict): Top-level results dict with optional keys
            ``"speed"``, ``"vram"``, ``"quality"``, ``"optimization"``.
    """
    logger.info("")
    logger.info("=" * 64)
    logger.info("  BENCHMARK SUMMARY")
    logger.info("=" * 64)

    # Speed table
    speed = results.get("speed", {})
    if speed:
        logger.info("")
        logger.info("  ■ Rendering Speed (ms, mean ± std)")
        logger.info("  %8s  %14s  %14s  %10s", "Res", "DVR", "DiffDRR", "Speedup")
        logger.info("  %s  %s  %s  %s", "-" * 8, "-" * 14, "-" * 14, "-" * 10)
        for r in speed.get("resolutions", []):
            d     = speed["dvr"].get(r)
            g     = speed["diffdrr"].get(r)
            d_str = f"{d['mean_ms']:6.1f}±{d['std_ms']:4.1f}" if d else "   N/A"
            g_str = f"{g['mean_ms']:6.1f}±{g['std_ms']:4.1f}" if g else "   N/A"
            s_str = f"{d['mean_ms'] / g['mean_ms']:6.2f}×" if (d and g) else "   N/A"
            logger.info("  %8s  %14s  %14s  %10s", f"{r}×{r}", d_str, g_str, s_str)

    # VRAM table
    vram = results.get("vram", {})
    if vram:
        logger.info("")
        logger.info("  ■ VRAM Footprint (MB peak)")
        logger.info(
            "  %8s  %9s  %9s  %12s  %12s",
            "Res", "DVR fwd", "DVR bwd", "DiffDRR fwd", "DiffDRR bwd",
        )
        logger.info("  %s  %s  %s  %s  %s",
                    "-" * 8, "-" * 9, "-" * 9, "-" * 12, "-" * 12)
        for r in vram.get("resolutions", []):
            d  = vram["dvr"].get(r)
            g  = vram["diffdrr"].get(r)
            df = f"{d['fwd_mb']:7.1f}" if d else "    N/A"
            db = f"{d['bwd_mb']:7.1f}" if d else "    N/A"
            gf = f"{g['fwd_mb']:7.1f}" if g else "       N/A"
            gb = f"{g['bwd_mb']:7.1f}" if g else "       N/A"
            logger.info("  %8s  %9s  %9s  %12s  %12s",
                        f"{r}×{r}", df, db, gf, gb)

    # Quality table
    quality = results.get("quality", {})
    if quality:
        logger.info("")
        logger.info("  ■ Image Quality vs Reference(s)")
        logger.info("  %22s  %10s  %10s  %8s  Ref",
                    "Method", "RMSE", "PSNR (dB)", "SSIM")
        logger.info("  %s  %s  %s  %s  ---",
                    "-" * 22, "-" * 10, "-" * 10, "-" * 8)
        for name, vals in quality.items():
            # vals may be a flat dict (legacy) or a nested dict keyed by GT name
            if isinstance(vals, dict) and any(
                isinstance(v, dict) for v in vals.values()
            ):
                for ref_key, metrics in vals.items():
                    logger.info(
                        "  %22s  %10s  %10.2f  %8.4f  %s",
                        name,
                        f"{metrics['rmse']:.4e}",
                        metrics["psnr"],
                        metrics["ssim"],
                        metrics.get("reference", ref_key),
                    )
            else:
                logger.info(
                    "  %22s  %10s  %10.2f  %8.4f  %s",
                    name,
                    f"{vals['rmse']:.4e}",
                    vals["psnr"],
                    vals["ssim"],
                    vals.get("reference", "?"),
                )

    # Optimisation table
    opt = results.get("optimization", {})
    if opt:
        logger.info("")
        logger.info("  ■ Registration (2D/3D) Convergence")
        logger.info("  %12s  %10s  %10s  %12s",
                    "Method", "Converged", "Time (s)", "Final ZNCC")
        logger.info("  %s  %s  %s  %s",
                    "-" * 12, "-" * 10, "-" * 10, "-" * 12)
        for method in ["dvr", "diffdrr"]:
            v     = opt.get(method)
            label = "DVR" if method == "dvr" else "DiffDRR"
            if v:
                logger.info("  %12s  %10s  %10.1f  %12.4f",
                            label, f"iter {v['converged_iter']}",
                            v["elapsed_s"], v["final_zncc"])
            else:
                logger.info("  %12s  %10s  %10s  %12s",
                            label, "N/A", "N/A", "N/A")

    logger.info("")
    logger.info("=" * 64)
