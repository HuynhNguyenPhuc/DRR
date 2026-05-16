"""
Main entrypoint for benchmarking DVR vs DiffDRR.

This script runs a comprehensive set of benchmarks (speed, VRAM, quality, and optimization)
for a given set of CT volume files or directories containing NIfTI files.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load_ct_volume, make_diffdrr_subject
from plotter import plot_speed, plot_vram, print_summary
from benchmarks import run_speed, run_vram, run_quality, run_optimization
from utils.logger import setup_logger, get_logger


# --- Logger --- #
setup_logger()

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DVR vs DiffDRR benchmark")
    
    parser.add_argument("--ct",          type=str,   nargs="*", default=None,
                        help="CT NIfTI path(s) or directories (.nii/.nii.gz). Defaults to DiffDRR example CT.")
    parser.add_argument("--size",        type=int,   default=128,
                        help="Resample CT to this cubic size (default: 128).")
    parser.add_argument("--warmup",      type=int,   default=5,
                        help="Warmup iterations before timing (default: 5).")
    parser.add_argument("--repeats",     type=int,   default=20,
                        help="Measurement iterations for timing (default: 20).")
    parser.add_argument("--resolutions", type=int,   nargs="+",
                        default=[100, 200, 300, 500],
                        help="Image resolutions to benchmark (default: 100 200 300 500).")
    parser.add_argument("--n-pts",       type=int,   default=320,
                        help="DVR: sample points per ray (default: 320).")
    parser.add_argument("--opt-iters",   type=int,   default=300,
                        help="Max registration optimisation iterations (default: 300).")
    parser.add_argument("--output",      type=str,   default="benchmark_results",
                        help="Output directory for plots and JSON (default: ./benchmark_results).")
    args = parser.parse_args()

    CT_SIZE     = args.size
    WARMUP      = args.warmup
    REPEATS     = args.repeats
    RESOLUTIONS = args.resolutions
    N_PTS       = args.n_pts
    OPT_ITERS   = args.opt_iters
    OUT_DIR     = Path(args.output)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    logger.info("=" * 64)
    logger.info("  DVR vs DiffDRR — Comprehensive Benchmark")
    logger.info("=" * 64)
    logger.info(
        "  Device      : %s%s",
        DEVICE,
        f"  [{torch.cuda.get_device_name(0)}]" if DEVICE.type == "cuda" else "",
    )
    logger.info("  CT size     : %d³  |  N_pts/ray (DVR): %d", CT_SIZE, N_PTS)
    logger.info("  Resolutions : %s", RESOLUTIONS)
    logger.info("  Warmup/Reps : %d / %d", WARMUP, REPEATS)
    logger.info("=" * 64)

    RESULTS_ALL = {}

    # ── Resolve CT Paths ──────────────────────────────────────────────────────
    ct_files = []
    if args.ct is not None:
        for path_str in args.ct:
            p = Path(path_str)
            if p.is_dir():
                for ext in ["*.nii", "*.nii.gz"]:
                    ct_files.extend(list(p.rglob(ext)))
            elif p.is_file():
                ct_files.append(p)
            else:
                logger.warning(f"Path not found: {path_str}")
        ct_files = list(set(ct_files))
    
    if not ct_files:
        ct_files = [None]  # Default DiffDRR example

    for ct_idx, ct_path in enumerate(ct_files):
        if ct_path is not None:
            logger.info("")
            logger.info("=" * 64)
            logger.info("  Processing CT [%d/%d]: %s", ct_idx + 1, len(ct_files), ct_path)
            logger.info("=" * 64)
            ct_out_dir = OUT_DIR / ct_path.stem.replace(".nii", "")
        else:
            logger.info("")
            logger.info("=" * 64)
            logger.info("  Processing Default Example CT")
            logger.info("=" * 64)
            ct_out_dir = OUT_DIR / "example_ct"
        
        ct_out_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Load shared data ───────────────────────────────────────────────────
        volume_tensor, subject, voxel_spacing = load_ct_volume(str(ct_path) if ct_path else None, CT_SIZE)

        if subject is None:
            logger.info("[DATA] Building synthetic torchio Subject for DiffDRR ...")
            subject = make_diffdrr_subject(volume_tensor, voxel_spacing)
            if subject is None:
                logger.warning("[DATA] torchio not available — DiffDRR benchmarks will be skipped.")

        # ── 2. Run benchmarks ─────────────────────────────────────────────────────
        RESULTS = {}
        RESULTS["speed"]        = run_speed(volume_tensor, subject, RESOLUTIONS, N_PTS, CT_SIZE, voxel_spacing, DEVICE, WARMUP, REPEATS)
        RESULTS["vram"]         = run_vram(volume_tensor, subject, RESOLUTIONS, N_PTS, CT_SIZE, voxel_spacing, DEVICE)
        RESULTS["quality"]      = run_quality(volume_tensor, subject, N_PTS, CT_SIZE, voxel_spacing, DEVICE, ct_out_dir)
        RESULTS["optimization"] = run_optimization(volume_tensor, subject, N_PTS, CT_SIZE, voxel_spacing, OPT_ITERS, DEVICE, ct_out_dir)

        # ── 3. Print summary table ────────────────────────────────────────────────
        print_summary(RESULTS)

        # ── 4. Generate charts ────────────────────────────────────────────────────
        logger.info("  Generating plots for this CT ...")
        if RESULTS.get("speed"):
            plot_speed(RESULTS["speed"], ct_out_dir)
        if RESULTS.get("vram"):
            plot_vram(RESULTS["vram"], ct_out_dir)

        # ── 5. Save raw results as JSON ───────────────────────────────────────────
        def _serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {str(k): _serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serializable(i) for i in obj]
            return obj

        json_path = ct_out_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(_serializable(RESULTS), f, indent=2)

        RESULTS_ALL[str(ct_path) if ct_path else "example_ct"] = RESULTS
        logger.info("  Results saved to: %s/", ct_out_dir)
        logger.info("  Raw JSON        : %s", json_path)

    logger.info("=" * 64)
    logger.info("  All benchmarks finished!")
    logger.info("=" * 64)




if __name__ == "__main__":
    main()
