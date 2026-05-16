"""Profiling utilities."""

import time
import numpy as np
import torch


def time_fn(fn, device: torch.device, warmup: int = 5, n: int = 20):
    """
    Measure the average wall-clock time of a callable.

    Args:
        fn: Zero-argument callable to time.
        device (torch.device): Target device (``"cuda"`` or ``"cpu"``).
        warmup (int): Number of warm-up iterations (default: 5).
        n (int): Number of timed iterations (default: 20).

    Returns:
        tuple[float, float]: ``(mean_ms, std_ms)`` — mean and standard
        deviation of per-call latency in milliseconds.
    """
    # Warmup the device and the function to get stable timing measurements.
    for _ in range(warmup):
        with torch.no_grad():
            fn()

    if device.type == "cuda":
        # Synchronize before starting timing to ensure warmup is complete.
        torch.cuda.synchronize()

        # Use CUDA events for accurate GPU timing.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n)]

        for i in range(n):
            # Record the start event before the function call.
            starts[i].record()

            # Run the function without tracking gradients.
            with torch.no_grad():
                fn()

            # Record the end event after the function call.
            ends[i].record()

        # Synchronize to ensure all events have completed before measuring elapsed time.
        torch.cuda.synchronize()

        # Compute elapsed time in milliseconds for each iteration.
        ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        ms = []
        for _ in range(n):
            # Use time.perf_counter() for high-resolution timing on CPU.
            t0 = time.perf_counter()

            # Run the function without tracking gradients.
            with torch.no_grad():
                fn()

            # Record elapsed time in milliseconds.
            ms.append((time.perf_counter() - t0) * 1000)

    # Return the mean and standard deviation of the timings.
    return float(np.mean(ms)), float(np.std(ms))


def peak_vram(fwd_fn, device: torch.device):
    """
    Measure peak VRAM for a forward pass and a forward+backward pass.

    Args:
        fwd_fn: Zero-argument callable that returns model output(s).
        device (torch.device): Must be a CUDA device; returns ``(0.0, 0.0)``
            on CPU.

    Returns:
        tuple[float, float]: ``(fwd_mb, bwd_mb)`` — peak allocated memory
        in MiB for the forward pass and for the combined forward+backward
        pass respectively.
    """
    if device.type != "cuda":
        return 0.0, 0.0

    # Empty the cache and reset peak memory stats before each measurement to get accurate readings.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run the forward pass.
    out = fwd_fn()

    # Synchronize to ensure all GPU operations have completed before measuring memory usage.    
    torch.cuda.synchronize()

    # Measure peak memory allocated during the forward pass.
    fwd_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    del out
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run the forward pass.
    out  = fwd_fn()

    # Get the loss value to backpropagate.
    loss = out[0].sum() if isinstance(out, (list, tuple)) else out.sum()

    # Backward pass to measure additional memory used for gradients.
    loss.backward()

    # Synchronize again to ensure all GPU operations have completed before measuring memory usage.
    torch.cuda.synchronize()

    # Measure peak memory allocated during the combined forward and backward pass.
    bwd_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    # Return the peak memory usage for the forward pass and the combined forward+backward pass.
    return round(fwd_mb, 1), round(bwd_mb, 1)
