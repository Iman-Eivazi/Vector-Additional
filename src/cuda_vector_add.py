"""
cuda_vector_add.py

Educational demo for 1D vector addition on CPU and GPU using Python, NumPy, and Numba.
This script is designed to be simple, well-documented, and suitable for teaching
introductory CUDA-style parallel programming concepts.

Author: Iman Eivazi
Year: 2025
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Literal, Tuple, List, Optional

import numpy as np

try:
    # Numba is required for CUDA support in this demo.
    from numba import cuda
    from numba.cuda.cudadrv.error import CudaSupportError
except Exception:  # pragma: no cover - fallback if Numba/CUDA is not available
    cuda = None  # type: ignore[assignment]
    CudaSupportError = Exception  # type: ignore[assignment]


DeviceMode = Literal["cpu", "gpu", "auto"]


@dataclass
class Config:
    """
    Simple configuration container for the demo.

    Attributes:
        size: Number of elements in each input vector.
        device: Execution device mode: "cpu", "gpu", or "auto".
        repeats: Number of repetitions for timing measurements.
    """

    size: int = 10_000_000
    device: DeviceMode = "auto"
    repeats: int = 3


def parse_args(argv: Optional[List[str]] = None) -> Config:
    """
    Parse command-line arguments into a Config object.

    Args:
        argv: Optional list of argument strings. If None, sys.argv is used.

    Returns:
        A Config instance with parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Educational demo: 1D vector addition on CPU and optionally on a "
            "CUDA-capable GPU using Numba."
        )
    )
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=10_000_000,
        help=(
            "Number of elements in each input vector. Larger values make GPU "
            "speedups more visible but require more memory."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help=(
            "Execution device mode. "
            "'cpu' = CPU only, 'gpu' = GPU only (requires CUDA), "
            "'auto' = prefer GPU if available, otherwise fall back to CPU."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help=(
            "Number of repetitions for each implementation when measuring "
            "execution time. The reported time is the average."
        ),
    )

    args = parser.parse_args(argv)

    if args.size <= 0:
        parser.error("Vector size must be a positive integer.")
    if args.repeats <= 0:
        parser.error("Number of repeats must be a positive integer.")

    return Config(size=args.size, device=args.device, repeats=args.repeats)


def cuda_is_available() -> bool:
    """
    Safely check whether CUDA is available through Numba.

    Returns:
        True if CUDA appears to be available and usable, False otherwise.
    """
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except CudaSupportError:
        return False
    except Exception:
        # Any unexpected error here is treated as "CUDA not available".
        return False


if cuda is not None:

    @cuda.jit
    def vector_add_kernel(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """
        CUDA kernel for element-wise vector addition.

        Each thread computes one element of the output vector c, if the
        computed global index is within bounds.

        Args:
            a: First input vector on the device.
            b: Second input vector on the device.
            c: Output vector on the device.
        """
        # Compute a 1D global index for this thread.
        # In CUDA terms: global_index = blockIdx.x * blockDim.x + threadIdx.x
        idx = cuda.grid(1)

        # Guard against threads whose index is outside the array bounds.
        if idx < a.size:
            c[idx] = a[idx] + b[idx]


def run_cpu_vector_add(
    a: np.ndarray,
    b: np.ndarray,
    repeats: int,
) -> Tuple[np.ndarray, float]:
    """
    Run a simple CPU implementation of vector addition and measure its time.

    The CPU implementation uses a plain Python loop over all indices. This is
    intentionally simple and explicit to emphasize the sequential nature of
    the computation.

    Args:
        a: First input vector (NumPy array on host).
        b: Second input vector (NumPy array on host).
        repeats: How many times to repeat the computation for timing.

    Returns:
        A tuple (result_vector, average_time_seconds).
    """
    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")

    result = np.empty_like(a)
    durations: List[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        # Plain for-loop over all elements.
        for i in range(a.size):
            result[i] = a[i] + b[i]
        end = time.perf_counter()
        durations.append(end - start)

    average_time = float(sum(durations) / len(durations))
    # Return a copy so that the caller has a stable reference to the final result.
    return result.copy(), average_time


def run_gpu_vector_add(
    a: np.ndarray,
    b: np.ndarray,
    repeats: int,
) -> Tuple[np.ndarray, float]:
    """
    Run a GPU implementation of vector addition (if CUDA is available).

    This function:
      * Checks for CUDA availability.
      * Copies input vectors to the GPU.
      * Launches the CUDA kernel multiple times for timing.
      * Copies the result back to the host.

    The timing focuses on the kernel execution, not on host-device transfers.

    Args:
        a: First input vector (NumPy array on host).
        b: Second input vector (NumPy array on host).
        repeats: How many times to repeat the computation for timing.

    Returns:
        A tuple (result_vector, average_time_seconds).

    Raises:
        RuntimeError: If CUDA is not available or Numba's CUDA support cannot be used.
    """
    if not cuda_is_available():
        raise RuntimeError("CUDA is not available on this system.")

    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape.")

    if cuda is None:
        # This should not happen if cuda_is_available() returned True.
        raise RuntimeError("Numba CUDA module is not available.")

    n = a.size

    # Transfer input arrays to the device (GPU memory).
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    # Allocate space for the result on the device.
    d_c = cuda.device_array_like(a)

    # Choose a 1D execution configuration.
    threads_per_block = 256
    blocks_per_grid = math.ceil(n / threads_per_block)

    durations: List[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        # Launch the kernel with the chosen configuration.
        vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
        # Synchronize to ensure the kernel has finished before measuring time.
        cuda.synchronize()
        end = time.perf_counter()
        durations.append(end - start)

    # Copy the result back to the host.
    result = d_c.copy_to_host()
    average_time = float(sum(durations) / len(durations))
    return result, average_time


def print_header(config: Config) -> None:
    """
    Print a short header summarizing the configuration.
    """
    print("=== cuda-vector-add demo ===")
    print(f"Vector length: {config.size}")
    print(f"Device mode: {config.device}")
    print(f"Repeats per implementation: {config.repeats}")
    print("-" * 40)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the cuda-vector-add demo.

    This function:
      * Parses command-line arguments.
      * Allocates random input vectors.
      * Runs the CPU implementation.
      * Optionally runs the GPU implementation depending on the device mode.
      * Compares the results and prints timing information.

    Args:
        argv: Optional list of argument strings. If None, sys.argv is used.

    Returns:
        Process exit code: 0 on success, non-zero on error.
    """
    config = parse_args(argv)
    print_header(config)

    # Use a fixed random seed for reproducibility.
    rng = np.random.default_rng(seed=42)

    # Allocate input vectors with random float32 values.
    a = rng.random(config.size, dtype=np.float32)
    b = rng.random(config.size, dtype=np.float32)

    # Always run the CPU implementation so that we have a reference result.
    print("Running CPU implementation...")
    cpu_result, cpu_time = run_cpu_vector_add(a, b, config.repeats)
    print(f"CPU time (average over {config.repeats} runs): {cpu_time:.6f} s")
    print()

    # Decide whether and how to run the GPU implementation.
    run_gpu = False
    if config.device == "gpu":
        if not cuda_is_available():
            print("ERROR: Device mode set to 'gpu', but CUDA is not available.")
            return 1
        run_gpu = True
    elif config.device == "auto":
        if cuda_is_available():
            run_gpu = True
        else:
            print("CUDA not available. Falling back to CPU-only execution.")
            run_gpu = False
    else:
        # Device mode is explicitly "cpu".
        run_gpu = False

    if run_gpu:
        print("Running GPU implementation...")
        try:
            gpu_result, gpu_time = run_gpu_vector_add(a, b, config.repeats)
        except RuntimeError as exc:
            print(f"Failed to run GPU implementation: {exc}")
            return 1

        # Verify that CPU and GPU results match within a reasonable tolerance.
        matches = np.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
        print(f"All results match: {matches}")
        print(f"GPU time (average over {config.repeats} runs): {gpu_time:.6f} s")

        if matches and cpu_time > 0.0:
            speedup = cpu_time / gpu_time if gpu_time > 0.0 else float("inf")
            print(f"Speedup (CPU time / GPU time): {speedup:.2f}x")
    else:
        print("GPU implementation was skipped (CPU-only run).")

    print("\nDemo finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# I.E