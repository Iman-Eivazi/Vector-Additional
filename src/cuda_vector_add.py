#!/usr/bin/env python3
"""
cuda-vector-add: Parallel vector addition using Numba and CUDA.

This script performs element-wise addition of two vectors using a GPU if available.
- Uses Numba's @cuda.jit decorator to define a CUDA kernel.
- Manages GPU memory transfers with cuda.to_device and copy_to_host.
- Falls back to a NumPy-based implementation if no CUDA GPU is found.
- Includes error checking and clear documentation for educational use.
Requirements: numba (CUDA-enabled) and numpy (install via pip).
"""

import sys
import math
import numpy as np
from numba import cuda

def vector_add_cpu(a, b):
    """
    Add two vectors on CPU using NumPy.

    Parameters:
        a, b (array-like): Input vectors of the same shape.
    Returns:
        numpy.ndarray: Element-wise sum of a and b.

    Raises:
        ValueError: If input shapes do not match.
    """
    # Convert inputs to NumPy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    # Error check: shapes must be equal
    if a.shape != b.shape:
        raise ValueError(f"vector_add_cpu: shapes {a.shape} and {b.shape} are not compatible")
    # NumPy does element-wise addition
    return a + b

@cuda.jit
def _vector_add_kernel(a, b, c):
    """
    CUDA kernel for vector addition: c[i] = a[i] + b[i].

    Each thread computes one element of the output array.
    """
    # Calculate the global thread index in a 1D grid
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

def vector_add_gpu(a, b):
    """
    Add two vectors on the GPU using a Numba CUDA kernel.

    Parameters:
        a, b (array-like): Input vectors of the same length.
    Returns:
        numpy.ndarray: The result vector (on host) containing a + b.

    Raises:
        RuntimeError: If no CUDA GPU is available.
        ValueError: If input sizes do not match.
    """
    # Check for CUDA GPU availability
    if not cuda.is_available():
        raise RuntimeError("No CUDA-capable GPU detected.")
    # Convert inputs to float32 NumPy arrays (common GPU dtype)
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    if a_np.shape != b_np.shape:
        raise ValueError(f"vector_add_gpu: shapes {a_np.shape} and {b_np.shape} do not match")
    n = a_np.size
    # Allocate device memory and copy inputs to GPU
    d_a = cuda.to_device(a_np)
    d_b = cuda.to_device(b_np)
    d_c = cuda.device_array_like(a_np)  # uninitialized result on GPU
    # Configure thread block and grid sizes
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    # Launch the CUDA kernel
    _vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    # Copy the result back to host memory
    return d_c.copy_to_host()

def main():
    """
    Main function to demonstrate vector addition.

    Generates two random vectors, attempts GPU addition, and falls back to CPU if needed.
    Verifies that results match for GPU vs. CPU.
    """
    # Example data size
    N = 1000000  # one million elements
    # Generate random vectors of float32 numbers
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)

    try:
        # Try GPU-accelerated vector addition
        result = vector_add_gpu(a, b)
        print("GPU vector_add succeeded.")
    except Exception as e:
        # If GPU is unavailable or any GPU error, fall back to CPU
        print(f"GPU vector_add failed ({e}). Falling back to CPU.")
        result = vector_add_cpu(a, b)

    # Verify correctness by comparing with NumPy result
    expected = a + b
    if np.allclose(result, expected):
        print("Result verified: output is correct.")
    else:
        print("Error: result does not match expected output.")
        sys.exit(1)

if __name__ == "__main__":
    main()

# I.E
