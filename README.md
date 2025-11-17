# Vector-Additional

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![GPU: Optional](https://img.shields.io/badge/GPU-CUDA%20(optional)-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**Author:** Iman Eivazi  
**Project Purpose:** Educational demo for introductory CUDA-style parallel programming using Python and Numba

---

## Introduction

This repository contains a small, focused educational project that demonstrates the core idea of **data-parallel computing** using a simple **vector addition** example. The goal is to show how the same computation

> c[i] = a[i] + b[i]

can be implemented in two ways:

- As a **sequential loop** on the CPU  
- As a **data-parallel kernel** that can run on a CUDA-capable GPU (via Numba)

The project is intentionally minimal and is designed as a companion to an introductory report on CUDA fundamentals and parallelization patterns. It is aimed at students who are new to GPU programming and want to see a clean, well-documented code example they can read, run, and modify.

The code is written in Python and uses **NumPy** for array handling and **Numba** for JIT compilation and optional CUDA support. If a CUDA-capable NVIDIA GPU is available, the script can run a GPU-accelerated version of the vector addition. If not, it will gracefully fall back to the CPU implementation so that the project remains runnable on a standard laptop.

---

## Table of Contents

- [Introduction](#introduction)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Configuration](#configuration)  
- [Example Output](#example-output)  
- [Author](#author)  
- [License](#license)  

---

## Installation

### 1. Clone the repository

Use `git` to clone the project to your local machine:

```bash
git clone https://github.com/iman-eivazi/cuda-vector-add.git
cd cuda-vector-add
If you are not using Git, you can also download the repository as a ZIP from GitHub and extract it manually into a folder named cuda-vector-add.
```

2. (Optional) Create and activate a virtual environment
It is recommended to use a virtual environment to keep dependencies isolated:

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell or CMD):
# .venv\Scripts\activate
3. Install Python dependencies
If a requirements.txt file is present (recommended), you can install all required packages with:

pip install -r requirements.txt
A minimal requirements.txt could look like:

numpy
numba
Alternatively, you can install them manually:

pip install numpy numba
4. CUDA toolkit (optional but recommended)
If you want to run the GPU-accelerated part of the demo, you need:

An NVIDIA GPU that supports CUDA

NVIDIA drivers and CUDA toolkit installed (matching your GPU and OS)

If CUDA is not available, the script will skip the GPU benchmark and run on CPU only.

## Usage
After installation, you can run the demo script from the command line. The main script lives under the src/ directory:

python src/cuda_vector_add.py
By default, the script will:

Allocate two large 1D vectors (NumPy arrays)

Run a CPU implementation of vector addition and measure its execution time

Attempt to run a GPU implementation using Numba’s CUDA support (if CUDA is available)

Compare the results from CPU and GPU for correctness

Print timing information and a short summary to the console

Command-line options
The main script is designed to be configurable via simple command-line arguments:

--size or -n
Length of the vectors (number of elements).
Default: 10_000_000 (ten million elements, if memory allows).

--device
Execution device. One of:

cpu – force CPU-only execution

gpu – force GPU execution (errors if CUDA is not available)

auto – try GPU if available, otherwise fall back to CPU
Default: auto

--repeats
Number of repetitions for each implementation when measuring timing.
Default: 3

Example:

python src/cuda_vector_add.py --size 5000000 --device auto --repeats 5
This command runs the demo with vector size 5,000,000, attempts to use the GPU if available, and repeats each measurement 5 times to obtain more stable timing statistics.

If no CUDA-capable GPU is available, the script will either:

Fall back to CPU-only mode and explicitly inform the user, or

Skip the GPU benchmark and only run the CPU implementation

depending on the final implementation in cuda_vector_add.py.

## Project Structure
The repository is organized to clearly separate documentation, source code, and the accompanying report:

cuda-vector-add/
  README.md                # Project documentation (this file)
  requirements.txt         # Python dependencies (numpy, numba, etc.)
  .gitignore               # Ignore cache and temporary files

  src/
    cuda_vector_add.py     # Main Python script: CPU and GPU vector addition demo

  report/
    cuda_basics_report_fa.pdf   # (Optional) Persian report on CUDA basics and parallel patterns
Notes:

All executable logic for this demo lives in src/cuda_vector_add.py.

The report/ directory is intended for the accompanying academic/educational report (for example, in Persian) that explains CUDA fundamentals and the vector-add pattern in more detail.

Keeping src/ and report/ separate makes it easier to reuse the code in other contexts (e.g., other courses or labs).

## Configuration
The behavior of the demo is mainly controlled by three aspects:

Vector size (--size / -n)
Controls how many elements are stored in the input arrays. Larger sizes make timing differences between CPU and GPU more visible, but also require more memory and longer runtime.

Execution device (--device)
Selects whether the demo should run on:

cpu only

gpu only

auto mode (prefer GPU, fall back to CPU if CUDA is not available)

Number of repeats (--repeats)
Controls how many times each implementation (CPU/GPU) is executed for timing. Averaging over multiple runs reduces noise in performance measurements.

Internally, the script will typically:

Use a straightforward NumPy-based implementation for the CPU version.

Use Numba’s @cuda.jit decorator for the GPU kernel implementation.

Check for CUDA availability using Numba’s runtime utilities and adjust behavior accordingly.

All of these details are documented with in-code comments so that students can read through the source and relate it back to the CUDA concepts described in the report.

## Example Output
A typical run on a machine with a CUDA-capable GPU might produce output similar to:

Vector length: 10_000_000
Device mode: auto

Running CPU implementation...
CPU time (average over 3 runs): 0.85 s

Running GPU implementation...
GPU time (average over 3 runs): 0.06 s

Verifying results...
All results match: True

Speedup (CPU time / GPU time): ~14.2x
On a machine without a CUDA-capable GPU, you might see something like:

Vector length: 5_000_000
Device mode: auto

CUDA not available. Falling back to CPU-only execution.

Running CPU implementation...
CPU time (average over 3 runs): 0.42 s

GPU implementation was skipped (no CUDA device detected).
These numbers are for illustration only. Real timings depend heavily on the specific CPU, GPU, memory, and software environment.

From an educational perspective, the key takeaways are:

For large vector sizes, the GPU version can significantly outperform the CPU version when a CUDA-capable GPU is available.

For small vector sizes, the overhead of setup and data movement can dominate, and the CPU may be competitive or even faster.

The vector-add pattern is a simple but powerful way to understand how kernels, threads, blocks, and grids work in practice.

## Author
Iman Eivazi
Developer and author of this educational project.

This repository was created as part of a learning and teaching effort on:

Basic CUDA concepts (kernel, thread, block, grid)

Parallel patterns for simple data-parallel problems

Using Python and Numba to bridge between high-level code and GPU execution

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software, provided that the following license text is included in all copies or substantial portions of the software.

MIT License

Copyright (c) 2025 Iman Eivazi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


[def]: #license
