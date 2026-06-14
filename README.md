# Vortex Engine: High-Performance Tiled Matrix Multiplication Kernel

Vortex Engine is a high-performance CUDA implementation of shared-memory tiled matrix multiplication ($C = A \times B$) engineered to optimize memory bandwidth utilization on NVIDIA GPUs. The project bridges a custom C++/CUDA compilation pipeline directly into Python using **PyBind11**, allowing seamless integration with PyTorch tensors while bypassing standard global memory bottlenecks.

## 🚀 Architecture Overview

A naive matrix multiplication kernel is fundamentally bound by memory bandwidth, executing a global memory read for every arithmetic operation. Vortex Engine implements **Shared Memory Tiling** to decouple memory access from compute execution.

### Memory Hierarchy Optimization
1. **SRAM Staging:** Threads within a block cooperatively load a $32 \times 32$ data tile from high-latency global memory into ultra-fast, on-chip Shared Memory (SRAM).
2. **Data Reuse:** Once staged in SRAM, each element is reused 32 times by the thread block, reducing the total global memory transaction overhead by a factor proportional to the tile size.
3. **Synchronized Execution:** Implements strict algorithmic execution barriers (`__syncthreads()`) between memory phases to prevent data hazards and race conditions across warp schedules.

The theoretical floating-point operation count for the square matrices evaluated is:
$$FLOPS = 2 \times M \times K \times N$$

---

## 🛠️ Project Structure

```text
Vortex-Engine/
├── src/
│   ├── bind.cpp          # PyBind11 encapsulation layer
│   └── vortex_kernel.cu  # Hardware-level shared-memory CUDA kernel
├── setup.py              # Monkey-patched Ninja build configuration script
├── benchmark.py          # Microsecond-accurate performance profiling tool
└── .gitignore            # Repository rules tracking active source tracks
