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

```

---

## 📊 Performance Benchmarks

Profiling was executed on local silicon using microsecond-accurate CUDA hardware events. Execution times represent the mathematical average across multiple warm-up and evaluation iterations.

| Matrix Dimensions ($M=K=N$) | Native PyTorch Time (ms) | Vortex Engine Time (ms) | Mathematical Verification | Architectural Profile |
| --- | --- | --- | --- | --- |
| **$1024 \times 1024$** | 0.296 ms | 2.304 ms | **PASSED** | 7.79x baseline latency |
| **$2048 \times 2048$** | 2.453 ms | 17.954 ms | **PASSED** | 7.32x baseline latency |
| **$4096 \times 4096$** | 17.833 ms | 153.500 ms | **PASSED** | 8.61x baseline latency |

### Critical Analysis

The implementation achieves **100% mathematical parity** with native PyTorch across all operational dimensions up to 68 billion operations ($4096^3$).

The baseline performance delta relative to native PyTorch (`torch.matmul`) highlights the delta between optimized CUDA C++ and assembly-level libraries:

* **Tensor Core Utilization:** Native PyTorch triggers NVIDIA's proprietary `cuBLAS` backend, routing execution through dedicated hardware Tensor Cores capable of executing mixed-precision matrix math instantly. Vortex Engine executes strictly on standard CUDA FP32 compute cores.
* **Instruction Level Tiling:** To close the remaining gap with cuBLAS, the architecture requires extension from shared memory tiling to 2D register tiling (moving data directly into physical thread registers) and memory vectorization via `float4` data loads.

---

## 🔧 Building and Running Locally

### Prerequisites

* Windows 10/11 x64
* NVIDIA GPU + CUDA Toolkit v13.1 (or compatible)
* Python 3.12 (industry-stable target for PyTorch extensions)
* C++ Build Tools (Microsoft Visual Studio Build Tools)

### Installation & Build

1. Clone the repository and navigate to the root folder:
```cmd
git clone [https://github.com/your-username/Vortex-Engine.git](https://github.com/your-username/Vortex-Engine.git)
cd Vortex-Engine

```


2. Initialize and activate your isolated environment:
```cmd
python -m venv venv
.\venv\Scripts\activate

```


3. Install compilation tools and CUDA-enabled PyTorch:
```cmd
pip install ninja
pip install torch --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

```


4. Build the extension module directly in place:
```cmd
python setup.py build_ext --inplace

```



### Execute Profiling Script

To trigger the hardware profiling routine and verify numerical precision:

```cmd
python benchmark.py

```


