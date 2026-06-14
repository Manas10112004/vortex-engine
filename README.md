# KryoEngine ❄️

A high-performance, low-level AI inference engine framework engineered to optimize LLM execution under constrained hardware profiles (e.g., 16 GB RAM architectures). By bypassing heavy Python abstraction, **KryoEngine** utilizes native Windows kernel APIs for zero-copy memory mapping and optimized bitwise parsing to execute quantized model layers with minimal latency.

## ⚡ Core Architectural Pillars

* **Zero-Copy Virtual Memory Mapping (`mmap`):** Implements raw Windows API virtual address space assignment (`CreateFileMapping`, `MapViewOfFile`). This allows the engine to address large quantized model weights directly on disk, bypassing Python's file systems and protecting host RAM from Out-Of-Memory (OOM) failures.
* **Bitwise 4-Bit (INT4) Quantization Parser:** A compiled C++ subsystem built with high-speed bit-manipulation loops. It unpacks compressed dual-nibble bytes into isolated runtime cache arrays instantly via bitwise masking (`& 0x0F`) and bit-shifting (`>> 4`).
* **Zero-Overhead Python-C++ Binding Pipeline:** Connected using standard C-interface declarations (`extern "C"`) and structured through Python's native `ctypes` library, providing an elegant Python orchestrator layer atop a high-performance native core.

## 🛠️ Repository Layout

```text
KryoEngine/
├── core/
│   ├── src/
│   │   └── kryo_core.cpp      # Native C++ virtual memory & bitwise unpacking logic
│   └── kryo_core.dll          # Optimized compiled system binary (ignored in git)
├── models/                    # Extracted quantized open-source layers (ignored in git)
├── kryo_engine.py             # Python orchestration and ctypes bindings layer
├── test_run.py                # System verification & benchmarking pipeline
└── .gitignore                 # Clear mapping exclusions

🚀 Local Compilation & Quickstart
Prerequisites
A Windows environment equipped with a 64-bit GCC toolchain (g++).

Python 3.12+ (Engine verified and stable on Python 3.14-64).

1. Compile the High-Performance C++ Core
To compile the self-contained native shared library with maximum optimization flags (-O3) and static linking configurations, run:

Bash
g++ -shared -O3 -static -static-libgcc -static-libstdc++ -o core/kryo_core.dll core/src/kryo_core.cpp
2. Execute Verification Subsystems
Launch the local orchestrator to initialize the file mapper, unpack simulated 4-bit weights, and test structural stability:

Bash
python test_run.py