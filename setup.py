import os
import sys

# 1. SET THE ENVIRONMENT VARIABLES FIRST
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"

# 2. IMPORT PYTORCH
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 3. THE MONKEY PATCH (Force PyTorch to ignore the version mismatch)
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

# 4. STANDARD SETUP
setup(
    name='vortex_engine',
    version='1.0',
    author='Manas',
    description='High-Performance Tiled Matrix Multiplication Kernel',
    ext_modules=[
        CUDAExtension(
            name='vortex_engine', 
            sources=[
                'src/bind.cpp',          
                'src/vortex_kernel.cu'   
            ],
            extra_compile_args={
                'cxx': ['-O3'],          
                'nvcc': [
                    '-O3',               
                    '-use_fast_math',    
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)