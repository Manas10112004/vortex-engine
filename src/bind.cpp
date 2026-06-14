#include <torch/extension.h>

// Forward declaration of the C++ launcher function in vortex_kernel.cu
void run_vortex_kernel(float* A, float* B, float* C, int M, int K, int N);

// The Python-facing wrapper function
void matmul_wrapper(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 1. Ensure the tensors are actually on the GPU and are contiguous in memory
    TORCH_CHECK(A.is_cuda(), "Matrix A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Matrix B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "Matrix C must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Matrix A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Matrix B must be contiguous");

    // 2. Extract the dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // 3. Extract the raw C++ pointers from the PyTorch Tensors
    float* ptr_A = A.data_ptr<float>();
    float* ptr_B = B.data_ptr<float>();
    float* ptr_C = C.data_ptr<float>();

    // 4. Launch the CUDA execution
    run_vortex_kernel(ptr_A, ptr_B, ptr_C, M, K, N);
}

// 5. The PyBind11 Registry (Exposes the function to Python)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_wrapper, "Vortex Engine Tiled MatMul");
}