import torch
import time
import vortex_engine  # This is your compiled CUDA extension!

def benchmark_matmul(M, K, N, iterations=10):
    print(f"Benchmarking Matrix Size: M={M}, K={K}, N={N}")
    
    # 1. Initialize random matrices directly on the GPU
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C_pytorch = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    C_custom = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    # Warm up the GPU to ensure accurate timing measurements
    for _ in range(3):
        torch.matmul(A, B, out=C_pytorch)
        vortex_engine.matmul(A, B, C_custom)
    torch.cuda.synchronize()

    # 2. Profile Native PyTorch (cuBLAS backend)
    start_pt = time.time()
    for _ in range(iterations):
        torch.matmul(A, B, out=C_pytorch)
    torch.cuda.synchronize()
    end_pt = time.time()
    pytorch_time = (end_pt - start_pt) / iterations

    # 3. Profile Your Custom Tiled Kernel
    start_custom = time.time()
    for _ in range(iterations):
        vortex_engine.matmul(A, B, C_custom)
    torch.cuda.synchronize()
    end_custom = time.time()
    custom_time = (end_custom - start_custom) / iterations

    # 4. Verify Mathematical Accuracy
    # Your kernel must produce identical results to PyTorch to be valid
    is_correct = torch.allclose(C_pytorch, C_custom, atol=1e-3)
    
    print(f"  PyTorch Avg Time: {pytorch_time * 1000:.3f} ms")
    print(f"  Vortex Avg Time:  {custom_time * 1000:.3f} ms")
    print(f"  Math Verification: {'PASSED' if is_correct else 'FAILED'}")
    if custom_time < pytorch_time:
        print(f"  Result: Custom kernel is {pytorch_time / custom_time:.2f}x FASTER than PyTorch!")
    else:
        print(f"  Result: Custom kernel is {custom_time / pytorch_time:.2f}x slower than PyTorch.")
    print("-" * 50)

if __name__ == "__main__":
    # Test standard LLM weight dimension slices
    benchmark_matmul(1024, 1024, 1024)
    benchmark_matmul(2048, 2048, 2048)
    benchmark_matmul(4096, 4096, 4096)