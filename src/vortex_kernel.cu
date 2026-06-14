#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 32

// 1. The GPU Kernel (Executes on the silicon)
__global__ void vortex_tiled_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    // Allocate ultra-fast Shared Memory (SRAM) for the tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float acc = 0.0f;

    // Slide the tile across matrix A (columns) and matrix B (rows)
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // Load data from slow Global Memory to fast Shared Memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to ensure the whole tile is loaded before doing math
        __syncthreads();

        // Compute the dot product for this tile entirely in fast SRAM
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        // Synchronize again before loading the next tile
        __syncthreads();
    }

    // Write final accumulated value to Global Memory once
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// 2. The CPU Launcher (Calculates grid sizes and pulls the trigger)
void run_vortex_kernel(float* A, float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    vortex_tiled_matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    
    // Catch any hardware errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}