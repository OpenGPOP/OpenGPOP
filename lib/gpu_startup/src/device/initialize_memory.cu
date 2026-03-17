#include "gpu_startup/device/initialize_memory.cuh"
#include <cuda_runtime.h>

__global__ void initialize_memory(double* positions, double* velocities, double* positionMatrix,
    double* velocityMatrix, double* temp_x, double* temp_v, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        temp_x[VECTOR_INDEX(idx, i)] = positions[VECTOR_INDEX(idx, i)];
        temp_v[VECTOR_INDEX(idx, i)] = velocities[VECTOR_INDEX(idx, i)];
        velocityMatrix[MATRIX_INDEX(idx, 4, i)] = velocities[VECTOR_INDEX(idx, i)];
        positionMatrix[MATRIX_INDEX(idx, 4, i)] = positions[VECTOR_INDEX(idx, i)];
    }
}