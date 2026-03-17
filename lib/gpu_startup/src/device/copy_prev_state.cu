#include "gpu_startup/device/copy_prev_state.cuh"
#include <cuda_runtime.h>

__global__ void CopyPrevStateKernel(
    double* positionMatrix, double* velocityMatrix, uint numSatellites, int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;
    int direction = step > 0 ? -1 : +1;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        positionMatrix[idx * 3 * 9 + (4 + step) * 3 + i]
            = positionMatrix[idx * 3 * 9 + (step + 4 + direction) * 3 + i];
        velocityMatrix[idx * 3 * 9 + (4 + step) * 3 + i]
            = velocityMatrix[idx * 3 * 9 + (step + 4 + direction) * 3 + i];
    }
}