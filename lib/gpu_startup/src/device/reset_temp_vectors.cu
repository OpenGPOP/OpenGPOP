#include "gpu_startup/device/reset_temp_vectors.cuh"

__global__ void ResetTempVectorsKernel(double* temp_x, double* temp_v, double* positionMatrix,
    double* velocityMatrix, double* kx, uint numSatellites, int step, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

    int direction = step > 0 ? -1 : 1;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        temp_x[idx * 3 + i] = positionMatrix[idx * 3 * 9 + (4 + step + direction) * 3 + i];
        temp_v[idx * 3 + i] = velocityMatrix[idx * 3 * 9 + (4 + step + direction) * 3 + i];
    }
}