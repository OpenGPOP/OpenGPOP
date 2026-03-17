#include "gpu_propagator/device/copy_s.cuh"

__global__ void CopySKernel(
    double* smallS, double* bigS, double* smallSMatrix, double* bigSMatrix, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        bigS[idx * 3 + i] = bigSMatrix[idx * 3 * 9 + 3 * 8 + i];
        smallS[idx * 3 + i] = smallSMatrix[idx * 3 * 9 + 3 * 8 + i];
    }
}