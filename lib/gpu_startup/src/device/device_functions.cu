#include "gpu_startup/device/device_functions.cuh"

__global__ void MoveStateToMatrixKernel(double* temp_x, double* temp_v, double* positionMatrix,
    double* velocityMatrix, int columnIndex, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

#pragma unroll
    for (int i = 0; i < 3; i++) {
        positionMatrix[3 * 9 * idx + 3 * columnIndex + i] = temp_x[3 * idx + i];
        velocityMatrix[3 * 9 * idx + 3 * columnIndex + i] = temp_v[3 * idx + i];
    }
}
__global__ void CopyTempToKKernel(double* temp, double* k, int columnIndex, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

#pragma unroll
    for (int i = 0; i < 3; i++) {
        k[3 * 10 * idx + 3 * columnIndex + i] = temp[3 * idx + i];
    }
}
