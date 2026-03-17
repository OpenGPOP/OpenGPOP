#include "gpu_startup/device/move_state_to_temp.cuh"

__global__ void MoveStateToTempKernel(double* tempX, double* tempV, double* positionMatrix,
    double* velocityMatrix, int rowIndex, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
#pragma unroll
    for (int i = 0; i < 3; i++) {
        tempX[3 * idx + i] = positionMatrix[3 * 9 * idx + 3 * rowIndex + i];
        tempV[3 * idx + i] = velocityMatrix[3 * 9 * idx + 3 * rowIndex + i];
    }
}
