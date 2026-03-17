#include "gpu_integrator/device/finalize_prediction.cuh"
#include "gpu_integrator/device/move_accelerations.cuh"

__global__ void MoveAccelerationsKernel(double* accelerationMatrix, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
    for (int n = 0; n < 8; n++) {
        for (int i = 0; i < 3; i++) {
            accelerationMatrix[idx * 9 * 3 + n * 3 + i]
                = accelerationMatrix[idx * 9 * 3 + (n + 1) * 3 + i];
        }
    }
}

__global__ void FinalizePredictionKernel(double* positions, double* velocities,
    double* accelerationMatrix, double* smallS, double* bigS, uint numSatellites, double timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        bigS[idx * 3 + i] += smallS[idx * 3 + i] + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] / 2;
        velocities[idx * 3 + i] = (velocities[idx * 3 + i] + smallS[idx * 3 + i]
                                      + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] / 2)
            * timeStep;
        positions[idx * 3 + i] = (positions[idx * 3 + i] + bigS[idx * 3 + i]) * timeStep * timeStep;
    }
}
