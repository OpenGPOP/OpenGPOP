#include "../../../gpu_integrator/include/gpu_integrator/device/finalize_prediction.cuh"
#include "../../../gpu_integrator/include/gpu_integrator/device/move_accelerations.cuh"
#include "gpu_propagator/device/copy_s.cuh"
#include "gpu_propagator/device/update_smalls.cuh"

__global__ void CopyStartupStateKernel(double* positions, double* velocities,
    double* positionMatrix, double* velocityMatrix, int stepNum, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }

    for (int i = 0; i < 3; i++) {
        positions[idx * 3 + i] = positionMatrix[idx * 9 * 3 + stepNum * 3 + i];
        velocities[idx * 3 + i] = velocityMatrix[idx * 9 * 3 + stepNum * 3 + i];
    }
}

__global__ void UpdateSmallSKernel(
    double* smallS, double* accelerationMatrix, double* prevSmallS, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        smallS[idx * 3 + i] = prevSmallS[idx * 3 + i]
            + (accelerationMatrix[idx * 3 * 9 + 3 * 8 + i]
                  + accelerationMatrix[idx * 3 * 9 + 3 * 7 + i])
                / 2;
    }
}

__global__ void FinalizeCorrectionKernel(double* positions, double* velocities,
    double* accelerationMatrix, double* smallS, double* bigS, double* a, double* b, double timeStep,
    uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        velocities[idx * 3 + i] = timeStep
            * (velocities[idx * 3 + i] + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] * b[8 * 10 + 8]
                + smallS[idx * 3 + i]);
        positions[idx * 3 + i] = timeStep * timeStep
            * (positions[idx * 3 + i] + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] * a[8 * 10 + 8]
                + bigS[idx * 3 + i]);
    }
}
