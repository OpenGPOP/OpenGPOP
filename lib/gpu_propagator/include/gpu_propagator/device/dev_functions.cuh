#ifndef GPU_PROPAGATOR_DEVICE_FINALIZE_PREDICTION_CUH
#define GPU_PROPAGATOR_DEVICE_FINALIZE_PREDICTION_CUH

#include <cuda_runtime.h>

__global__ void CopyStartupStateKernel(double* positions, double* velocities,
    double* positionMatrix, double* velocityMatrix, int stepNum, uint numSatellites);

__global__ void FinalizeCorrectionKernel(double* positions, double* velocities,
    double* accelerationMatrix, double* smallS, double* bigS, double* a, double* b, double timeStep,
    uint numSatellites);

__global__ void MoveColumnToVectorKernel(
    double* matrix, double* vector, int columnIndex, uint numSatellites);

#endif // GPU_PROPAGATOR_DEVICE_FINALIZE_PREDICTION_CUH