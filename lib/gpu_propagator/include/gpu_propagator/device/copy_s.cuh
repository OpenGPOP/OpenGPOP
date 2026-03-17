#ifndef GPU_PROPAGATOR_DEVICE_COPY_S_CUH
#define GPU_PROPAGATOR_DEVICE_COPY_S_CUH
#include <cuda_runtime.h>
__global__ void CopySKernel(
    double* smallS, double* bigS, double* smallSMatrix, double* bigSMatrix, uint numSatellites);

#endif // GPU_PROPAGATOR_DEVICE_COPY_S_CUH