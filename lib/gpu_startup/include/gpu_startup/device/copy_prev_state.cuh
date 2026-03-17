#ifndef GPU_STARTUP_DEVICE_COPY_PREV_STATE_CUH
#define GPU_STARTUP_DEVICE_COPY_PREV_STATE_CUH
#include <cuda_runtime.h>
__global__ void CopyPrevStateKernel(
    double* positionMatrix, double* velocityMatrix, uint numSatellites, int step);

#endif // GPU_STARTUP_DEVICE_COPY_PREV_STATE_CUH