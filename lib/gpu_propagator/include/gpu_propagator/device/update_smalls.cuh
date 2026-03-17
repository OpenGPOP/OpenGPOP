#ifndef UPDATE_SMALLS_CUH
#define UPDATE_SMALLS_CUH

#include <cuda_runtime.h>
__global__ void UpdateSmallSKernel(
    double* smallS, double* accelerationMatrix, double* prevSmallS, uint numSatellites);

#endif // UPDATE_SMALLS_CUH