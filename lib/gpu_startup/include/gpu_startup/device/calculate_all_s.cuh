#ifndef CALCULATE_ALL_S_CUH
#define CALCULATE_ALL_S_CUH

#include <cuda_runtime.h>

__global__ void CalculateAllSKernel(
    double* bigS, double* smallS, double* accelerationMatrix, uint numSatellites);
#endif // CALCULATE_ALL_S_CUH