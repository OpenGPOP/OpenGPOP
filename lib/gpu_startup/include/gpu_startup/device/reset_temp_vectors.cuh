#ifndef RESET_TEMP_VECTORS_CUH
#define RESET_TEMP_VECTORS_CUH

#include <cuda_runtime.h>

__global__ void ResetTempVectorsKernel(double* temp_x, double* temp_v, double* positionMatrix,
    double* velocityMatrix, double* kx, uint numSatellites, int step, int k);

#endif