#pragma once
#include <cuda_runtime.h>

__global__ void MoveStateToTempKernel(double* tempX, double* tempV, double* positionMatrix,
    double* velocityMatrix, int rowIndex, uint numSatellites);
