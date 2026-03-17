//
// Created by igor on 6/19/25.
//

#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

__global__ void MoveStateToMatrixKernel(double* temp_x, double* temp_v, double* positionMatrix,
    double* velocityMatrix, int columnIndex, uint numSatellites);

__global__ void CopyTempToKKernel(double* temp, double* k, int columnIndex, uint numSatellites);

#endif // DEVICE_FUNCTIONS_CUH