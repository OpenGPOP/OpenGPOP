#ifndef GPU_STARTUP_DEVICE_INITIALIZE_MEMORY_CUH
#define GPU_STARTUP_DEVICE_INITIALIZE_MEMORY_CUH

#include <cuda_runtime.h>

#define MATRIX_INDEX(sat, row, col) ((sat) * 3 * 9 + (row) * 3 + (col))
#define VECTOR_INDEX(sat, col) ((sat) * 3 + (col))
#define KMATRIX_INDEX(sat, row, col) ((sat) * 3 * 10 + (row) * 3 + (col))

__global__ void initialize_memory(double* positions, double* velocities,
    double* positionMatrix, double* velocityMatrix, double* temp_x, double* temp_v,
    uint numSatellites);

#endif // GPU_STARTUP_DEVICE_INITIALIZE_MEMORY_CUH