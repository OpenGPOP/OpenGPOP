#ifndef FINZALIZE_PREDICTION_CUH
#define FINZALIZE_PREDICTION_CUH

#include <cuda_runtime.h>
__global__ void FinalizePredictionKernel(double* positions, double* velocities,
    double* accelerationMatrix, double* smallS, double* bigS, uint numSatellites, double timeStep);

#endif // FINZALIZE_PREDICTION_CUH
