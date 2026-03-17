#include "gpu_startup/device/calculate_all_s.cuh"

__global__ void CalculateAllSKernel(
    double* bigS, double* smallS, double* accelerationMatrix, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

    for (int i = 1; i <= 4; i++) {
        for (int j = 0; j < 3; j++) {
            smallS[idx * 3 * 9 + 3 * (4 + i) + j] = smallS[idx * 3 * 9 + 3 * (4 + i - 1) + j]
                + (accelerationMatrix[idx * 3 * 9 + 3 * (4 + i - 1) + j]
                      + accelerationMatrix[idx * 3 * 9 + 3 * (4 + i) + j])
                    / 2;

            smallS[idx * 3 * 9 + 3 * (4 - i) + j] = smallS[idx * 3 * 9 + 3 * (4 - i + 1) + j]
                - (accelerationMatrix[idx * 3 * 9 + 3 * (4 - i + 1) + j]
                      + accelerationMatrix[idx * 3 * 9 + 3 * (4 - i) + j])
                    / 2;

            bigS[idx * 3 * 9 + 3 * (4 + i) + j] = smallS[idx * 3 * 9 + 3 * (4 + i - 1) + j]
                + bigS[idx * 3 * 9 + 3 * (4 + i - 1) + j]
                + accelerationMatrix[idx * 3 * 9 + 3 * (4 + i - 1) + j] / 2;

            bigS[idx * 3 * 9 + 3 * (4 - i) + j] = -smallS[idx * 3 * 9 + 3 * (4 - i + 1) + j]
                + bigS[idx * 3 * 9 + 3 * (4 - i + 1) + j]
                + accelerationMatrix[idx * 3 * 9 + 3 * (4 - i + 1) + j] / 2;
        }
    }
}