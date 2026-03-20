// Open GPU-Parallelised Orbit Propagator (OpenGPOP)
// Copyright (C) 2026 Igor Januszkiewicz, Michał Kiedrzyński
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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