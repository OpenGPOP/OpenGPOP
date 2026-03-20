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

#include "gpu_integrator/device/finalize_prediction.cuh"
#include "gpu_integrator/device/move_accelerations.cuh"

__global__ void MoveAccelerationsKernel(double* accelerationMatrix, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
    for (int n = 0; n < 8; n++) {
        for (int i = 0; i < 3; i++) {
            accelerationMatrix[idx * 9 * 3 + n * 3 + i]
                = accelerationMatrix[idx * 9 * 3 + (n + 1) * 3 + i];
        }
    }
}

__global__ void FinalizePredictionKernel(double* positions, double* velocities,
    double* accelerationMatrix, double* smallS, double* bigS, uint numSatellites, double timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        bigS[idx * 3 + i] += smallS[idx * 3 + i] + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] / 2;
        velocities[idx * 3 + i] = (velocities[idx * 3 + i] + smallS[idx * 3 + i]
                                      + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] / 2)
            * timeStep;
        positions[idx * 3 + i] = (positions[idx * 3 + i] + bigS[idx * 3 + i]) * timeStep * timeStep;
    }
}
