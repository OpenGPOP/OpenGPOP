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
#include "gpu_propagator/device/copy_s.cuh"
#include "gpu_propagator/device/update_smalls.cuh"

__global__ void CopyStartupStateKernel(double* positions, double* velocities,
    double* positionMatrix, double* velocityMatrix, int stepNum, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }

    for (int i = 0; i < 3; i++) {
        positions[idx * 3 + i] = positionMatrix[idx * 9 * 3 + stepNum * 3 + i];
        velocities[idx * 3 + i] = velocityMatrix[idx * 9 * 3 + stepNum * 3 + i];
    }
}

__global__ void UpdateSmallSKernel(
    double* smallS, double* accelerationMatrix, double* prevSmallS, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        smallS[idx * 3 + i] = prevSmallS[idx * 3 + i]
            + (accelerationMatrix[idx * 3 * 9 + 3 * 8 + i]
                  + accelerationMatrix[idx * 3 * 9 + 3 * 7 + i])
                / 2;
    }
}

__global__ void FinalizeCorrectionKernel(double* positions, double* velocities,
    double* accelerationMatrix, double* smallS, double* bigS, double* a, double* b, double timeStep,
    uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        velocities[idx * 3 + i] = timeStep
            * (velocities[idx * 3 + i] + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] * b[8 * 10 + 8]
                + smallS[idx * 3 + i]);
        positions[idx * 3 + i] = timeStep * timeStep
            * (positions[idx * 3 + i] + accelerationMatrix[idx * 3 * 9 + 3 * 8 + i] * a[8 * 10 + 8]
                + bigS[idx * 3 + i]);
    }
}

__global__ void MoveColumnToVectorKernel(
    double* matrix, double* vector, int columnIndex, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        vector[3 * idx + i] = matrix[3 * 9 * idx + columnIndex * 3 + i];
    }
}
