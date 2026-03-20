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

#include "gpu_startup/device/initialize_memory.cuh"
#include <cuda_runtime.h>

__global__ void initialize_memory(double* positions, double* velocities, double* positionMatrix,
    double* velocityMatrix, double* temp_x, double* temp_v, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        temp_x[VECTOR_INDEX(idx, i)] = positions[VECTOR_INDEX(idx, i)];
        temp_v[VECTOR_INDEX(idx, i)] = velocities[VECTOR_INDEX(idx, i)];
        velocityMatrix[MATRIX_INDEX(idx, 4, i)] = velocities[VECTOR_INDEX(idx, i)];
        positionMatrix[MATRIX_INDEX(idx, 4, i)] = positions[VECTOR_INDEX(idx, i)];
    }
}