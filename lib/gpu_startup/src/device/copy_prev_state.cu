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

#include "gpu_startup/device/copy_prev_state.cuh"
#include <cuda_runtime.h>

__global__ void CopyPrevStateKernel(
    double* positionMatrix, double* velocityMatrix, uint numSatellites, int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;
    int direction = step > 0 ? -1 : +1;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        positionMatrix[idx * 3 * 9 + (4 + step) * 3 + i]
            = positionMatrix[idx * 3 * 9 + (step + 4 + direction) * 3 + i];
        velocityMatrix[idx * 3 * 9 + (4 + step) * 3 + i]
            = velocityMatrix[idx * 3 * 9 + (step + 4 + direction) * 3 + i];
    }
}