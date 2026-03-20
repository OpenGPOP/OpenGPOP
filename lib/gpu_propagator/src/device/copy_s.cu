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

#include "gpu_propagator/device/copy_s.cuh"

__global__ void CopySKernel(
    double* smallS, double* bigS, double* smallSMatrix, double* bigSMatrix, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        bigS[idx * 3 + i] = bigSMatrix[idx * 3 * 9 + 3 * 8 + i];
        smallS[idx * 3 + i] = smallSMatrix[idx * 3 * 9 + 3 * 8 + i];
    }
}