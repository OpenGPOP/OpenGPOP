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

#include "gpu_startup/device/device_functions.cuh"

__global__ void MoveStateToMatrixKernel(double* temp_x, double* temp_v, double* positionMatrix,
    double* velocityMatrix, int columnIndex, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

#pragma unroll
    for (int i = 0; i < 3; i++) {
        positionMatrix[3 * 9 * idx + 3 * columnIndex + i] = temp_x[3 * idx + i];
        velocityMatrix[3 * 9 * idx + 3 * columnIndex + i] = temp_v[3 * idx + i];
    }
}
__global__ void CopyTempToKKernel(double* temp, double* k, int columnIndex, uint numSatellites)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSatellites)
        return;

#pragma unroll
    for (int i = 0; i < 3; i++) {
        k[3 * 10 * idx + 3 * columnIndex + i] = temp[3 * idx + i];
    }
}
