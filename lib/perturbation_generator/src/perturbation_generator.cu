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

#include "gpu_utils/gpu_utils.cuh"
#include "perturbation_generator/perturbation_generator.cuh"
#include <curand_kernel.h>

__global__ void createSatellitesKernel(int numSatellites, double* epochPosition,
    double* epochVelocity, double* positions, double* velocities, double perturbationStrength,
    uint seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numSatellites)
        return;
    curandState state;
    curand_init(seed, idx, 0, &state);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        positions[idx * 3 + i] = epochPosition[i];
        velocities[idx * 3 + i]
            = epochVelocity[i] + curand_normal_double(&state) * perturbationStrength;
    }
}

std::tuple<double*, double*> PerturbationGenerator::generatePerturbedSatellites(
    int numSatellites, double perturbationStrength, double* objectPosition, double* objectVelocity)
{
    double* epochPosition;
    double* epochVelocity;
    double* positions;
    double* velocities;

    size_t allocSize = numSatellites * 3 * sizeof(double);
    if (objectPosition != nullptr && objectVelocity != nullptr) {
        allocSize += 3 * sizeof(double);
    }

    gpuErrchk(cudaMalloc((void**)&epochPosition, 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&epochVelocity, 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&positions, allocSize));
    gpuErrchk(cudaMalloc((void**)&velocities, allocSize));

    gpuErrchk(
        cudaMemcpy(epochPosition, epochPositionHost, 3 * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(epochVelocity, epochVelocityHost, 3 * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (numSatellites + blockSize - 1) / blockSize;

    createSatellitesKernel<<<numBlocks, blockSize>>>(numSatellites, epochPosition, epochVelocity,
        positions, velocities, perturbationStrength, 1234);

    cudaFree(epochPosition);
    cudaFree(epochVelocity);

    if (objectPosition != nullptr && objectVelocity != nullptr) {
        gpuErrchk(cudaMemcpy(positions + numSatellites * 3, objectPosition, 3 * sizeof(double),
            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(velocities + numSatellites * 3, objectVelocity, 3 * sizeof(double),
            cudaMemcpyHostToDevice));
    }

    return { positions, velocities };
}
