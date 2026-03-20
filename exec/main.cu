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

#include "closest_approach/closest_approach.cuh"
#include "force_model/force_model.cuh"
#include "perturbation_generator/perturbation_generator.cuh"
#include <iostream>

void PrintSatellites(double* positionsHost, double* velocitiesHost, int numSatellites)
{
    for (int i = 0; i < numSatellites; i += 1) {
        std::cout << "Satellite " << i << ": Position: (" << positionsHost[i * 3] << ", "
                  << positionsHost[i * 3 + 1] << ", " << positionsHost[i * 3 + 2]
                  << "), Velocity: (" << velocitiesHost[i * 3] << ", " << velocitiesHost[i * 3 + 1]
                  << ", " << velocitiesHost[i * 3 + 2] << ")\n";
    }
}

void doWork()
{
    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));

    double epochPositionHost[3] = { -3884872, -5749111, 14723 }; // [m]
    double epochVelocityHost[3] = {
        -795.304,
        573.772,
        7518.179,
    }; // [m/s]

    double objectPositionHost[3] = { 7378137, 0, 0 };
    double objectVelocityHost[3] = { 0, 7350.13862961, 0 };

    PerturbationGenerator perturbationGenerator(epochPositionHost, epochVelocityHost);
    const int numSatellites = 8;
    auto [positions, velocities] = perturbationGenerator.generatePerturbedSatellites(
        numSatellites, 0.1, objectPositionHost, objectVelocityHost);

    double* positionsHost = new double[numSatellites * 3];
    double* velocitiesHost = new double[numSatellites * 3];
    cudaMemcpy(
        positionsHost, positions, numSatellites * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        velocitiesHost, velocities, numSatellites * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    PrintSatellites(positionsHost, velocitiesHost, numSatellites);

    FullPerturbationParameters params;
    params.StartTimeUTC.Year = 2015;
    params.StartTimeUTC.Month = 5;
    params.StartTimeUTC.Day = 4;
    params.StartTimeUTC.Hour = 22;
    params.StartTimeUTC.Min = 22;
    params.StartTimeUTC.Sec = 2;

    size_t limit;
    gpuErrchk(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
    std::cout << "The heap size limit is: " << limit << std::endl;

    const int numSteps = 10;
    const double stepSize = 30.0;

    ForceModel forceModel(&handle, numSatellites, 1.0, &params);
    ClosestApproach closestApproach(&handle, &params, forceModel, stepSize);

    // initialize vectors with data
    std::vector<double> positionsVec(positionsHost, positionsHost + numSatellites * 3);
    std::vector<double> velocitiesVec(velocitiesHost, velocitiesHost + numSatellites * 3);
    std::vector<unsigned int> setIndicesVec(numSatellites);
    for (int i = 0; i < numSatellites; ++i) {
        setIndicesVec[i] = i / 4; // Group satellites in sets of 4
    }
    closestApproach.Initialize(positionsVec, velocitiesVec, setIndicesVec);

    closestApproach.Propagate(numSteps, true);
    ClosestApproachData data;
    closestApproach.FindClosest(data);

    // Iterate through all possible pairs
    for (int i = 0; i < numSatellites; ++i) {
        for (int j = i + 1; j < numSatellites; ++j) {
            // Only print pairs that belong to different sets
            if (setIndicesVec[i] != setIndicesVec[j]) {
                std::cout << "Pair " << i << "-" << j << ": Distance = " << data.Get(i, j).distance
                          << " m, Time = " << data.Get(i, j).time << " s\n";
            }
        }
    }

    cudaFree(positions);
    cudaFree(velocities);
    delete[] positionsHost;
    delete[] velocitiesHost;
    cublasDestroy(handle);
}

int main()
{
    // Separate function to ensure proper cleanup of resources (in destructors) before exiting
    doWork();

    // Check for uncaught errors
    gpuErrchk(cudaGetLastError());
    cudaDeviceReset();

    return 0;
}
