#include "monte_carlo/monte_carlo.cuh"
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

int main()
{
    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));

    double epochPositionHost[3]
        = { 2859170.4190420364, -6331556.7306755139, 14718.591153407715 }; // [m]
    double epochVelocityHost[3]
        = { -900.03288638945071, -378.6757130492515, 7512.6936320553532 }; // [m/s]

    double objectPositionHost[3] = { 7378137, 0, 0 };
    double objectVelocityHost[3] = { 0, 7350.13862961, 0 };

    PerturbationGenerator perturbationGenerator(epochPositionHost, epochVelocityHost);
    const int numSatellites = 8;
    auto [positions, velocities] = perturbationGenerator.generatePerturbedSatellites(
        numSatellites, 1, objectPositionHost, objectVelocityHost);

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

    params.Area_drag = 55.64;
    params.Area_solar = 88.4;
    params.mass = 8000.0;
    params.CR = 1.0;
    params.CD = 2.7;
    params.n = 70;
    params.m = 70;
    params.Drag = false;

    params.Moon = false;
    params.Sun = false;
    params.Relativity = false;
    params.SRad = false;

    size_t limit;
    gpuErrchk(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
    std::cout << "The heap size limit is: " << limit << std::endl;

    const int numSteps = 50;

    MonteCarlo monteCarlo(
        &handle, &params, positions, velocities, numSatellites, 120.0, numSteps, 10.0);

    monteCarlo.CoarsePropagationPhase();
    monteCarlo.FinePropagationPhase();

    double threshold = 10'000.0; // [m]
    int count = monteCarlo.CollectResults(threshold);
    std::cout << "Number of satellites closer than " << threshold << " m: " << count << std::endl;

    cudaFree(positions);
    cudaFree(velocities);

    cublasDestroy(handle);

    delete[] positionsHost;
    delete[] velocitiesHost;
    return 0;
}
