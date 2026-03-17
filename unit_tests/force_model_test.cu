#include "force_model/force_model.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "perturbation_generator/perturbation_generator.cuh"

#include <gtest/gtest.h>

class ForceModelTest : public ::testing::Test {
protected:
    void SetUp() { cublasCreate(&_cublasHandle); }

    void TearDown() override
    {
        cublasDestroy(_cublasHandle);
        gpuErrchk(cudaGetLastError());
        cudaDeviceReset();
    }

    cublasHandle_t _cublasHandle;
};

TEST_F(ForceModelTest, SameResultsForAllSatellites)
{
    double epochPositionHost[3] = { -3884872, -5749111, 14723 }; // [m]
    double epochVelocityHost[3] = {
        -795.304,
        573.772,
        7518.179,
    }; // [m/s]
    PerturbationGenerator perturbationGenerator(epochPositionHost, epochVelocityHost);
    const int numSatellites = 8;
    const double stepSize = 30.0;
    auto [devPositions, devVelocities]
        = perturbationGenerator.generatePerturbedSatellites(numSatellites, 0.0, nullptr, nullptr);

    double* devAccelerationMatrix;
    gpuErrchk(cudaMalloc(&devAccelerationMatrix, sizeof(double) * numSatellites * 9 * 3));

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

    ForceModel forceModel(&_cublasHandle, numSatellites, stepSize, &params);
    int numSteps = 10;
    forceModel.PreInitialize();
    forceModel.InitializeGeneration(4, 4);
    forceModel.InitializePropagation(4, numSteps);

    double hostAccelerationMatrix[3 * 9 * numSatellites];
    forceModel.ComputePropagationAcceleration(
        devPositions, devVelocities, devAccelerationMatrix, 0, 0);

    gpuErrchk(cudaMemcpy(hostAccelerationMatrix, devAccelerationMatrix,
        sizeof(double) * 3 * 9 * numSatellites, cudaMemcpyDeviceToHost));

    for (int sat = 1; sat < numSatellites; sat++) {
        EXPECT_EQ(hostAccelerationMatrix[3 * 9 * sat + 0], hostAccelerationMatrix[0])
            << "sat num: " << sat;
        EXPECT_EQ(hostAccelerationMatrix[3 * 9 * sat + 1], hostAccelerationMatrix[1])
            << "sat num: " << sat;
        EXPECT_EQ(hostAccelerationMatrix[3 * 9 * sat + 2], hostAccelerationMatrix[2])
            << "sat num: " << sat;
    }

    cudaFree(devPositions);
    cudaFree(devVelocities);
    cudaFree(devAccelerationMatrix);
}
