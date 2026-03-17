#include "force_model/force_model.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "perturbation_generator/perturbation_generator.cuh"

#include <gtest/gtest.h>

class PropagatorTest : public ::testing::Test {
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

TEST_F(PropagatorTest, SameResultsForAllSatellites)
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

    ForceModel force_model(&_cublasHandle, numSatellites, stepSize, &params);

    int numSteps = 10;

    GpuPropagator propagator(devPositions, devVelocities, numSatellites, &_cublasHandle, stepSize,
        &params, force_model, numSteps);
    double hostPositions[numSatellites * 3];
    double hostVelocities[numSatellites * 3];

    for (int i = 0; i < numSteps; i++) {
        propagator.Step(hostPositions, hostVelocities);

        for (int sat = 1; sat < numSatellites; sat++) {
            EXPECT_EQ(hostPositions[sat * 3 + 0], hostPositions[0]) << "Sat num: " << sat;
            EXPECT_EQ(hostPositions[sat * 3 + 1], hostPositions[1]) << "Sat num: " << sat;
            EXPECT_EQ(hostPositions[sat * 3 + 2], hostPositions[2]) << "Sat num: " << sat;

            EXPECT_EQ(hostVelocities[sat * 3 + 0], hostVelocities[0]) << "Sat num: " << sat;
            EXPECT_EQ(hostVelocities[sat * 3 + 1], hostVelocities[1]) << "Sat num: " << sat;
            EXPECT_EQ(hostVelocities[sat * 3 + 2], hostVelocities[2]) << "Sat num: " << sat;
        }
    }
    cudaFree(devPositions);
    cudaFree(devVelocities);
}
