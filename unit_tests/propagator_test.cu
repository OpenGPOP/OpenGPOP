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
    // Changing to EIGEN returns errors that can be traced back to thrust::reduce. Likely due
    // to a different N, summation between threads is split in a different way, and since float
    // addition is not associative, this results in GJKH matrices turining out slightly differently
    // for each satellite.
    // TODO: We might address this further down the line, since current use of thrust seeems to be
    // suboptimal
    params.dsFiles.gravityModel = "../../data/JGM3.gfc";

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
