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

#include "cublas_v2.h"
#include "force_model/acceleration/gravity_source.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "mock_time_points.cuh"
#include "types/Parameters.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>

class GravitySourceTest : public ::testing::Test {
protected:
    cublasHandle_t cublasHandle;
    MockFMTimePoints timePoints { };
    DatasetFilePaths dsFiles = { };

    // Variables for the mock's identity matrix on the device
    double* devIdentityMatrix = nullptr;
    TimePointParametersSlice default_params;

    void SetUp() override
    {
        cublasStatus_t stat = cublasCreate(&cublasHandle);
        ASSERT_EQ(stat, CUBLAS_STATUS_SUCCESS);
        const double hostIdentityMatrix[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
        ASSERT_EQ(cudaMalloc((void**)&devIdentityMatrix, 9 * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(devIdentityMatrix, hostIdentityMatrix, 9 * sizeof(double),
                      cudaMemcpyHostToDevice),
            cudaSuccess);
        default_params.ITRS_to_GCRS = devIdentityMatrix;

        // Whenever GravitySource calls GetParametersDev (for any step), return the identity matrix
        ON_CALL(timePoints, GetParametersDev(testing::_))
            .WillByDefault(testing::Return(default_params));
    }

    void TearDown() override
    {
        // Clean up the mocked device memory to prevent leaks
        if (devIdentityMatrix != nullptr) {
            cudaFree(devIdentityMatrix);
            devIdentityMatrix = nullptr;
        }

        if (cublasHandle != nullptr) {
            cublasDestroy(cublasHandle);
        }

        // Check for uncaught errors
        gpuErrchk(cudaGetLastError());
        cudaDeviceReset();
    }
};

TEST_F(GravitySourceTest, CanConstructAndDestruct)
{
    GravitySource gravity(&cublasHandle, 1, &timePoints, dsFiles);
}

TEST_F(GravitySourceTest, InitializeCompletesSuccessfully)
{
    GravitySource gravity(&cublasHandle, 1, &timePoints, dsFiles);
    gravity.Initialize();
}

double positions[30] = {
    6.778136300000000e+06, 1.000000000000000e+03, 1.000000000000000e+03, // Entity 0
    0.000000000000000e+00, 6.878136300000000e+06, 5.000000000000000e+03, // Entity 1
    1.000000000000000e+03, 2.000000000000000e+03, -6.978136300000000e+06, // Entity 2
    5.217130210248624e+06, 5.217130210248623e+06, 1.000000000000000e+04, // Entity 3
    -7.000000000000000e+06, -5.000000000000000e+06, 3.000000000000000e+06, // Entity 4
    2.000000000000000e+07, 0.000000000000000e+00, 0.000000000000000e+00, // Entity 5
    0.000000000000000e+00, -2.500000000000000e+07, 1.000000000000000e+05, // Entity 6
    6.378136300000000e+06, 6.378136300000000e+06, 6.378136300000000e+06, // Entity 7
    -8.000000000000000e+06, 8.000000000000000e+06, -8.000000000000000e+06, // Entity 8
    -6.578136300000000e+06, 6.578136300000000e+06, -6.578136300000000e+06 // Entity 9
};

// Expected accelerations were calculated using MATLAB scripts published in
// Eckman, R. A., Brown, A. J., Adamo, D. R., & Gottlieb, R. G. (2016). Normalization and
// implementation of three gravitational acceleration models (No. NASA/TP-2016-218604).
// https://ntrs.nasa.gov/citations/20160011252
double expAccels[30] = {
    -8.688512585751166e+00, -1.306014485437521e-03, -1.257696785446421e-03, // Entity 0
    -2.582667093861692e-04, -8.437050803978574e+00, -6.163521917854259e-03, // Entity 1
    -1.030412992305426e-03, -2.285861566359708e-03, 8.163505409516388e+00, // Entity 2
    -5.183755318186486e+00, -5.183875047834164e+00, -9.957411924758089e-03, // Entity 3
    3.691249064843031e+00, 2.636634693149141e+00, -1.584503282402068e+00, // Entity 4
    -9.966668939635662e-01, -2.945383387897605e-07, 1.037052511930428e-07, // Entity 5
    1.569778426358556e-07, 6.378123760414850e-01, -2.551886609172898e-03, // Entity 6
    -1.884989255869376e+00, -1.885012768325866e+00, -1.887034359138477e+00, // Entity 7
    1.198326115108038e+00, -1.198330866500345e+00, 1.199158569758444e+00, // Entity 8
    1.772149453385123e+00, -1.772160848876822e+00, 1.773973126474241e+00 // Entity 9
};

TEST_F(GravitySourceTest, ComputeAccelerationOneByOne)
{
    DatasetFilePaths dfFiles { .gravityModel = "../../data/JGM3.gfc" };
    GravitySource gravity(&cublasHandle, 1, &timePoints, dfFiles);
    gravity.Initialize();

    double *devPosition, *devVelocity, *out;
    size_t bytes = 3 * sizeof(double);
    cudaMalloc(&devPosition, bytes);
    cudaMalloc(&devVelocity, bytes);
    cudaMalloc(&out, bytes);

    int stepNum = 0;
    int outStride = 1;
    double tolerance = 1e-10;

    EXPECT_CALL(timePoints, GetParametersDev(stepNum)).Times(10);
    for (int i = 0; i < 10; ++i) {
        double pos[3] = { positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2] };
        cudaMemcpy(devPosition, pos, bytes, cudaMemcpyHostToDevice);
        cudaMemset(devVelocity, 0, bytes);

        gravity.ComputeAcceleration(devPosition, devVelocity, stepNum, out, outStride);
        double acc[3];
        cudaMemcpy(acc, out, bytes, cudaMemcpyDeviceToHost);

        double expAcc[3] = { expAccels[i * 3 + 0], expAccels[i * 3 + 1], expAccels[i * 3 + 2] };
        EXPECT_NEAR(acc[0], expAcc[0], tolerance);
        EXPECT_NEAR(acc[1], expAcc[1], tolerance);
        EXPECT_NEAR(acc[2], expAcc[2], tolerance);
    }

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cudaFree(out);
}

TEST_F(GravitySourceTest, ComputeAccelerationBatched)
{
    int numEntities = 10;
    DatasetFilePaths dfFiles { .gravityModel = "../../data/JGM3.gfc" };
    GravitySource gravity(&cublasHandle, numEntities, &timePoints, dfFiles);
    gravity.Initialize();

    double *devPosition, *devVelocity, *out;
    size_t bytes = 3 * numEntities * sizeof(double);
    cudaMalloc(&devPosition, bytes);
    cudaMalloc(&devVelocity, bytes);
    cudaMalloc(&out, bytes);

    cudaMemcpy(devPosition, positions, bytes, cudaMemcpyHostToDevice);
    cudaMemset(devVelocity, 0, bytes);
    cudaMemset(out, 0, bytes);

    int stepNum = 0;
    int outStride = 1;

    EXPECT_CALL(timePoints, GetParametersDev(stepNum)).Times(1);
    gravity.ComputeAcceleration(devPosition, devVelocity, stepNum, out, outStride);

    double hostOut[30];
    cudaMemcpy(hostOut, out, bytes, cudaMemcpyDeviceToHost);

    double tolerance = 1e-10;
    for (int i = 0; i < numEntities; ++i) {
        int idx = i * 3;
        EXPECT_NEAR(hostOut[idx + 0], expAccels[idx + 0], tolerance);
        EXPECT_NEAR(hostOut[idx + 1], expAccels[idx + 1], tolerance);
        EXPECT_NEAR(hostOut[idx + 2], expAccels[idx + 2], tolerance);
    }

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cudaFree(out);
}