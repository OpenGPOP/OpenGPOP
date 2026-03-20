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

#include "gpu_constants/gpu_constants.cuh"
#include "gpu_startup/gpu_startup.cuh"
#include "gpu_test_helpers.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "mock_force_model.cuh"

#include <gtest/gtest.h>

class RK8Test : public ::testing::Test {
protected:
    void TearDown() override
    {
        gpuErrchk(cudaGetLastError());
        cudaDeviceReset();
    }
};

TEST_F(RK8Test, Stationary)
{
    int numSatellites = 1;
    double stepSize = 1.0;
    double hostPosition[3] = { 1.0, 1.0, 1.0 };
    double hostVelocity[3] = { 0.0, 0.0, 0.0 };

    double* devPositionMatrix;
    double* devVelocityMatrix;
    double* devTempX;
    double* devTempV;

    gpuErrchk(cudaMalloc((void**)&devPositionMatrix, numSatellites * 3 * 9 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devVelocityMatrix, numSatellites * 3 * 9 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devTempX, numSatellites * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devTempV, numSatellites * 3 * sizeof(double)));

    cudaMemcpy(devPositionMatrix + 3 * 4, hostPosition, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocityMatrix + 3 * 4, hostVelocity, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devTempX, hostPosition, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devTempV, hostVelocity, 3 * sizeof(double), cudaMemcpyHostToDevice);

    NoAcceleration noAcceleration;
    MockForceModel forceModel(numSatellites, stepSize, noAcceleration);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    RK8 rk8(numSatellites, devPositionMatrix, devVelocityMatrix, devTempX, devTempV, forceModel,
        stepSize, &cublas);
    for (int i = 1; i <= 4; i++) {
        rk8.step(i);
        rk8.step(-i);
    }

    double expectedPositions[9 * 3] = {
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    };
    double expectedVelocity[9 * 3] = {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    };

    double calculatedPosition[9 * 3];
    double calculatedVelocity[9 * 3];

    cudaMemcpy(
        calculatedPosition, devPositionMatrix, 9 * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        calculatedVelocity, devVelocityMatrix, 9 * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9 * 3; i++) {
        EXPECT_EQ(expectedPositions[i], calculatedPosition[i]) << "Mismatch at " << i;
        EXPECT_EQ(expectedVelocity[i], calculatedVelocity[i]) << "Mismatch at " << i;
    }

    cudaFree(devPositionMatrix);
    cudaFree(devVelocityMatrix);
    cudaFree(devTempX);
    cudaFree(devTempV);
    cublasDestroy(cublas);
}

template <typename accelerationFunction>
std::vector<double> Propagate(accelerationFunction accelFun, double stepSize)
{
    int numSatellites = 1;
    double hostPosition[3] = { 0.0, 0.0, 0.0 };
    double hostVelocity[3] = { 0.0, 0.0, 0.0 };

    double* devPositionMatrix;
    double* devVelocityMatrix;
    double* devTempX;
    double* devTempV;

    gpuErrchk(cudaMalloc((void**)&devPositionMatrix, numSatellites * 3 * 9 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devVelocityMatrix, numSatellites * 3 * 9 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devTempX, numSatellites * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devTempV, numSatellites * 3 * sizeof(double)));

    cudaMemcpy(devPositionMatrix + 3 * 4, hostPosition, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocityMatrix + 3 * 4, hostVelocity, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devTempX, hostPosition, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devTempV, hostVelocity, 3 * sizeof(double), cudaMemcpyHostToDevice);

    MockForceModel forceModel(numSatellites, stepSize, accelFun);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    RK8 rk8(numSatellites, devPositionMatrix, devVelocityMatrix, devTempX, devTempV, forceModel,
        stepSize, &cublas);
    for (int i = 1; i <= 4; i++) {
        rk8.step(i);
        rk8.step(-i);
    }

    double calculatedPosition[9 * 3];

    cudaMemcpy(
        calculatedPosition, devPositionMatrix, 9 * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(devPositionMatrix);
    cudaFree(devVelocityMatrix);
    cudaFree(devTempX);
    cudaFree(devTempV);
    cublasDestroy(cublas);
    std::vector<double> positions(9 * 3);
    for (size_t i = 0; i < positions.size(); i++) {
        positions[i] = calculatedPosition[i];
    }
    return positions;
}

template <typename accelerationFunction, typename actualFunction>
void testShouldBeExact(accelerationFunction accelFun, actualFunction actualFun, double stepSize)
{
    std::vector positions = Propagate<accelerationFunction>(accelFun, stepSize);
    std::vector<double> expectedPositions(3 * 9);
    for (int i = -4; i <= 4; i++) {
        expectedPositions[3 * (4 + i) + 0] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 1] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 2] = actualFun(i * stepSize);
    }

    for (size_t i = 0; i < expectedPositions.size(); i++) {
        EXPECT_DOUBLE_EQ(expectedPositions[i], positions[i]);
    }
}

template <typename accelerationFunction, typename actualFunction>
void testShouldNotBeExact(accelerationFunction accelFun, actualFunction actualFun, double stepSize)
{
    std::vector positions = Propagate<accelerationFunction>(accelFun, stepSize);
    std::vector<double> expectedPositions(3 * 9);
    for (int i = -4; i <= 4; i++) {
        expectedPositions[3 * (4 + i) + 0] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 1] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 2] = actualFun(i * stepSize);
    }

    bool atLeastOneWrong = false;
    for (size_t i = 0; i < expectedPositions.size(); i++) {
        double diff = abs(expectedPositions[i] - positions[i]);
        if (diff > 1e-10) {
            atLeastOneWrong = true;
        }
    }
    EXPECT_TRUE(atLeastOneWrong);
}

TEST_F(RK8Test, stationary2)
{
    NoAcceleration noAccel;
    ZeroPosition zeroPosition;

    testShouldBeExact<NoAcceleration, ZeroPosition>(noAccel, zeroPosition, 1.0);
}

TEST_F(RK8Test, poly7)
{
    Poly7Accel accel;
    Exact7Position exactPosition;

    testShouldBeExact(accel, exactPosition, 1.0);
}

TEST_F(RK8Test, poly8)
{
    Poly8Accel accel;
    Exact8Position exactPosition;
    testShouldBeExact<Poly8Accel, Exact8Position>(accel, exactPosition, 1.0);
}

TEST_F(RK8Test, poly9_ShouldNotBeExact)
{
    Poly9Accel accel;
    Exact9Position exactPosition;
    testShouldNotBeExact(accel, exactPosition, 1.0);
}