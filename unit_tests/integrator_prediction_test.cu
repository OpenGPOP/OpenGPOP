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
#include "gpu_integrator/prediction.cuh"
#include "gpu_startup/gpu_startup.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "include/gpu_test_helpers.cuh"
#include "include/mock_force_model.cuh"

#include <gtest/gtest.h>

TEST(Prediction, HandCalculation)
{
    double epochPosition[3] = { 0.0, 0.0, 0.0 };
    double epochVelocity[3] = { 0.0, 0.0, 0.0 };
    double accelerationMatrix[27]
        = { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9 };
    double bigS[3] = { 1, 1, 1 };
    double smallS[3] = { 2, 2, 2 };
    int numSatellites = 1;
    double stepSize = 0.1;
    Poly7Accel accelFun;

    double* devPosition;
    double* devVelocity;
    double* devSmallSVector;
    double* devBigSVector;
    double* devAccelerationMatrix;

    gpuErrchk(cudaMalloc((void**)&devPosition, numSatellites * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devVelocity, numSatellites * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devAccelerationMatrix, numSatellites * 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devSmallSVector, numSatellites * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devBigSVector, numSatellites * 3 * sizeof(double)));

    cudaMemcpy(devPosition, epochPosition, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, epochVelocity, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devBigSVector, bigS, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devSmallSVector, smallS, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(
        devAccelerationMatrix, accelerationMatrix, 3 * 9 * sizeof(double), cudaMemcpyHostToDevice);
    double* devA;
    double* devB;
    gpu_constants::createCoefficients(&devA, &devB);

    MockForceModel forceModel(numSatellites, stepSize, accelFun);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    forceModel.InitializePropagation(4, 10);

    Predictor predictor(forceModel, &cublas, numSatellites, stepSize);
    predictor.SetPointers(devPosition, devVelocity, devAccelerationMatrix, devA, devB,
        devSmallSVector, devBigSVector);

    predictor.Predict(5);

    double expectedPosition[3] = {
        0.083333333333333,
        0.083333333333333,
        0.083333333333333,
    };
    double expectedVelocity[3] = { 1.141666666666669, 1.141666666666669, 1.141666666666669 };

    double calculatedPosition[3];
    double calculatedVelocity[3];
    cudaMemcpy(calculatedPosition, devPosition, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(expectedPosition[0], calculatedPosition[0]);
    EXPECT_FLOAT_EQ(expectedPosition[1], calculatedPosition[1]);
    EXPECT_FLOAT_EQ(expectedPosition[2], calculatedPosition[2]);

    cudaMemcpy(calculatedVelocity, devVelocity, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_DOUBLE_EQ(expectedVelocity[0], calculatedVelocity[0]);
    EXPECT_DOUBLE_EQ(expectedVelocity[1], calculatedVelocity[1]);
    EXPECT_DOUBLE_EQ(expectedVelocity[2], calculatedVelocity[2]);

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devBigSVector);
    cudaFree(devSmallSVector);
    cudaFree(devAccelerationMatrix);
    cublasDestroy(cublas);

    gpuErrchk(cudaGetLastError());
    cudaDeviceReset();
}
