#include "gpu_constants/gpu_constants.cuh"
#include "gpu_propagator/device/dev_functions.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "gpu_test_helpers.cuh"
#include "mock_force_model.cuh"

#include <gtest/gtest.h>

class IntegratorTest : public ::testing::Test {
protected:
    void TearDown() override
    {
        gpuErrchk(cudaGetLastError());
        cudaDeviceReset();
    }
};

TEST_F(IntegratorTest, MockForceModel)
{
    AccelerationFunction accelerationFunction;

    MockForceModel forceModel(1, 1, accelerationFunction);
    forceModel.PreInitialize();
    forceModel.InitializeGeneration(4, 4);
    double *devPosition, *devVelocity, *devKMatrix;
    double hostKMatrix[10 * 3];
    cudaMalloc((void**)&devPosition, 3 * 9 * sizeof(double));
    cudaMalloc((void**)&devVelocity, 3 * 9 * sizeof(double));
    cudaMalloc((void**)&devKMatrix, 3 * 10 * sizeof(double));
    forceModel.ComputeGenerationAcceleration(devPosition, devVelocity, devKMatrix, 1, 0);
    cudaMemcpy(hostKMatrix, devKMatrix, 3 * 9 * sizeof(double), cudaMemcpyDeviceToHost);

    EXPECT_EQ(hostKMatrix[0], 1);
    EXPECT_EQ(hostKMatrix[1], 2);
    EXPECT_EQ(hostKMatrix[2], 3);

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cudaFree(devKMatrix);
}

TEST_F(IntegratorTest, Stationary)
{
    NoAcceleration acceleration;
    MockForceModel forceModel(1, 1, acceleration);

    double hostPositions[6] = { 0, 0, 0, 1, 1, 1 };
    double hostVelocities[6] = { 0, 0, 0, 0, 0, 0 };
    double hostEpochPostions[6] = { 0, 0, 0, 1, 1, 1 };
    double hostEpochVelocities[6] = { 0, 0, 0, 0, 0, 0 };
    double *devPosition, *devVelocity;
    cudaMalloc((void**)&devPosition, 2 * 3 * sizeof(double));
    cudaMalloc((void**)&devVelocity, 2 * 3 * sizeof(double));
    cudaMemcpy(devPosition, hostPositions, 2 * 3 * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    GpuPropagator propagator(devPosition, devVelocity, 2, &cublas, 1, nullptr, forceModel, 10);
    for (int i = 0; i < 10; i++) {
        propagator.Step(hostPositions, hostVelocities);
        for (int j = 0; j < 6; j++) {
            EXPECT_EQ(hostPositions[j], hostEpochPostions[j]);
            EXPECT_EQ(hostVelocities[j], hostEpochVelocities[j]);
        }
    }

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cublasDestroy(cublas);
}

TEST_F(IntegratorTest, ConstantSpeed)
{
    NoAcceleration acceleration;
    MockForceModel forceModel(1, 1, acceleration);

    double hostPositions[6] = { 0, 0, 0, 1, 1, 1 };
    double hostVelocities[6] = { 1, 1, 1, 1, 1, 1 };
    double hostEpochPostions[6] = { 0, 0, 0, 1, 1, 1 };
    double hostEpochVelocities[6] = { 1, 1, 1, 1, 1, 1 };
    double *devPosition, *devVelocity;
    cudaMalloc((void**)&devPosition, 2 * 3 * sizeof(double));
    cudaMalloc((void**)&devVelocity, 2 * 3 * sizeof(double));
    cudaMemcpy(devPosition, hostPositions, 2 * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, hostVelocities, 2 * 3 * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    GpuPropagator propagator(devPosition, devVelocity, 2, &cublas, 1, nullptr, forceModel, 10);
    for (int i = 0; i < 10; i++) {
        propagator.Step(hostPositions, hostVelocities);
        for (int j = 0; j < 6; j++) {
            EXPECT_DOUBLE_EQ(hostPositions[j], hostEpochPostions[j] + i + 1);
            EXPECT_DOUBLE_EQ(hostVelocities[j], hostEpochVelocities[j]);
        }
    }

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cublasDestroy(cublas);
}

template <typename accelerationFunction, typename actualFunction>
double getGlobalError(int steps, double stepSize, double hostPosition[3], double hostVelocity[3],
    accelerationFunction acceleration, actualFunction actual, int numCorrections)
{
    constexpr int numSatellites = 1;
    MockForceModel model(numSatellites, stepSize, acceleration);
    constexpr int dim = 3;

    double* devPosition;
    double* devVelocity;

    cudaMalloc((void**)&devPosition, sizeof(double) * dim);
    cudaMalloc((void**)&devVelocity, sizeof(double) * dim);

    cudaMemcpy(devPosition, hostPosition, sizeof(double) * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, hostVelocity, sizeof(double) * dim, cudaMemcpyHostToDevice);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    GpuPropagator propagator(devPosition, devVelocity, numSatellites, &cublas, stepSize, nullptr,
        model, steps, numCorrections);

    // Propagate and find the biggest error
    double err = 0;
    double expectedPosition[dim];
    for (int i = 0; i < steps; ++i) {
        double calculatedPosition[3];
        double calculatedVelocity[3];
        propagator.Step(calculatedPosition, calculatedVelocity);
        double exact = actual((i + 1) * stepSize);
        expectedPosition[0] = exact;
        expectedPosition[1] = exact;
        expectedPosition[2] = exact;
        double currentError = l2_error(expectedPosition, calculatedPosition, dim);
        if (currentError > err) {
            err = currentError;
        }
    }

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cublasDestroy(cublas);

    return err;
}

TEST_F(IntegratorTest, Poly7)
{
    int stepCount = 20;
    double stepSize = 0.1;
    double hostPosition[3] = { 0, 0, 0 };
    double hostVelocity[3] = { 0, 0, 0 };

    double err = getGlobalError(
        stepCount, stepSize, hostPosition, hostVelocity, Poly7Accel(), Exact7Position(), 1);
    EXPECT_NEAR(err, 0, 1e-13);
}

TEST_F(IntegratorTest, Poly8)
{
    int stepCount = 20;
    double stepSize = 0.1;
    double hostPosition[3] = { 0, 0, 0 };
    double hostVelocity[3] = { 0, 0, 0 };

    double err = getGlobalError(
        stepCount, stepSize, hostPosition, hostVelocity, Poly8Accel(), Exact8Position(), 1);
    EXPECT_NEAR(err, 0, 2e-13);
}

TEST_F(IntegratorTest, Convergance)
{
    int stepCount = 40;
    double stepSize = 0.1;
    double hostPosition[3] = { 1, 1, 1 };
    double hostVelocity[3] = { 0, 0, 0 };

    double prevErr = 1000000000;
    for (int numCorrections = 1; numCorrections <= 5; numCorrections++) {

        double err = getGlobalError(stepCount, stepSize, hostPosition, hostVelocity,
            DampedOscillatorAccel(), DampedOscillatorExact(), numCorrections);
        EXPECT_GT(prevErr, err) << "Num corrections: " << numCorrections;
        prevErr = err;
    }
}

TEST_F(IntegratorTest, FinalizeCorrectionKernel)
{
    double stepSize = 0.1;
    double hostPosition[3] = { 1, 1, 1 };
    double hostVelocity[3] = { 2, 2, 2 };
    double hostBigS[3] = { 3, 3, 3 };
    double hostSmallS[3] = { 4, 4, 4 };

    double hostAccelerationMatrix[27]
        = { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9 };
    double *devPosition, *devVelocity, *devBigS, *devSmallS, *devAccelerationMatrix;
    gpuErrchk(cudaMalloc((void**)&devPosition, sizeof(double) * 3));
    gpuErrchk(cudaMalloc((void**)&devVelocity, sizeof(double) * 3));
    gpuErrchk(cudaMalloc((void**)&devBigS, sizeof(double) * 3));
    gpuErrchk(cudaMalloc((void**)&devSmallS, sizeof(double) * 3));
    gpuErrchk(cudaMalloc((void**)&devAccelerationMatrix, sizeof(double) * 27));

    cudaMemcpy(devPosition, hostPosition, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, hostVelocity, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devBigS, hostBigS, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devSmallS, hostSmallS, sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(
        devAccelerationMatrix, hostAccelerationMatrix, sizeof(double) * 27, cudaMemcpyHostToDevice);

    double* devA;
    double* devB;
    gpu_constants::createCoefficients(&devA, &devB);

    FinalizeCorrectionKernel<<<1, 1>>>(devPosition, devVelocity, devAccelerationMatrix, devSmallS,
        devBigS, devA, devB, stepSize, 1);
    gpuErrchk(cudaGetLastError());
    double calculatedPosition[3];
    double calculatedVelocity[3];

    cudaMemcpy(calculatedPosition, devPosition, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(calculatedVelocity, devVelocity, 3 * sizeof(double), cudaMemcpyDeviceToHost);

    double expectedVelocity[3] = { 0.408277901785714, 0.408277901785714, 0.408277901785714 };
    double expectedPosition[3] = { 0.045496538487554, 0.045496538487554, 0.045496538487554 };

    EXPECT_FLOAT_EQ(expectedPosition[0], calculatedPosition[0]);
    EXPECT_FLOAT_EQ(expectedPosition[1], calculatedPosition[1]);
    EXPECT_FLOAT_EQ(expectedPosition[2], calculatedPosition[2]);
    EXPECT_FLOAT_EQ(expectedVelocity[0], calculatedVelocity[0]);
    EXPECT_FLOAT_EQ(expectedVelocity[1], calculatedVelocity[1]);
    EXPECT_FLOAT_EQ(expectedVelocity[2], calculatedVelocity[2]);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devPosition);
    cudaFree(devVelocity);
    cudaFree(devBigS);
    cudaFree(devSmallS);
    cudaFree(devAccelerationMatrix);
}
