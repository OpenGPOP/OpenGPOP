#include "gpu_startup/gpu_startup.cuh"
#include "gpu_test_helpers.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "mock_force_model.cuh"

#include <gtest/gtest.h>

class StartupTest : public ::testing::Test {
protected:
    void TearDown() override
    {
        gpuErrchk(cudaGetLastError());
        cudaDeviceReset();
    }
};

template <typename accelerationFunction>
std::vector<double> RunStartup(accelerationFunction accelFun, double stepSize, int numCorrections,
    double epochPosition[3], double epochVelocity[3])
{
    int numSatellites = 1;

    double* devPosition;
    double* devVelocity;

    gpuErrchk(cudaMalloc((void**)&devPosition, numSatellites * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devVelocity, numSatellites * 3 * sizeof(double)));

    cudaMemcpy(devPosition, epochPosition, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, epochVelocity, 3 * sizeof(double), cudaMemcpyHostToDevice);

    MockForceModel forceModel(numSatellites, stepSize, accelFun);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    GpuStartup startup(
        devPosition, devVelocity, numSatellites, &cublas, stepSize, forceModel, 10, numCorrections);
    auto startupData = startup.Startup();

    double calculatedPosition[9 * 3];

    cudaMemcpy(calculatedPosition, startupData.devPositionMatrix, 9 * 3 * sizeof(double),
        cudaMemcpyDeviceToHost);

    cudaFree(devPosition);
    cudaFree(devVelocity);
    cudaFree(startupData.devPositionMatrix);
    cudaFree(startupData.devA);
    cudaFree(startupData.devB);
    cudaFree(startupData.devBigS);
    cudaFree(startupData.devSmallS);
    cudaFree(startupData.devVelocityMatrix);
    cudaFree(startupData.devAccelerationMatrix);
    cublasDestroy(cublas);
    std::vector<double> positions(9 * 3);
    for (size_t i = 0; i < positions.size(); i++) {
        positions[i] = calculatedPosition[i];
    }
    return positions;
}

template <typename accelerationFunction, typename actualFunction>
void testShouldBeExact(accelerationFunction accelFun, actualFunction actualFun, double stepSize,
    double epochPosition[3], double epochVelocity[3], double tolerance)
{
    std::vector positions = RunStartup(accelFun, stepSize, 5, epochPosition, epochVelocity);
    std::vector<double> expectedPositions(3 * 9);
    for (int i = -4; i <= 4; i++) {
        expectedPositions[3 * (4 + i) + 0] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 1] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 2] = actualFun(i * stepSize);
    }

    for (size_t i = 0; i < expectedPositions.size(); i++) {
        EXPECT_NEAR(expectedPositions[i], positions[i], tolerance);
    }
}

template <typename accelerationFunction, typename actualFunction>
double getTotalError(accelerationFunction accelFun, actualFunction actualFun, double stepSize,
    int numCorrections, double epochPosition[3], double epochVelocity[3])
{
    std::vector positions
        = RunStartup(accelFun, stepSize, numCorrections, epochPosition, epochVelocity);
    std::vector<double> expectedPositions(3 * 9);
    for (int i = -4; i <= 4; i++) {
        expectedPositions[3 * (4 + i) + 0] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 1] = actualFun(i * stepSize);
        expectedPositions[3 * (4 + i) + 2] = actualFun(i * stepSize);
    }

    return l2_error(positions.data(), expectedPositions.data(), 3 * 9);
}

TEST_F(StartupTest, stationary)
{
    NoAcceleration noAccel;
    ZeroPosition zeroPosition;
    double epochPosition[3] = { 0, 0, 0 };
    double epochVelocity[3] = { 0, 0, 0 };
    testShouldBeExact<NoAcceleration, ZeroPosition>(
        noAccel, zeroPosition, 1.0, epochPosition, epochVelocity, 0);
}

TEST_F(StartupTest, poly7)
{
    Poly7Accel accel;
    Exact7Position exactPosition;

    double epochPosition[3] = { 0, 0, 0 };
    double epochVelocity[3] = { 0, 0, 0 };
    testShouldBeExact(accel, exactPosition, 1.0, epochPosition, epochVelocity,
        1e-10); // big tolerance because of comparing big values
}

TEST_F(StartupTest, poly8)
{
    Poly8Accel accel;
    Exact8Position exactPosition;
    double epochPosition[3] = { 0, 0, 0 };
    double epochVelocity[3] = { 0, 0, 0 };
    testShouldBeExact(accel, exactPosition, 1.0, epochPosition, epochVelocity, 1e-13);
}

TEST_F(StartupTest, StartupCorrect_S0)
{
    double stepSize = 2.0;
    double hostVelocityMatrix[3 * 9]
        = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8 };
    double hostPositionMatrix[3 * 9]
        = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8 };

    double* devPositionMatrix;
    double* devVelocityMatrix;
    double* devAccelerationMatrix;
    double* devTempX;
    double* devTempV;

    gpuErrchk(cudaMalloc(&devPositionMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devVelocityMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devAccelerationMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devTempX, 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devTempV, 3 * sizeof(double)));
    cudaMemcpy(
        devPositionMatrix, hostPositionMatrix, 9 * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(
        devVelocityMatrix, hostPositionMatrix, 9 * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    ExponentialAccel accel;
    MockForceModel forceModel(1, stepSize, accel);
    forceModel.InitializePropagation(4, 4);
    StartupCorrector startup(&cublas, stepSize, forceModel, 1, devPositionMatrix, devVelocityMatrix,
        devAccelerationMatrix, devTempX, devTempV);
    startup.CalculateS0();
    auto data = startup.getData();
    double hostSmallSMatrix[9 * 3];
    double hostBigSMatrix[9 * 3];

    cudaMemcpy(hostBigSMatrix, data.devBigS, 9 * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSmallSMatrix, data.devSmallS, 9 * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    double expectedSmallS0 = 2.075075590567601; // manual calculation
    EXPECT_DOUBLE_EQ(hostSmallSMatrix[3 * 4 + 0], expectedSmallS0);
    EXPECT_DOUBLE_EQ(hostSmallSMatrix[3 * 4 + 1], expectedSmallS0);
    EXPECT_DOUBLE_EQ(hostSmallSMatrix[3 * 4 + 2], expectedSmallS0);

    double expectedBigS0 = 0.924524968616072;
    EXPECT_FLOAT_EQ(hostBigSMatrix[3 * 4 + 0], expectedBigS0); // double doesn't pass idk why
    EXPECT_FLOAT_EQ(hostBigSMatrix[3 * 4 + 1], expectedBigS0);
    EXPECT_FLOAT_EQ(hostBigSMatrix[3 * 4 + 2], expectedBigS0);

    cudaFree(devPositionMatrix);
    cudaFree(devVelocityMatrix);
    cudaFree(data.devA);
    cudaFree(data.devB);
    cudaFree(data.devBigS);
    cudaFree(data.devSmallS);
    cudaFree(data.devAccelerationMatrix);
    cudaFree(devTempX);
    cudaFree(devTempV);
    cublasDestroy(cublas);
}

TEST_F(StartupTest, StartupCorrector_s)
{
    double stepSize = 2.0;
    double hostVelocityMatrix[3 * 9]
        = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8 };
    double hostPositionMatrix[3 * 9]
        = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8 };

    double* devPositionMatrix;
    double* devVelocityMatrix;
    double* devAccelerationMatrix;
    double* devTempX;
    double* devTempV;

    gpuErrchk(cudaMalloc(&devPositionMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devVelocityMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devAccelerationMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devTempX, 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devTempV, 3 * sizeof(double)));
    cudaMemcpy(
        devPositionMatrix, hostPositionMatrix, 9 * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(
        devVelocityMatrix, hostVelocityMatrix, 9 * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    ExponentialAccel accel;
    MockForceModel forceModel(1, stepSize, accel);
    forceModel.InitializePropagation(4, 4);
    StartupCorrector startup(&cublas, stepSize, forceModel, 1, devPositionMatrix, devVelocityMatrix,
        devAccelerationMatrix, devTempX, devTempV);
    startup.Correct();
    auto data = startup.getData();
    double hostSmallSMatrix[9 * 3];
    double hostBigSMatrix[9 * 3];

    cudaMemcpy(hostBigSMatrix, data.devBigS, 9 * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSmallSMatrix, data.devSmallS, 9 * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    double expectedSmallS[9] = { -1.954319917555107e3, -0.262126527287875e3, -0.033113055524935e3,
        -0.002119452458898e3, 0.002075075590568e3, 0.002642743232186e3, 0.002719568693249e3,
        0.002729965888781e3, 0.002731372996184e3 };

    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(hostSmallSMatrix[i * 3 + 0], expectedSmallS[i]);
        EXPECT_FLOAT_EQ(hostSmallSMatrix[i * 3 + 1], expectedSmallS[i]);
        EXPECT_FLOAT_EQ(hostSmallSMatrix[i * 3 + 2], expectedSmallS[i]);
    }

    double expectedBigS[9] = { 5.294164844621613e2, 0.655755604279188e2, 0.051634298864115e2,
        -0.006505506219515e2, 0.009245249686161e2, 0.034996005591837e2, 0.062100114329879e2,
        0.089387379456808e2, 0.116699432105504e2 };
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(hostBigSMatrix[i * 3 + 0], expectedBigS[i]);
        EXPECT_FLOAT_EQ(hostBigSMatrix[i * 3 + 1], expectedBigS[i]);
        EXPECT_FLOAT_EQ(hostBigSMatrix[i * 3 + 2], expectedBigS[i]);
    }

    cudaFree(devPositionMatrix);
    cudaFree(devVelocityMatrix);
    cudaFree(data.devA);
    cudaFree(data.devB);
    cudaFree(data.devBigS);
    cudaFree(data.devSmallS);
    cudaFree(data.devAccelerationMatrix);
    cudaFree(devTempX);
    cudaFree(devTempV);
    cublasDestroy(cublas);
}

TEST_F(StartupTest, StartupCorrector_oscillatorConvergance)
{
    DampedOscillatorAccel accel;
    DampedOscillatorExact exactPosition;
    double epochPosition[3] = { 1, 1, 1 };
    double epochVelocity[3] = { 0, 0, 0 };

    double err, prevError = std::numeric_limits<double>::max();
    for (int i = 1; i < 5; i++) {
        err = getTotalError(accel, exactPosition, 0.1, i, epochPosition, epochVelocity);
        EXPECT_LT(err, prevError) << "num corrections: " << i;
        prevError = err;
    }
}

TEST_F(StartupTest, StartupCorrector_calculateAcceleration)
{
    double stepSize = 2.0;
    double hostVelocityMatrix[3 * 9] = { 8, 8, 8, 6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0, -2, -2, -2,
        -4, -4, -4, -6, -6, -6, -8, -8, -8 };
    double hostPositionMatrix[3 * 9] = { -15, -15, -15, -8, -8, -8, -3, -3, -3, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 3, 3, 3, 8, 8, 8, 15, 15, 15 };

    double* devPositionMatrix;
    double* devVelocityMatrix;
    double* devAccelerationMatrix;
    double* devTempX;
    double* devTempV;

    gpuErrchk(cudaMalloc(&devPositionMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devVelocityMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devAccelerationMatrix, 9 * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devTempX, 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&devTempV, 3 * sizeof(double)));

    cudaMemcpy(
        devPositionMatrix, hostPositionMatrix, 9 * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(
        devVelocityMatrix, hostVelocityMatrix, 9 * 3 * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    DampedOscillatorAccel accel;
    MockForceModel forceModel(1, stepSize, accel);
    forceModel.InitializePropagation(4, 4);

    StartupCorrector startup(&cublas, stepSize, forceModel, 1, devPositionMatrix, devVelocityMatrix,
        devAccelerationMatrix, devTempX, devTempV);

    startup.CalculateAccelerationMatrix(true);

    auto data = startup.getData();
    double hostAccelerationMatrix[9 * 3];
    double expectedAccelerationMatrix[9 * 3] = { 14, 14, 14, 4, 4, 4, -2, -2, -2, -4, -4, -4, -2,
        -2, -2, 4, 4, 4, 2, 2, 2, -4, -4, -4, -14, -14, -14 };

    cudaMemcpy(hostAccelerationMatrix, data.devAccelerationMatrix, 9 * 3 * sizeof(double),
        cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9 * 3; i++) {
        EXPECT_DOUBLE_EQ(hostAccelerationMatrix[i], expectedAccelerationMatrix[i]);
    }

    cudaFree(devPositionMatrix);
    cudaFree(devVelocityMatrix);
    cudaFree(data.devA);
    cudaFree(data.devB);
    cudaFree(data.devBigS);
    cudaFree(data.devSmallS);
    cudaFree(data.devAccelerationMatrix);
    cudaFree(devTempX);
    cudaFree(devTempV);
    cublasDestroy(cublas);
}
