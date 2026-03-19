#include "closest_approach/closest_approach.cuh"
#include "gpu_test_helpers.cuh"
#include "mock_force_model.cuh"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

class ClosestApproachTest : public ::testing::Test {
protected:
    cublasHandle_t cublasHandle;
    FullPerturbationParameters params;

    void SetUp() override
    {
        cublasStatus_t stat = cublasCreate(&cublasHandle);
        ASSERT_EQ(stat, CUBLAS_STATUS_SUCCESS) << "cuBLAS initialization failed!";
    }

    void TearDown() override
    {
        cublasDestroy(cublasHandle);

        // Check for uncaught errors
        gpuErrchk(cudaGetLastError());
        cudaDeviceReset();
    }
};

TEST_F(ClosestApproachTest, InitializeAllocatesMemoryWithoutCrashing)
{
    int numSatellites = 2;
    double stepSize = 1.0;

    NoAcceleration noAccel;
    MockForceModel forceModel(numSatellites, stepSize, noAccel);
    ClosestApproach ca(&cublasHandle, &params, forceModel, stepSize);

    std::vector<double> pos = { 0.0, 0.0, 0.0, 10.0, 10.0, 10.0 };
    std::vector<double> vel = { 1.0, 0.0, 0.0, -1.0, 0.0, 0.0 };
    std::vector<unsigned> sets = { 0, 1 };

    EXPECT_NO_THROW(ca.Initialize(pos, vel, sets));
}

TEST_F(ClosestApproachTest, PropagateHeadOnCollision)
{
    int numSatellites = 2;
    double stepSize = 0.5; // Decreased step size to 0.5 to allow for more steps

    NoAcceleration noAccel;
    MockForceModel forceModel(numSatellites, stepSize, noAccel);
    ClosestApproach ca(&cublasHandle, &params, forceModel, stepSize);

    // Entity 0: at origin, moving +X at 1 m/s
    // Entity 1: at X=10, moving -X at 1 m/s
    std::vector<double> pos = { 0.0, 0.0, 0.0, 10.0, 0.0, 0.0 };
    std::vector<double> vel = { 1.0, 0.0, 0.0, -1.0, 0.0, 0.0 };
    std::vector<unsigned> sets = { 0, 1 };

    ca.Initialize(pos, vel, sets);

    // Propagate 10 steps of 0.5 seconds (evaluating at t=5.0)
    ca.Propagate(10);

    ClosestApproachData data;
    ca.FindClosest(data);

    Approach app = data.Get(0, 1);

    // Distance should be effectively 0 at precisely t = 5.0
    EXPECT_NEAR(app.distance, 0.0, 1e-14);
    EXPECT_NEAR(app.time, 5.0, 1e-14);
}

TEST_F(ClosestApproachTest, ParallelTrajectoriesMaintainDistance)
{
    int numSatellites = 2;
    double stepSize = 0.5;

    NoAcceleration noAccel;
    MockForceModel forceModel(numSatellites, stepSize, noAccel);
    ClosestApproach ca(&cublasHandle, &params, forceModel, stepSize);

    // Entity 0: at origin, moving +Y at 2 m/s
    // Entity 1: at X=5, moving +Y at 2 m/s
    std::vector<double> pos = { 0.0, 0.0, 0.0, 5.0, 0.0, 0.0 };
    std::vector<double> vel = { 0.0, 2.0, 0.0, 0.0, 2.0, 0.0 };
    std::vector<unsigned> sets = { 0, 1 };

    ca.Initialize(pos, vel, sets);

    // Increased to 20 steps to be well past the 10-step minimum
    ca.Propagate(20);

    ClosestApproachData data;
    ca.FindClosest(data);

    Approach app = data.Get(0, 1);

    // Because they are moving parallel, the distance remains strictly 5.0
    EXPECT_DOUBLE_EQ(app.distance, 5.0);
    EXPECT_GE(app.time, 0.0);
}

TEST_F(ClosestApproachTest, MultiplePropagationPhasesReturnGlobalMinimum)
{
    int numSatellites = 2;
    double stepSize = 0.1; // Reduced step size to 0.1 to multiply step counts by 10

    NoAcceleration noAccel;
    MockForceModel forceModel(numSatellites, stepSize, noAccel);
    ClosestApproach ca(&cublasHandle, &params, forceModel, stepSize);

    // Entity 0: at origin, stationary
    // Entity 1: at X=10, moving -X at 1 m/s
    std::vector<double> pos = { 0.0, 0.0, 0.0, 10.0, 0.0, 0.0 };
    std::vector<double> vel = { 0.0, 0.0, 0.0, -1.0, 0.0, 0.0 };
    std::vector<unsigned> sets = { 0, 1 };

    ca.Initialize(pos, vel, sets);

    ClosestApproachData data;

    // Phase 1: Propagate 40 steps to t=4.0 (distance is 6)
    ca.Propagate(40);
    ca.FindClosest(data);
    EXPECT_NEAR(data.Get(0, 1).distance, 6.0, 1e-14);

    // Phase 2: Propagate 60 steps to t=10.0 (collides at t=10, distance is 0)
    ca.Propagate(60);
    ca.FindClosest(data);
    EXPECT_NEAR(data.Get(0, 1).distance, 0.0, 1e-14);
    EXPECT_NEAR(data.Get(0, 1).time, 10.0, 1e-14);

    // Phase 3: Propagate 50 steps to t=15.0 (moves past, distance goes back up to 5)
    ca.Propagate(50);
    ca.FindClosest(data);

    // The FindClosest function should remember the global minimum from Phase 2
    EXPECT_NEAR(data.Get(0, 1).distance, 0.0, 1e-14);
    EXPECT_NEAR(data.Get(0, 1).time, 10.0, 1e-14);
}

TEST_F(ClosestApproachTest, ClosestApproachBetweenStepsNonZero)
{
    int numSatellites = 2;
    double stepSize = 0.1; // Steps at t = 0.0, 0.1, 0.2, 0.3...

    NoAcceleration noAccel;
    MockForceModel forceModel(numSatellites, stepSize, noAccel);
    ClosestApproach ca(&cublasHandle, &params, forceModel, stepSize);

    // Entity 0: at origin, stationary
    // Entity 1: starts at X=0.5, Y=3.0, moving -X at 2 m/s
    std::vector<double> pos = { 0.0, 0.0, 0.0, 0.5, 3.0, 0.0 };
    std::vector<double> vel = { 0.0, 0.0, 0.0, -2.0, 0.0, 0.0 };
    std::vector<unsigned> sets = { 0, 1 };

    ca.Initialize(pos, vel, sets);

    // Propagate 5 steps to t=0.5s.
    // The closest approach occurs when Entity 1 crosses X=0.
    // At V=-2.0 m/s, X=0 is reached at t=0.25s.
    // This is exactly halfway between the step at 0.2s and 0.3s.
    ca.Propagate(5);

    ClosestApproachData data;
    ca.FindClosest(data);

    EXPECT_NEAR(data.Get(0, 1).distance, 3.0, 1e-14);
    EXPECT_NEAR(data.Get(0, 1).time, 0.25, 1e-14);
}

TEST_F(ClosestApproachTest, MultiObjectSetsEvaluateCorrectPairs)
{
    int numSatellites = 3;
    double stepSize = 0.1;

    NoAcceleration noAccel;
    MockForceModel forceModel(numSatellites, stepSize, noAccel);
    ClosestApproach ca(&cublasHandle, &params, forceModel, stepSize);

    // Entity 0 (Set 0): at origin, stationary
    // Entity 1 (Set 1): starts at X=10, Y=5, Z=0, moving -X at 1 m/s
    // Entity 2 (Set 1): starts at X=3, Y=0, Z=8, moving -Z at 2 m/s
    std::vector<double> pos = {
        0.0, 0.0, 0.0, // Entity 0
        10.0, 5.0, 0.0, // Entity 1
        3.0, 0.0, 8.0 // Entity 2
    };
    std::vector<double> vel = {
        0.0, 0.0, 0.0, // Entity 0
        -1.0, 0.0, 0.0, // Entity 1
        0.0, 0.0, -2.0 // Entity 2
    };
    std::vector<unsigned> sets = { 0, 1, 1 };

    ca.Initialize(pos, vel, sets);

    // Propagate 120 steps to t=12.0s to cover both closest approach events
    ca.Propagate(120);

    ClosestApproachData data;
    ca.FindClosest(data);

    // Check Entity 0 vs Entity 1 (Pair 0-1)
    // Closest approach occurs at t=10.0s when X=0. Minimum distance is the Y-offset (5.0).
    EXPECT_NEAR(data.Get(0, 1).distance, 5.0, 1e-14);
    EXPECT_NEAR(data.Get(0, 1).time, 10.0, 1e-14);

    // Check Entity 0 vs Entity 2 (Pair 0-2)
    // Closest approach occurs at t=4.0s when Z=0. Minimum distance is the X-offset (3.0).
    EXPECT_NEAR(data.Get(0, 2).distance, 3.0, 1e-14);
    EXPECT_NEAR(data.Get(2, 0).time, 4.0, 1e-14);
}