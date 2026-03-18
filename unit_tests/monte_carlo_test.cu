#include "monte_carlo/monte_carlo.cuh"
#include "monte_carlo/utility.cuh"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>

// --- Helper to fetch a single double from device memory ---
double fetchDeviceDouble(double* d_ptr, int index = 0)
{
    double h_val;
    cudaMemcpy(&h_val, d_ptr + index, sizeof(double), cudaMemcpyDeviceToHost);
    return h_val;
}

double fetchDeviceUnsigned(unsigned* d_ptr, int index = 0)
{
    unsigned h_val;
    cudaMemcpy(&h_val, d_ptr + index, sizeof(unsigned), cudaMemcpyDeviceToHost);
    return h_val;
}

// --- Helper to copy arrays to device ---
void copyToDevice(double* d_dest, const std::vector<double>& h_src)
{
    cudaMemcpy(d_dest, h_src.data(), h_src.size() * sizeof(double), cudaMemcpyHostToDevice);
}

class ClosestApproachTest : public ::testing::Test {
protected:
    unsigned pairNum = 1;
    unsigned numObjects = 2; // Testing 1 pair (Object 0 and Object 1)

    PartitionParameters d_partitionParams;
    ApproachParameters d_ap;

    // Flat arrays for pos, vel, acc
    double *d_pos, *d_vel, *d_acc;
    double *d_lastPos, *d_lastVel, *d_lastAcc;

    void SetUp() override
    {
        // 1. Allocate main simulation arrays
        cudaMalloc(&d_pos, numObjects * 3 * sizeof(double));
        cudaMalloc(&d_vel, numObjects * 3 * sizeof(double));
        cudaMalloc(&d_acc, numObjects * 3 * sizeof(double));
        cudaMalloc(&d_lastPos, numObjects * 3 * sizeof(double));
        cudaMalloc(&d_lastVel, numObjects * 3 * sizeof(double));
        cudaMalloc(&d_lastAcc, numObjects * 3 * sizeof(double));

        // 2. Setup PartitionParameters for 1 pair
        PartitionParameters tempParams;
        CalculatePartitionParameters(std::vector<unsigned> { 0, 1 }, tempParams);
        tempParams.setIds = new unsigned[2] { 0, 1 };
        cudaMalloc(&d_partitionParams.setOffsets, 2 * sizeof(unsigned));
        cudaMalloc(&d_partitionParams.numHigherIds, 2 * sizeof(unsigned));
        cudaMalloc(&d_partitionParams.mapOffsets, 2 * sizeof(unsigned));
        cudaMalloc(&d_partitionParams.setIds, 2 * sizeof(unsigned));
        cudaMemcpy(d_partitionParams.setOffsets, tempParams.setOffsets, 2 * sizeof(unsigned),
            cudaMemcpyHostToDevice);
        cudaMemcpy(d_partitionParams.numHigherIds, tempParams.numHigherIds, 2 * sizeof(unsigned),
            cudaMemcpyHostToDevice);
        cudaMemcpy(d_partitionParams.mapOffsets, tempParams.mapOffsets, 2 * sizeof(unsigned),
            cudaMemcpyHostToDevice);
        cudaMemcpy(d_partitionParams.setIds, tempParams.setIds, 2 * sizeof(unsigned),
            cudaMemcpyHostToDevice);
        delete[] tempParams.setOffsets;
        delete[] tempParams.numHigherIds;
        delete[] tempParams.mapOffsets;
        delete[] tempParams.setIds;

        // 3. Allocate self-managing approach parameters
        d_ap.cudaAllocate(pairNum);

        // 4. Initialize closestDistances to infinity, closestSteps to 0
        std::vector<double> initDist(pairNum, std::numeric_limits<double>::max());
        std::vector<unsigned> initSteps(pairNum, 0);
        cudaMemcpy(d_ap.closestDistances, initDist.data(), pairNum * sizeof(double),
            cudaMemcpyHostToDevice);
        cudaMemcpy(d_ap.closestSteps, initSteps.data(), pairNum * sizeof(unsigned),
            cudaMemcpyHostToDevice);
    }

    void TearDown() override
    {
        d_ap.cudaFree();
        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc);
        cudaFree(d_lastPos);
        cudaFree(d_lastVel);
        cudaFree(d_lastAcc);
        cudaFree(d_partitionParams.setOffsets);
        cudaFree(d_partitionParams.numHigherIds);
        cudaFree(d_partitionParams.mapOffsets);
        cudaFree(d_partitionParams.setIds);

        cudaDeviceReset(); // Clean up all resources and reset device state
    }
};

TEST_F(ClosestApproachTest, DetectsClosestAndAfterStates)
{
    // --- STEP 1: Objects are moving closer ---
    std::vector<double> h_lastPos = { -1, 0, 0, 12, 0, 0 }; // Distance was 13 in step 0
    std::vector<double> h_pos = { 2, 0, 0, 10, 0, 0 }; // Distance is 8 in step 1
    std::vector<double> h_vel = { 1, 0, 0, -1, 0, 0 }; // Dummy velocities
    std::vector<double> h_acc = { 0, 0, 0, 0, 0, 0 };

    copyToDevice(d_lastPos, h_lastPos);
    copyToDevice(d_pos, h_pos);
    copyToDevice(d_vel, h_vel);
    copyToDevice(d_acc, h_acc);

    UpdateClosestApproachKernel<<<1, 1>>>(
        d_partitionParams, d_pos, d_vel, d_acc, d_lastPos, d_lastVel, d_lastAcc, d_ap, 1);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Verify 'Closest' was updated
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closestDistances, 0), 8.0);
    EXPECT_EQ(fetchDeviceUnsigned(d_ap.closestSteps, 0), 1u);

    // Object B
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.before.o2.position.x, 0), 12.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.before.o2.position.y, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.before.o2.position.z, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closest.o2.position.x, 0), 10.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closest.o2.position.y, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closest.o2.position.z, 0), 0.0);

    // Object A
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.before.o1.position.x, 0), -1.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.before.o1.position.y, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.before.o1.position.z, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closest.o1.position.x, 0), 2.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closest.o1.position.y, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closest.o1.position.z, 0), 0.0);

    // --- STEP 2: Objects move further apart ---
    std::vector<double> h_pos_step2 = { 0, 0, 0, 11, 0, 0 };
    copyToDevice(d_pos, h_pos_step2);

    UpdateClosestApproachKernel<<<1, 1>>>(
        d_partitionParams, d_pos, d_vel, d_acc, d_lastPos, d_lastVel, d_lastAcc, d_ap, 2);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Distance is 11, which is > 8. Closest should NOT update.
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closestDistances, 0), 8.0);
    EXPECT_EQ(fetchDeviceUnsigned(d_ap.closestSteps, 0), 1u); // Still step 1

    // However, closestStep was 1, and current step is 2. The 'After' state should trigger
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.after.o2.position.x, 0), 11.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.after.o2.position.y, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.after.o2.position.z, 0), 0.0);

    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.after.o1.position.x, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.after.o1.position.y, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.after.o1.position.z, 0), 0.0);

    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_lastPos, 0), 0.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_lastPos, 3), 11.0);

    // --- STEP 3: Objects move even further apart ---
    std::vector<double> h_pos_step3 = { -1, 0, 0, 13, 0, 0 };
    copyToDevice(d_pos, h_pos_step3);

    UpdateClosestApproachKernel<<<1, 1>>>(
        d_partitionParams, d_pos, d_vel, d_acc, d_lastPos, d_lastVel, d_lastAcc, d_ap, 3);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Distance is 14, which is > 8. Closest should NOT update.
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_ap.closestDistances, 0), 8.0);
    EXPECT_EQ(fetchDeviceUnsigned(d_ap.closestSteps, 0), 1u); // Still step 1
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_lastPos, 0), -1.0);
    EXPECT_DOUBLE_EQ(fetchDeviceDouble(d_lastPos, 3), 13.0);
}