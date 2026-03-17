#pragma once

#include "force_model/iforce_model.cuh"
#include "gpu_utils/gpu_utils.cuh"

#include <cuda_runtime.h>

// using AccelerationFunction_t = std::array<double, 3> (*)(
//     std::array<double, 3> position, std::array<double, 3> velocity, double time);

template <typename accelerationFunction>
__global__ void ComputeMockAccelerationKernel(int numEntities, double* position, double* velocity,
    double time, accelerationFunction acceleration, double* out, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEntities) {
        return;
    }

    double* pos = position + idx * 3;
    double* vel = velocity + idx * 3;

    auto acc = acceleration(pos, vel, time);
    out[idx * stride] = acc[0];
    out[idx * stride + 1] = acc[1];
    out[idx * stride + 2] = acc[2];
    delete[] acc;
}

template <typename accelerationFunction> class MockForceModel : public IForceModel {
public:
    MockForceModel(int numEntities, double stepSize, accelerationFunction acceleration)
        : _numEntities(numEntities)
        , _stepSize(stepSize)
        , _acceleration(acceleration)
    {
    }
    ~MockForceModel() = default;

    virtual void PreInitialize() override { }
    virtual void InitializeGeneration(int beforeEpoch, int afterEpoch) override { }

    // This method computes the acceleration for the initial point generation phase
    //
    // IMPORTANT: kIndex is indexed from 0
    //
    // Device pointers passed here are not managed by this class and should be allocated and freed
    // by the caller.
    virtual void ComputeGenerationAcceleration(double* devPosition, double* devVelocity,
        double* devKMatrix, int stepNum, int kIndex) override
    {
        int kMatrixColumnCount = 10;
        int direction = stepNum > 0 ? -1 : 1;
        constexpr double rk8Midpoints[] = { 0.0, 4.0 / 27.0, 2.0 / 9.0, 1.0 / 3.0, 1.0 / 2.0,
            2.0 / 3.0, 1.0 / 6.0, 1.0, 5.0 / 6.0, 1.0 };
        double time = static_cast<double>(stepNum + direction) * _stepSize
            + (-direction) * rk8Midpoints[kIndex] * _stepSize;

        int threadsPerBlock = 256;
        int blockCount = _numEntities / threadsPerBlock + 1;
        ComputeMockAccelerationKernel<accelerationFunction>
            <<<blockCount, threadsPerBlock>>>(_numEntities, devPosition, devVelocity, time,
                _acceleration, devKMatrix + kIndex * 3, 3 * kMatrixColumnCount);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    // ----------------------------
    // Correction and propagation phases

    // This method should be called after the generation phase and before the correction and
    // propagation phases. The device pointers passed here are not managed by this class and should
    // be allocated and freed by the caller.
    virtual void InitializePropagation(int beforeEpoch, int afterEpoch) override { }

    // TODO: steps indexed from 0
    // This method overwrites the last column of the acceleration matrix
    virtual void ComputePropagationAcceleration(double* devPosition, double* devVelocity,
        double* devAccelerationMatrix, int stepNum, int returnColumnIndex) override
    {
        int threadsPerBlock = 256;
        int blockCount = _numEntities / threadsPerBlock + 1;
        double time = (double)(stepNum - 4) * _stepSize;

        ComputeMockAccelerationKernel<accelerationFunction>
            <<<blockCount, threadsPerBlock>>>(_numEntities, devPosition, devVelocity, time,
                _acceleration, devAccelerationMatrix + returnColumnIndex * 3, 3 * 9);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

private:
    int _numEntities;
    double _stepSize;
    accelerationFunction _acceleration;
};