#pragma once

struct IForceModel {
    virtual void PreInitialize() = 0;

    virtual void InitializeGeneration(int beforeEpoch, int afterEpoch) = 0;

    virtual void ComputeGenerationAcceleration(
        double* devPosition, double* devVelocity, double* devKMatrix, int stepNum, int kIndex) = 0;

    virtual void InitializePropagation(int beforeEpoch, int afterEpoch) = 0;

    virtual void ComputePropagationAcceleration(double* devPosition, double* devVelocity,
        double* devAcceleration, int stepNum, int returnColumnIndex) = 0;
};