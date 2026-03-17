#pragma once
#include "force_model/types.cuh"

// Responsible for calculating and providing simulation parameters that depend only on time
struct IFMTimePoints {
    virtual ~IFMTimePoints() = default;

    virtual void Initialize() = 0;
    virtual void CalculateParameters(int arrayLength, bool useMidpoints, int beforeEpoch) = 0;

    // Returns a structure of device pointers shifted to the requested position
    virtual TimePointParameters GetParametersDev(int stepNum) = 0;
};