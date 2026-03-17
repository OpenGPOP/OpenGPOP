#pragma once

// A common interface for all acceleration sources
struct IAccelerationSource {
    virtual ~IAccelerationSource() = default;

    virtual void Initialize() = 0;

    // Computes the acceleration coming from a source and returns the vectors to the out array,
    // placed every outStride * 3 elements.
    //
    // For outStride = 1: out = { ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ... }
    // For outStride = 2: out = { ax1, ay1, az1, ___, ___, ___, ax2, ay2, az2, ___, ___, ___, ... }
    // where ___ are left as before
    virtual void ComputeAcceleration(
        double* devPosition, double* devVelocity, int stepNum, double* out, int outStride) = 0;
};