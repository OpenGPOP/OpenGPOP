#include "monte_carlo/utility.cuh"

__device__ void UpdateDimensions(Arrays3D arrays, double* src, unsigned tIdx, unsigned objIdx)
{
    arrays.x[tIdx] = src[objIdx * 3];
    arrays.y[tIdx] = src[objIdx * 3 + 1];
    arrays.z[tIdx] = src[objIdx * 3 + 2];
}

__device__ void UpdateApproachStates(Objects objs, double* positions, double* velocities,
    double* accelerations, unsigned idx, unsigned idxA, unsigned idxB)
{
    UpdateDimensions(objs.o1.position, positions, idx, idxA);
    UpdateDimensions(objs.o1.velocity, velocities, idx, idxA);
    UpdateDimensions(objs.o1.acceleration, accelerations, idx, idxA);

    UpdateDimensions(objs.o2.position, positions, idx, idxB);
    UpdateDimensions(objs.o2.velocity, velocities, idx, idxB);
    UpdateDimensions(objs.o2.acceleration, accelerations, idx, idxB);
}

__device__ void UpdateLastArray(double* last, double* current, unsigned idxA, unsigned idxB)
{
    auto lastA = last + idxA * 3;
    auto currentA = current + idxA * 3;
    lastA[0] = currentA[0];
    lastA[1] = currentA[1];
    lastA[2] = currentA[2];

    auto lastB = last + idxB * 3;
    auto currentB = current + idxB * 3;
    lastB[0] = currentB[0];
    lastB[1] = currentB[1];
    lastB[2] = currentB[2];
}

__global__ void UpdateClosestApproachKernel(PartitionParameters partitionParams, double* pos,
    double* vel, double* acc, double* lastPos, double* lastVel, double* lastAcc,
    ApproachParameters ap, unsigned stepNum)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= partitionParams.pairNum)
        return;

    unsigned idxA, idxB;
    if (!GetPartitionMapPair(partitionParams, idx, idxA, idxB))
        return; // Should never happen

    auto posA = pos + idxA * 3;
    auto posB = pos + idxB * 3;
    double dx = posA[0] - posB[0];
    double dy = posA[1] - posB[1];
    double dz = posA[2] - posB[2];
    double dist = sqrt(dx * dx + dy * dy + dz * dz);

    // New closest approach
    if (dist < ap.closestDistances[idx]) {
        ap.closestDistances[idx] = dist;
        ap.closestSteps[idx] = stepNum;
        UpdateApproachStates(ap.closest, pos, vel, acc, idx, idxA, idxB);
        UpdateApproachStates(ap.before, lastPos, lastVel, lastAcc, idx, idxA, idxB);
    }
    // Step after the closest approach
    else if (ap.closestSteps[idx] == stepNum - 1) {
        UpdateApproachStates(ap.after, pos, vel, acc, idx, idxA, idxB);
    }

    // TODO: Fix race
    UpdateLastArray(lastPos, pos, idxA, idxB);
    UpdateLastArray(lastVel, vel, idxA, idxB);
    UpdateLastArray(lastAcc, acc, idxA, idxB);
}