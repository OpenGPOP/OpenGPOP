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

#include "closest_approach/closest_approach.cuh"
#include "closest_approach/utility/interpolation.cuh"
#include "closest_approach/utility/partition_parameters.cuh"

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
    if (idx >= partitionParams.numPairs)
        return;

    unsigned idxA, idxB;
    if (!partitionParams.GetPartitionMapPair(idx, idxA, idxB))
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

    // TODO: Fix race?
    UpdateLastArray(lastPos, pos, idxA, idxB);
    UpdateLastArray(lastVel, vel, idxA, idxB);
    UpdateLastArray(lastAcc, acc, idxA, idxB);
}

__global__ void CalculateClosestApproachesKernel(PartitionParameters partParams,
    ApproachParameters appParams, double stepSize, unsigned totalSteps, double* closestTimes)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= partParams.numPairs)
        return;

    // Check if first/last step was closest
    auto closestStep = appParams.closestSteps[idx];
    if (closestStep == 0) {
        closestTimes[idx] = 0.0;
        return;
    }
    if (closestStep == totalSteps - 1) {
        closestTimes[idx] = (double)totalSteps * stepSize;
        return;
    }

    // Interpolate object A
    HermitePolynomial polyAx(appParams.before.o1.position.x[idx],
        appParams.before.o1.velocity.x[idx], appParams.before.o1.acceleration.x[idx], // before
        appParams.closest.o1.position.x[idx], appParams.closest.o1.velocity.x[idx],
        appParams.closest.o1.acceleration.x[idx], // closest
        appParams.after.o1.position.x[idx], appParams.after.o1.velocity.x[idx],
        appParams.after.o1.acceleration.x[idx], // after
        stepSize);
    HermitePolynomial polyAy(appParams.before.o1.position.y[idx],
        appParams.before.o1.velocity.y[idx], appParams.before.o1.acceleration.y[idx], // before
        appParams.closest.o1.position.y[idx], appParams.closest.o1.velocity.y[idx],
        appParams.closest.o1.acceleration.y[idx], // closest
        appParams.after.o1.position.y[idx], appParams.after.o1.velocity.y[idx],
        appParams.after.o1.acceleration.y[idx], // after
        stepSize);
    HermitePolynomial polyAz(appParams.before.o1.position.z[idx],
        appParams.before.o1.velocity.z[idx], appParams.before.o1.acceleration.z[idx], // before
        appParams.closest.o1.position.z[idx], appParams.closest.o1.velocity.z[idx],
        appParams.closest.o1.acceleration.z[idx], // closest
        appParams.after.o1.position.z[idx], appParams.after.o1.velocity.z[idx],
        appParams.after.o1.acceleration.z[idx], // after
        stepSize);
    InterpolatedObject objA { polyAx, polyAy, polyAz };

    // Interpolate object B
    HermitePolynomial polyBx(appParams.before.o2.position.x[idx],
        appParams.before.o2.velocity.x[idx], appParams.before.o2.acceleration.x[idx], // before
        appParams.closest.o2.position.x[idx], appParams.closest.o2.velocity.x[idx],
        appParams.closest.o2.acceleration.x[idx], // closest
        appParams.after.o2.position.x[idx], appParams.after.o2.velocity.x[idx],
        appParams.after.o2.acceleration.x[idx], // after
        stepSize);
    HermitePolynomial polyBy(appParams.before.o2.position.y[idx],
        appParams.before.o2.velocity.y[idx], appParams.before.o2.acceleration.y[idx], // before
        appParams.closest.o2.position.y[idx], appParams.closest.o2.velocity.y[idx],
        appParams.closest.o2.acceleration.y[idx], // closest
        appParams.after.o2.position.y[idx], appParams.after.o2.velocity.y[idx],
        appParams.after.o2.acceleration.y[idx], // after
        stepSize);
    HermitePolynomial polyBz(appParams.before.o2.position.z[idx],
        appParams.before.o2.velocity.z[idx], appParams.before.o2.acceleration.z[idx], // before
        appParams.closest.o2.position.z[idx], appParams.closest.o2.velocity.z[idx],
        appParams.closest.o2.acceleration.z[idx], // closest
        appParams.after.o2.position.z[idx], appParams.after.o2.velocity.z[idx],
        appParams.after.o2.acceleration.z[idx], // after
        stepSize);
    InterpolatedObject objB { polyBx, polyBy, polyBz };

    ApproachFunc approachFunc { objA, objB };
    double t = FindRoot(approachFunc, 0.0, stepSize);
    closestTimes[idx] = t + appParams.closestSteps[idx] * stepSize;

    // Calculate the closest distance
    double dx = objA.x(t) - objB.x(t);
    double dy = objA.y(t) - objB.y(t);
    double dz = objA.z(t) - objB.z(t);
    appParams.closestDistances[idx] = sqrt(dx * dx + dy * dy + dz * dz);
}