#pragma once

#include "cublas_v2.h"
#include "monte_carlo/types.cuh"
#include "types/Parameters.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <vector>

__global__ void UpdateClosestApproachKernel(PartitionParameters partitionParams, double* pos,
    double* vel, double* acc, double* lastPos, double* lastVel, double* lastAcc,
    ApproachParameters ap, unsigned stepNum);

class ClosestApproach {
public:
    ClosestApproach(cublasHandle_t* cublasHandle, FullPerturbationParameters* params)
        : _cublasHandle(cublasHandle)
        , _params(params)
    {
    }

    // Initializes simulation parameters and allocates memory on the device. The positions and
    // velocities vectors must be of the same size. The setIndices vector must be of size equal to
    // the number of entities. The indices must start from 0 and the difference between consecutive
    // indices cannot exceed 1.
    void Initialize(std::vector<double>& positions, std::vector<double>& velocities,
        std::vector<unsigned>& setIndices);

    // Propagates the objects for the specified number of steps with the given step size. On
    // subsequent calls, the propagation will continue from the last state.
    void Propagate(double stepSize, int numSteps);

    // The function finds the number of satellites that are closer than the given distance. If
    // hostClosestDistances is provided, it will be filled with the distances. When called after
    // multiple propagation phases, the closest distances across all phases will be returned.
    int FindClosest(double distanceThreshold, double* hostClosestDistances = nullptr);

private:
    cublasHandle_t* _cublasHandle;
    FullPerturbationParameters* _params;

    unsigned _numEntities = 0;
    double* _devPositions;
    double* _devVelocities;
    double* _devAccelerations;
    double* _devLastPositions;
    double* _devLastVelocities;
    double* _devLastAccelerations;

    PartitionParameters _partitionParams;
    ApproachParameters _approachStates;
};