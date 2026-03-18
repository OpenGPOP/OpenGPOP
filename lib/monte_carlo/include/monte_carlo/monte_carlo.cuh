#pragma once

#include "cublas_v2.h"
#include "monte_carlo/types.cuh"
#include "monte_carlo/utility.cuh"
#include "types/Parameters.hpp"
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

__global__ void UpdateClosestApproachKernel(PartitionParameters partitionParams, double* pos,
    double* vel, double* acc, double* lastPos, double* lastVel, double* lastAcc,
    ApproachParameters ap, unsigned stepNum);

__global__ void CalculateClosestApproachesKernel(PartitionParameters partParams,
    ApproachParameters appParams, double stepSize, double* closestTimes);

struct Approach {
    double distance;
    double time;
};

class ClosestApproaches {
    friend class ClosestApproachFinder;

    double* _distances = nullptr;
    double* _times = nullptr;
    PartitionParameters _partParams;

    inline void Populate(PartitionParameters partParams, double* distances, double* times)
    {
        Free();
        _partParams = partParams;
        _distances = distances;
        _times = times;
    }

    void Free()
    {
        delete[] _distances;
        delete[] _times;
        _partParams.hostFree();
    }

public:
    // Returns the closest approach between objA and objB. If the pair is invalid (was not
    // included in the partition) or the structure was not populated, { -1.0, -1.0 } is returned.
    inline Approach Get(unsigned objA, unsigned objB) const
    {
        if (_distances == nullptr || _times == nullptr) {
            return { -1.0, -1.0 }; // Not populated
        }
        if (objA == objB) {
            return { 0.0, 0.0 };
        }
        auto idx = GetPartitionMapIdx(_partParams, objA, objB);
        if (idx < 0) {
            return { -1.0, -1.0 }; // Invalid pair
        }
        return { _distances[idx], _times[idx] };
    }

    inline ~ClosestApproaches() { Free(); }
};

class ClosestApproachFinder {
public:
    ClosestApproachFinder(cublasHandle_t* cublasHandle, FullPerturbationParameters* params)
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
    void FindClosest(ClosestApproaches& closestApproaches);

private:
    cublasHandle_t* _cublasHandle;
    FullPerturbationParameters* _params;

    double _lastStepSize = 0;
    unsigned _numEntities = 0;
    double* _devPositions;
    double* _devVelocities;
    double* _devAccelerations;
    double* _devLastPositions;
    double* _devLastVelocities;
    double* _devLastAccelerations;

    PartitionParameters _hostPartitionParams;
    PartitionParameters _devPartitionParams;
    ApproachParameters _devApproachParams;
};