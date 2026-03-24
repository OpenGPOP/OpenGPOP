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

#pragma once

#include "closest_approach/types.cuh"
#include "closest_approach/utility/partition_parameters.cuh"
#include "cublas_v2.h"
#include "force_model/iforce_model.cuh"
#include "types/Parameters.hpp"
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

__global__ void UpdateClosestApproachKernel(PartitionParameters partitionParams, double* pos,
    double* vel, double* acc, double* lastPos, double* lastVel, double* lastAcc,
    ApproachParameters ap, unsigned stepNum);

__global__ void CalculateClosestApproachesKernel(PartitionParameters partParams,
    ApproachParameters appParams, double stepSize, unsigned totalSteps, double* closestTimes);

class ClosestApproach {
public:
    ClosestApproach(cublasHandle_t* cublasHandle, FullPerturbationParameters* params,
        IForceModel& forceModel, double stepSize)
        : _cublasHandle(cublasHandle)
        , _params(params)
        , _stepSize(stepSize)
        , _forceModel(forceModel)
    {
        assert(stepSize > 0);
    }
    ~ClosestApproach();

    // Initializes simulation parameters and allocates memory on the device. The positions and
    // velocities vectors must be of the same size. The setIndices vector must be of size equal to
    // the number of entities. The indices must start from 0 and the difference between consecutive
    // indices cannot exceed 1.
    void Initialize(const std::vector<std::array<double, 3>>& positions,
        const std::vector<std::array<double, 3>>& velocities,
        const std::vector<unsigned>& setIndices);

    // Propagates the objects for the specified number of steps with the given step size. On
    // subsequent calls, the propagation will continue from the last state.
    void Propagate(int numSteps, bool verbose = false);

    // The function finds the number of satellites that are closer than the given distance. If
    // hostClosestDistances is provided, it will be filled with the distances. When called after
    // multiple propagation phases, the closest distances across all phases will be returned.
    void FindClosest(ClosestApproachData& closestApproaches);

private:
    cublasHandle_t* _cublasHandle;
    FullPerturbationParameters* _params;

    double _stepSize = 0;
    double _totalNumSteps = 0;
    double _lastNumSteps = 0;

    unsigned _numEntities = 0;
    double* _devPositions;
    double* _devVelocities;
    double* _devAccelerations;

    double* _devLastPositions;
    double* _devLastVelocities;
    double* _devLastAccelerations;

    IForceModel& _forceModel;

    PartitionParameters _hostPartitionParams;
    PartitionParameters _devPartitionParams;
    ApproachParameters _devApproachParams;
};