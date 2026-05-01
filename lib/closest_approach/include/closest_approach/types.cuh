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

#include "closest_approach/utility/partition_parameters.cuh"
#include "gpu_utils/gpu_utils.cuh"

struct Arrays3D {
    double* x;
    double* y;
    double* z;

    void cudaAllocate(size_t memSize)
    {
        gpuErrchk(cudaMalloc(&x, memSize));
        gpuErrchk(cudaMalloc(&y, memSize));
        gpuErrchk(cudaMalloc(&z, memSize));
    }

    void cudaFree()
    {
        CUDA_SAFE_FREE(x);
        CUDA_SAFE_FREE(y);
        CUDA_SAFE_FREE(z);
    }
};

struct ObjectData {
    Arrays3D position;
    Arrays3D velocity;
    Arrays3D acceleration;

    void cudaAllocate(size_t memSize)
    {
        position.cudaAllocate(memSize);
        velocity.cudaAllocate(memSize);
        acceleration.cudaAllocate(memSize);
    }

    void cudaFree()
    {
        position.cudaFree();
        velocity.cudaFree();
        acceleration.cudaFree();
    }
};

struct Objects {
    ObjectData o1;
    ObjectData o2;

    void cudaAllocate(size_t memSize)
    {
        o1.cudaAllocate(memSize);
        o2.cudaAllocate(memSize);
    }

    void cudaFree()
    {
        o1.cudaFree();
        o2.cudaFree();
    }
};

struct ApproachParameters {
    unsigned int* closestSteps;
    double* closestDistances;

    double* closestPositionsA;
    double* closestPositionsB;

    Objects closest;
    Objects before;
    Objects after;

    void cudaAllocate(unsigned pairNum)
    {
        gpuErrchk(cudaMalloc(&closestDistances, sizeof(double) * pairNum));
        gpuErrchk(cudaMalloc(&closestSteps, sizeof(int) * pairNum));
        gpuErrchk(cudaMalloc(&closestPositionsA, sizeof(double) * pairNum * 3));
        gpuErrchk(cudaMalloc(&closestPositionsB, sizeof(double) * pairNum * 3));
        closest.cudaAllocate(sizeof(double) * pairNum);
        before.cudaAllocate(sizeof(double) * pairNum);
        after.cudaAllocate(sizeof(double) * pairNum);
    }

    void cudaFree()
    {
        CUDA_SAFE_FREE(closestDistances);
        CUDA_SAFE_FREE(closestSteps);
        CUDA_SAFE_FREE(closestPositionsA);
        CUDA_SAFE_FREE(closestPositionsB);
        closest.cudaFree();
        before.cudaFree();
        after.cudaFree();
    }
};

struct Approach {
    double distance;
    double time;

    double positionA[3];
    double positionB[3];
};

class ClosestApproachData {
    friend class ClosestApproach;

    double* _distances = nullptr;
    double* _times = nullptr;
    double* _positionsA = nullptr;
    double* _positionsB = nullptr;
    PartitionParameters _partParams;

    inline void Populate(PartitionParameters partParams, double* distances, double* times,
        double* positionsA, double* positionsB)
    {
        Free();
        _partParams = partParams;
        _distances = distances;
        _times = times;
        _positionsA = positionsA;
        _positionsB = positionsB;
    }

    void Free()
    {
        delete[] _distances;
        delete[] _times;
        delete[] _positionsA;
        delete[] _positionsB;
        _partParams.hostFree();
    }

public:
    // Returns the closest approach between objA and objB. If the pair is invalid (was not
    // included in the partition) or the structure was not populated, { -1.0, -1.0 } is returned.
    inline Approach Get(unsigned objA, unsigned objB)
    {
        if (_distances == nullptr || _times == nullptr) {
            return { -1.0, -1.0, { -1.0, -1.0, -1.0 }, { -1.0, -1.0, -1.0 } }; // Not populated
        }
        if (objA == objB) {
            return { 0.0, 0.0, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };
        }
        auto idx = _partParams.GetPartitionMapIdx(objA, objB);
        if (idx < 0) {
            return { -1.0, -1.0, { -1.0, -1.0, -1.0 }, { -1.0, -1.0, -1.0 } }; // Invalid pair
        }
        auto posA = &_positionsA[idx * 3];
        auto posB = &_positionsB[idx * 3];
        return { _distances[idx], _times[idx], { posA[0], posA[1], posA[2] },
            { posB[0], posB[1], posB[2] } };
    }

    inline ~ClosestApproachData() { Free(); }
};