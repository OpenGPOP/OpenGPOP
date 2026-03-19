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

    Objects closest;
    Objects before;
    Objects after;

    void cudaAllocate(unsigned pairNum)
    {
        gpuErrchk(cudaMalloc(&closestDistances, sizeof(double) * pairNum));
        gpuErrchk(cudaMalloc(&closestSteps, sizeof(int) * pairNum));
        closest.cudaAllocate(sizeof(double) * pairNum);
        before.cudaAllocate(sizeof(double) * pairNum);
        after.cudaAllocate(sizeof(double) * pairNum);
    }

    void cudaFree()
    {
        CUDA_SAFE_FREE(closestDistances);
        CUDA_SAFE_FREE(closestSteps);
        closest.cudaFree();
        before.cudaFree();
        after.cudaFree();
    }
};

struct Approach {
    double distance;
    double time;
};

class ClosestApproachData {
    friend class ClosestApproach;

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
    inline Approach Get(unsigned objA, unsigned objB)
    {
        if (_distances == nullptr || _times == nullptr) {
            return { -1.0, -1.0 }; // Not populated
        }
        if (objA == objB) {
            return { 0.0, 0.0 };
        }
        auto idx = _partParams.GetPartitionMapIdx(objA, objB);
        if (idx < 0) {
            return { -1.0, -1.0 }; // Invalid pair
        }
        return { _distances[idx], _times[idx] };
    }

    inline ~ClosestApproachData() { Free(); }
};