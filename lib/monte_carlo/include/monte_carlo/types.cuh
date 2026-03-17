#pragma once

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
        closest.cudaFree();
        before.cudaFree();
        after.cudaFree();
    }
};

struct PartitionParameters {
    unsigned setNum; // Number of sets
    unsigned pairNum; // Number of pairs

    unsigned* setIds = nullptr;

    // Offsets of the beginnings of each set in the setIds array
    // e.g.
    // setIds     = [ 0 0 0 1 1 2 2 2 3 3 ]
    // setOffsets = [ 0 3 5 8 ]
    unsigned* setOffsets = nullptr;

    // The number of elements with higher setIds
    // e.g.
    // setIds       = [ 0 0 0 1 1 2 2 2 3 3 ]
    // numHigherIds = [ 7 5 2 0 ]
    unsigned* numHigherIds = nullptr;

    // Offsets of the beginnings of each set in the partition map array.
    // Equivalent to exclusive cummulation of setSize[set] * numHigherIds[set]
    // e.g.
    // setIds       = [ 0 0 0 1 1 2 2 2 3 3 ]
    // setSize      = [ 3  2  3  2  ]
    // numHigherIds = [ 7  5  2  0  ]
    // mapOffsets   = [ 0  21 31 37 ]
    unsigned* mapOffsets = nullptr;
};