#pragma once

#include "monte_carlo/types.cuh"
#include <cstring>
#include <vector>


__host__ __device__ int GetPartitionMapIdx(PartitionParameters params, unsigned idxA, unsigned idxB)
{
    if (idxA > idxB) {
        auto tmp = idxA;
        idxA = idxB;
        idxB = tmp;
    }

    auto setA = params.setIds[idxA];
    if (setA == params.setNum - 1)
        return -1;

    int offset1 = (int)idxB - (int)params.setOffsets[setA + 1];
    if (offset1 < 0)
        return -1;

    auto offset2 = params.mapOffsets[setA];
    auto offset3 = (idxA - params.setOffsets[setA]) * params.numHigherIds[setA];
    return offset1 + offset2 + offset3;
}

__host__ __device__ bool GetPartitionMapPair(
    PartitionParameters params, unsigned idx, unsigned& idxA, unsigned& idxB)
{
    // Find the A set (it cannot be the last one)
    unsigned setA = 0;
    while (setA < params.setNum - 1 && params.mapOffsets[setA + 1] <= idx)
        setA++;
    if (setA >= params.setNum - 1) {
        return false;
    }

    auto mapOffset = params.mapOffsets[setA];
    auto numHigherIds = params.numHigherIds[setA];

    idxA = (idx - mapOffset) / numHigherIds;
    idxA += params.setOffsets[setA];

    idxB = (idx - mapOffset) % numHigherIds;
    idxB += params.setOffsets[setA + 1];
    return true;
}

// Calculates the PartitionParameters structure and fills it, apart from the setIds array.
bool CalculatePartitionParameters(std::vector<unsigned> setIds, PartitionParameters& params)
{
    std::vector<unsigned> setOffsets;
    std::vector<unsigned> sizes;
    std::vector<unsigned> numHigherIds;
    std::vector<unsigned> mapOffsets;

    setOffsets.push_back(0);
    mapOffsets.push_back(0);

    unsigned prevId = setIds[0];
    unsigned prevChangeIdx = 0;

    for (unsigned i = 1; i < setIds.size(); i++) {
        if (setIds[i] < prevId || setIds[i] > prevId + 1) {
            return false;
        }
        if (setIds[i] > prevId) {
            setOffsets.push_back(i);
            sizes.push_back(i - prevChangeIdx);
            numHigherIds.push_back(setIds.size() - i);
            mapOffsets.push_back(mapOffsets.back() + sizes.back() * numHigherIds.back());
            prevChangeIdx = i;
        }
        prevId = setIds[i];
    }

    sizes.push_back(setIds.size() - prevChangeIdx);
    numHigherIds.push_back(0);

    params.setNum = static_cast<unsigned>(setOffsets.size());
    params.pairNum = mapOffsets.back();
    params.setOffsets = new unsigned[params.setNum];
    params.numHigherIds = new unsigned[params.setNum];
    params.mapOffsets = new unsigned[params.setNum];
    std::memcpy(params.setOffsets, setOffsets.data(), params.setNum * sizeof(unsigned));
    std::memcpy(params.numHigherIds, numHigherIds.data(), params.setNum * sizeof(unsigned));
    std::memcpy(params.mapOffsets, mapOffsets.data(), params.setNum * sizeof(unsigned));
    return true;
}