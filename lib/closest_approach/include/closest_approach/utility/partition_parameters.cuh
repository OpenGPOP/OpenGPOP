#pragma once

#include <cstring>
#include <vector>

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

    inline __host__ __device__ int GetPartitionMapIdx(unsigned idxA, unsigned idxB)
    {
        if (idxA > idxB) {
            auto tmp = idxA;
            idxA = idxB;
            idxB = tmp;
        }

        auto setA = setIds[idxA];
        if (setA == setNum - 1)
            return -1;

        int offset1 = (int)idxB - (int)setOffsets[setA + 1];
        if (offset1 < 0)
            return -1;

        auto offset2 = mapOffsets[setA];
        auto offset3 = (idxA - setOffsets[setA]) * numHigherIds[setA];
        return offset1 + offset2 + offset3;
    }

    inline __host__ __device__ bool GetPartitionMapPair(
        unsigned idx, unsigned& idxA, unsigned& idxB)
    {
        // Find the A set (it cannot be the last one)
        unsigned setA = 0;
        while (setA < setNum - 1 && mapOffsets[setA + 1] <= idx)
            setA++;
        if (setA >= setNum - 1) {
            return false;
        }

        auto mapOffset = mapOffsets[setA];
        auto higherIds = numHigherIds[setA];

        idxA = (idx - mapOffset) / higherIds;
        idxA += setOffsets[setA];

        idxB = (idx - mapOffset) % higherIds;
        idxB += setOffsets[setA + 1];
        return true;
    }

    // Calculates the PartitionParameters structure and fills it, apart from the setIds array.
    inline __host__ bool Calculate(std::vector<unsigned> setIds)
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

        this->setNum = static_cast<unsigned>(setOffsets.size());
        this->pairNum = mapOffsets.back();
        this->setIds = new unsigned[setIds.size()];
        this->setOffsets = new unsigned[this->setNum];
        this->numHigherIds = new unsigned[this->setNum];
        this->mapOffsets = new unsigned[this->setNum];
        std::memcpy(this->setIds, setIds.data(), setIds.size() * sizeof(unsigned));
        std::memcpy(this->setOffsets, setOffsets.data(), this->setNum * sizeof(unsigned));
        std::memcpy(this->numHigherIds, numHigherIds.data(), this->setNum * sizeof(unsigned));
        std::memcpy(this->mapOffsets, mapOffsets.data(), this->setNum * sizeof(unsigned));
        return true;
    }

    void hostFree()
    {
        delete[] setIds;
        delete[] setOffsets;
        delete[] numHigherIds;
        delete[] mapOffsets;
    }
};