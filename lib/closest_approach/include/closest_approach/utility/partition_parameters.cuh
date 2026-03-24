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

#include <cstring>
#include <cuda_runtime.h>
#include <vector>

struct PartitionParameters {
    unsigned numSets;
    unsigned numPairs;

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
        if (setA == numSets - 1)
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
        while (setA < numSets - 1 && mapOffsets[setA + 1] <= idx)
            setA++;
        if (setA >= numSets - 1) {
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

        this->numSets = static_cast<unsigned>(setOffsets.size());
        this->numPairs = mapOffsets.back();
        this->setIds = new unsigned[setIds.size()];
        this->setOffsets = new unsigned[this->numSets];
        this->numHigherIds = new unsigned[this->numSets];
        this->mapOffsets = new unsigned[this->numSets];
        std::memcpy(this->setIds, setIds.data(), setIds.size() * sizeof(unsigned));
        std::memcpy(this->setOffsets, setOffsets.data(), this->numSets * sizeof(unsigned));
        std::memcpy(this->numHigherIds, numHigherIds.data(), this->numSets * sizeof(unsigned));
        std::memcpy(this->mapOffsets, mapOffsets.data(), this->numSets * sizeof(unsigned));
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