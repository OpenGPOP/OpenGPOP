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
#include "force_model/types.cuh"

// Responsible for calculating and providing simulation parameters that depend only on time
struct IFMTimePoints {
    virtual ~IFMTimePoints() = default;

    virtual void Initialize() = 0;
    virtual void CalculateParameters(int arrayLength, bool useMidpoints, int beforeEpoch) = 0;

    // Returns a structure of device pointers shifted to the requested position
    virtual TimePointParameters GetParametersDev(int stepNum) = 0;

    virtual TimeData GetStartTime() = 0;
};