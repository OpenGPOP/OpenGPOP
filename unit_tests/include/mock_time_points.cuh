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

#include "force_model/time_points/itime_points.cuh"
#include "force_model/types.cuh"
#include "types/Parameters.hpp"
#include <gmock/gmock.h>

class MockFMTimePoints : public IFMTimePoints {
public:
    MOCK_METHOD(void, Initialize, (), (override));
    MOCK_METHOD(void, CalculateParameters, (int arrayLength, bool useMidpoints, int beforeEpoch),
        (override));
    MOCK_METHOD(TimePointParametersSlice, GetParametersDev, (int stepNum), (override));
    MOCK_METHOD(TimeData, GetStartTime, (), (override));
};