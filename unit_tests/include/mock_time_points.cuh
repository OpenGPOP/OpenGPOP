#pragma once

#include "force_model/time_points/itime_points.cuh"
#include "force_model/types.cuh"
#include <gmock/gmock.h>

class MockFMTimePoints : public IFMTimePoints {
public:
    MOCK_METHOD(void, Initialize, (), (override));
    MOCK_METHOD(void, CalculateParameters, (int arrayLength, bool useMidpoints, int beforeEpoch),
        (override));
    MOCK_METHOD(TimePointParameters, GetParametersDev, (int stepNum), (override));
};