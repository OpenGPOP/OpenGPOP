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
#include "cublas_v2.h"
#include "force_model/time_points/itime_points.cuh"
#include "force_model/types.cuh"

__global__ void TimePointParamsKernel(TimePointDatasets* datasets,
    TimePointTempParameters* tempParams, TimeData startTime, double stepSize, int arrayLength,
    bool useMidpoints, int beforeEpoch, TimePointParameters tpp);

// This class is responsible for calculating and providing simulation
// parameters that depend only on time
class FMTimePoints : public IFMTimePoints {
public:
    FMTimePoints(cublasHandle_t* cublasHandle, double stepSize, TimeData startTime)
        : _cublasHandle(cublasHandle)
        , _stepSize(stepSize)
        , _startTime(startTime)
    {
    }
    ~FMTimePoints()
    {
        FreeDatasets();
        FreeParameters();
        FreeTempParameters();
    }

    virtual void Initialize() override;
    virtual void CalculateParameters(int arrayLength, bool useMidpoints, int beforeEpoch) override;
    virtual TimePointParameters GetParametersDev(int stepNum) override;
    virtual TimeData GetStartTime() override;

private:
    cublasHandle_t* _cublasHandle;
    double _stepSize;
    TimeData _startTime;

    TimePointDatasets _datasets;
    TimePointDatasets* _devDatasets = nullptr;
    void FreeDatasets();

    TimePointParameters _timePointParameters;
    TimePointParameters* _devTimePointParameters = nullptr;
    void AllocateParameters(int arrayLength);
    void FreeParameters();

    TimePointTempParameters _tempInitParameters;
    TimePointTempParameters* _devTempInitParameters = nullptr;
    void AllocateTempParameters(int arrayLength);
    void FreeTempParameters();

    void MultiplyMatrices(int arrayLength, double* A, double* B, double* out);
    void ComputeITRStoGCRS(int arrayLength);
};
