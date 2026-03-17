#pragma once
#include "cublas_v2.h"
#include "force_model/time_points/itime_points.cuh"
#include "force_model/types.cuh"

__global__ void TimePointParamsKernel(TimePointDatasets* datasets,
    TimePointTempParameters* tempParams, StartTime startTime, double stepSize, int arrayLength,
    bool useMidpoints, int beforeEpoch);

// This class is responsible for calculating and providing simulation
// parameters that depend only on time
class FMTimePoints : public IFMTimePoints {
public:
    FMTimePoints(cublasHandle_t* cublasHandle, double stepSize, StartTime startTime)
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

private:
    cublasHandle_t* _cublasHandle;
    double _stepSize;
    StartTime _startTime;

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
