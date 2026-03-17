#pragma once

#include "../../../gpu_integrator/include/gpu_integrator/prediction.cuh"
#include "force_model/iforce_model.cuh"
#include "gpu_startup/gpu_startup.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "types/Parameters.hpp"
#include <cublas_v2.h>

class GpuPropagator {
private:
    GpuStartup _startup;
    Predictor _predictor;
    FullStartupData _startupData;
    uint _numSatellites;
    double* _positions;
    double* _velocities;
    double* _smallS = nullptr;
    double* _prevSmallS = nullptr;
    double* _bigS = nullptr;
    uint _stepNumber = 1;
    double _timeStep;
    IForceModel& _forceModel;
    FullPerturbationParameters* _forceModelParams;
    cublasHandle_t* _cublasHandle;
    int _numCorrections;
    double* _tempSum1;
    double* _tempSum2;
    int _maxSteps;

    void Correct();

public:
    GpuPropagator(double* epochPositions, double* epochVelocities, uint numSatellites,
        cublasHandle_t* cublasHandle, double timeStep, FullPerturbationParameters* params,
        IForceModel& forceModel, int maxSteps, int numCorrections = 7);

    void Step(double* hostPositions = nullptr, double* hostVelocities = nullptr,
        double* devAccelerationVector = nullptr);
    void FreeAllMemory();
    ~GpuPropagator();
};
