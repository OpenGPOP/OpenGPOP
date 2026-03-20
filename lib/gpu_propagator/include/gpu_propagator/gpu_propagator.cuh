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

#include "force_model/iforce_model.cuh"
#include "gpu_integrator/prediction.cuh"
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
