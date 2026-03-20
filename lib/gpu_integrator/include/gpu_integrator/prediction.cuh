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

#include <cublas_v2.h>

class Predictor {
    double* _devPositionVector;
    double* _devVelocityVector;
    double* _devAccelerationMatrix;
    double* _devA;
    double* _devB;
    double* _devSmallSVector;
    double* _devBigSVector;

    IForceModel& _forceModel;
    cublasHandle_t* _cublasHandle;
    int _numSatellites;
    double _stepSize;

public:
    Predictor(
        IForceModel& forceModel, cublasHandle_t* cublasHandle, int numSatellites, double stepSize);

    void SetPointers(double* devPositionVector, double* devVelocityVector,
        double* devAccelerationMatrix, double* devA, double* devB, double* devSmallSVector,
        double* devBigSVector);

    void Predict(int stepNum);
};
