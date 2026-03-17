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
