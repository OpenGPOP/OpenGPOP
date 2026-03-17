#pragma once

#include "force_model/iforce_model.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>

struct FullStartupData {
    double* devBigS;
    double* devSmallS;
    double* devAccelerationMatrix;
    double* devPositionMatrix;
    double* devVelocityMatrix;
    double* devA;
    double* devB;
};

struct StartupPredictionData {
    double* devPositionMatrix;
    double* devVelocityMatrix;
};

class RK8 {
    int _numSatellites;
    double* _devPositionMatrix;
    double* _devVelocityMatrix;
    double* _devKx;
    double* _devKv;
    double* _devTempX;
    double* _devTempV;
    double* _devRk8aTransposed;
    double* _devRk8b;
    IForceModel& _forceModel;
    double _timeStep;
    cublasHandle_t* _cublasHandle;

public:
    RK8(int numSatellites, double* devPositionMatrix, double* devVelocityMatrix, double* devTempX,
        double* devTempV, IForceModel& forceModel, double timeStep, cublasHandle_t* cublasHandle);

    ~RK8();

    void step(int step);
};

class StartupPredictor {

public:
    StartupPredictionData Predict(int numSatellites, double* devPosition, double* devVelocity,
        double* devTempX, double* devTempV, IForceModel& forceModel, double timeStep,
        cublasHandle_t* cublasHandle);
};

class StartupCorrector {
    cublasHandle_t* _cublasHandle;
    double _timeStep;
    IForceModel& _forceModel;
    int _numSatellites;
    double* _devPositionMatrix;
    double* _devVelocityMatrix;
    double* _devAccelerationMatrix;
    double* _devA; // Eighth-Order Gauss-Jackson Coefficients in Ordinate Form (table 6)
    double* _devB; // Eighth-Order Summed-Adams Coefficients in Ordinate Form (table 5)
    double* _devTempX;
    double* _devTempV;
    double* _devSmallS;
    double* _devBigS;

public:
    void CalculateS0();
    void CalculateAccelerationMatrix(bool calculateForEpoch);
    StartupCorrector(cublasHandle_t* cublasHandle, double timeStep, IForceModel& forceModel,
        int numSatellites, double* devPositionMatrix, double* devVelocityMatrix,
        double* devAccelerationMatrix, double* devTempX, double* devTempV);
    void Correct();
    FullStartupData getData();
};

class GpuStartup {
    double* _devPositions;
    double* _devVelocities;
    double* _devPositionMatrix;
    double* _devVelocityMatrix;
    double* _devTempX;
    double* _devTempV;
    double _timeStep;
    uint _numSatellites;
    cublasHandle_t* _cublasHandle;
    IForceModel& _forceModel;
    double* _devAccelerationMatrix;
    int _numCorrections;
    int _maxPropagationSteps;

    void Predict();

public:
    // Default value came to me in my dream
    GpuStartup(double* epochPositions, double* epochVelocities, uint numSatellites,
        cublasHandle_t* cublasHandle, double timeStep, IForceModel& forceModel,
        int maxPropagationSteps, int numCorrections = 7 /* must be at least one*/);

    FullStartupData Startup();
    ~GpuStartup();
};
