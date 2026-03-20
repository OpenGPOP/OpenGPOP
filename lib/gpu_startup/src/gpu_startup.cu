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

#include "gpu_constants/gpu_constants.cuh"
#include "gpu_startup/device/calculate_all_s.cuh"
#include "gpu_startup/device/copy_prev_state.cuh"
#include "gpu_startup/device/device_functions.cuh"
#include "gpu_startup/device/initialize_memory.cuh"
#include "gpu_startup/device/move_state_to_temp.cuh"
#include "gpu_startup/device/reset_temp_vectors.cuh"
#include "gpu_startup/gpu_startup.cuh"

#include "gpu_utils/gpu_utils.cuh"

#include <assert.h>
#include <cublas_v2.h>
#include <fstream>
#include <string>

GpuStartup::GpuStartup(double* epochPositions, double* epochVelocities, uint numSatellites,
    cublasHandle_t* cublasHandle, double timeStep, IForceModel& forceModel, int maxPropagationSteps,
    int numCorrections)
    : _devPositions(epochPositions)
    , _devVelocities(epochVelocities)
    , _numSatellites(numSatellites)
    , _cublasHandle(cublasHandle)
    , _timeStep(timeStep)
    , _forceModel(forceModel)
    , _numCorrections(numCorrections)
    , _maxPropagationSteps(maxPropagationSteps)
{
    assert(_numSatellites > 0);
    assert(_numCorrections > 0);
    gpuErrchk(cudaMalloc((void**)&_devAccelerationMatrix, 9 * 3 * numSatellites * sizeof(double)));
    forceModel.PreInitialize();
    gpuErrchk(cudaMalloc((void**)&_devTempX, 3 * numSatellites * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&_devTempV, 3 * numSatellites * sizeof(double)));

    // cublasLoggerConfigure(1, 1, 0, nullptr);
}

FullStartupData GpuStartup::Startup()
{
    _forceModel.InitializeGeneration(4, 4);
    Predict();
    StartupCorrector corrector(_cublasHandle, _timeStep, _forceModel, _numSatellites,
        _devPositionMatrix, _devVelocityMatrix, _devAccelerationMatrix, _devPositions,
        _devVelocities);

    for (int i = 0; i < _numCorrections; i++) {
        corrector.Correct();
    }

    return corrector.getData();
}

RK8::RK8(int numSatellites, double* devPositionMatrix, double* devVelocityMatrix, double* devTempX,
    double* devTempV, IForceModel& forceModel, double timeStep, cublasHandle_t* cublasHandle)
    : _numSatellites(numSatellites)
    , _devPositionMatrix(devPositionMatrix)
    , _devVelocityMatrix(devVelocityMatrix)
    , _devTempX(devTempX)
    , _devTempV(devTempV)
    , _forceModel(forceModel)
    , _timeStep(timeStep)
    , _cublasHandle(cublasHandle)
{
    gpuErrchk(cudaMalloc((void**)&_devKx, 10 * 3 * numSatellites * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&_devKv, 10 * 3 * numSatellites * sizeof(double)));

    gpuErrchk(cudaMalloc((void**)&_devRk8aTransposed, 9 * 9 * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&_devRk8b, 10 * sizeof(double)));

    double rk8alinear[9 * 9];
    // TODO: change to __constant__
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            rk8alinear[SQ_C2IDX(i, j, 9)]
                = gpu_constants::RK8a[j][i]; // j,i swapped to transpose rk8a
        }
    }
    // initialize rk8 coefficients
    cublasErrchk(cublasSetMatrix(9, 9, sizeof(double), rk8alinear, 9, _devRk8aTransposed, 9));
    cublasErrchk(cublasSetVector(10, sizeof(double), gpu_constants::RK8b, 1, _devRk8b, 1));
}

RK8::~RK8()
{
    cudaFree(_devKx);
    cudaFree(_devKv);
    cudaFree(_devRk8aTransposed);
    cudaFree(_devRk8b);
}

void RK8::step(int step)
{
    double beta = 1;
    double timeStep = step > 0 ? _timeStep : -_timeStep;
    ResetTempVectorsKernel<<<(_numSatellites + 255) / 256, 256>>>(_devTempX, _devTempV,
        _devPositionMatrix, _devVelocityMatrix, _devKx, _numSatellites, step, 0);
    CopyTempToKKernel<<<(_numSatellites + 255) / 256, 256>>>(_devTempV, _devKx, 0, _numSatellites);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    _forceModel.ComputeGenerationAcceleration(_devTempX, _devTempV, _devKv, step, 0);
    gpuErrchk(cudaDeviceSynchronize());

    for (int k = 1; k <= 9; k++) {
        cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 9, &timeStep, _devKx, 3, 3 * 10,
            _devRk8aTransposed + (k - 1) * 9, 1, 0, &beta, _devTempX, 1, 3, _numSatellites);
        cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 9, &timeStep, _devKv, 3, 3 * 10,
            _devRk8aTransposed + (k - 1) * 9, 1, 0, &beta, _devTempV, 1, 3, _numSatellites);
        _forceModel.ComputeGenerationAcceleration(_devTempX, _devTempV, _devKv, step, k);
        CopyTempToKKernel<<<(_numSatellites + 255) / 256, 256>>>(
            _devTempV, _devKx, k, _numSatellites);
        ResetTempVectorsKernel<<<(_numSatellites + 255) / 256, 256>>>(_devTempX, _devTempV,
            _devPositionMatrix, _devVelocityMatrix, _devKx, _numSatellites, step, k);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    CopyPrevStateKernel<<<(_numSatellites + 255) / 256, 256>>>(
        _devPositionMatrix, _devVelocityMatrix, _numSatellites, step);
    gpuErrchk(cudaGetLastError());

    cublasErrchk(cublasDgemmStridedBatched(*_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 1, 10,
        &timeStep, _devKv, 3, 3 * 10, _devRk8b, 10, 0, &beta, _devVelocityMatrix + 3 * (4 + step),
        3, 3 * 9, _numSatellites));
    cublasErrchk(cublasDgemmStridedBatched(*_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 1, 10,
        &timeStep, _devKx, 3, 3 * 10, _devRk8b, 10, 0, &beta, _devPositionMatrix + 3 * (4 + step),
        3, 3 * 9, _numSatellites));
    gpuErrchk(cudaDeviceSynchronize());
}

StartupPredictionData StartupPredictor::Predict(int numSatellites, double* devPosition,
    double* devVelocity, double* devTempX, double* devTempV, IForceModel& forceModel,
    double timeStep, cublasHandle_t* cublasHandle)
{
    double* devPositionMatrix;
    double* devVelocityMatrix;
    gpuErrchk(cudaMalloc((void**)&devPositionMatrix, 9 * 3 * numSatellites * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&devVelocityMatrix, 9 * 3 * numSatellites * sizeof(double)));

    initialize_memory<<<(numSatellites + 255) / 256, 256>>>(devPosition, devVelocity,
        devPositionMatrix, devVelocityMatrix, devTempX, devTempV, numSatellites);
    gpuErrchk(cudaGetLastError());

    RK8 rk8(numSatellites, devPositionMatrix, devVelocityMatrix, devTempX, devTempV, forceModel,
        timeStep, cublasHandle);

    for (int i = 1; i <= 4; i++) {
        rk8.step(i); // Forward step
        rk8.step(-i); // Backward step
    }

    return StartupPredictionData(devPositionMatrix, devVelocityMatrix);
}

void StartupCorrector::CalculateS0()
{
    // Equations (73), (75) and (86)

    double alpha = -1.0;
    double beta = 0.0;
    double timeStepInv = 1 / _timeStep;
    double timeStepSquaredInv = 1 / (_timeStep * _timeStep);
    cublasErrchk(
        cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 9, &alpha, _devAccelerationMatrix,
            3, 3 * 9, _devB + 4, 10, 0, &beta, _devSmallS + 3 * 4, 1, 3 * 9, _numSatellites));

    cublasErrchk(cublasDaxpy(*_cublasHandle, _numSatellites * 3 * 9, &timeStepInv,
        _devVelocityMatrix, 1, _devSmallS, 1));

    cublasErrchk(
        cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 9, &alpha, _devAccelerationMatrix,
            3, 3 * 9, _devA + 4, 10, 0, &beta, _devBigS + 3 * 4, 1, 3 * 9, _numSatellites));

    cublasErrchk(cublasDaxpy(*_cublasHandle, _numSatellites * 3 * 9, &timeStepSquaredInv,
        _devPositionMatrix, 1, _devBigS, 1));

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
void StartupCorrector::CalculateAccelerationMatrix(bool calculateForEpoch)
{

    for (int i = 0; i < 4; i++) {
        MoveStateToTempKernel<<<(_numSatellites + 255) / 256, 256>>>(
            _devTempX, _devTempV, _devPositionMatrix, _devVelocityMatrix, i, _numSatellites);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        _forceModel.ComputePropagationAcceleration(
            _devTempX, _devTempV, _devAccelerationMatrix, i, i);
    }
    if (calculateForEpoch) {
        MoveStateToTempKernel<<<(_numSatellites + 255) / 256, 256>>>(
            _devTempX, _devTempV, _devPositionMatrix, _devVelocityMatrix, 4, _numSatellites);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        _forceModel.ComputePropagationAcceleration(
            _devTempX, _devTempV, _devAccelerationMatrix, 4, 4);
    }
    for (int i = 5; i < 9; i++) {
        MoveStateToTempKernel<<<(_numSatellites + 255) / 256, 256>>>(
            _devTempX, _devTempV, _devPositionMatrix, _devVelocityMatrix, i, _numSatellites);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
        _forceModel.ComputePropagationAcceleration(
            _devTempX, _devTempV, _devAccelerationMatrix, i, i);
    }
    gpuErrchk(cudaDeviceSynchronize());
}

StartupCorrector::StartupCorrector(cublasHandle_t* cublasHandle, double timeStep,
    IForceModel& forceModel, int numSatellites, double* devPositionMatrix,
    double* devVelocityMatrix, double* devAccelerationMatrix, double* devTempX, double* devTempV)
    : _cublasHandle(cublasHandle)
    , _timeStep(timeStep)
    , _forceModel(forceModel)
    , _numSatellites(numSatellites)
    , _devPositionMatrix(devPositionMatrix)
    , _devVelocityMatrix(devVelocityMatrix)
    , _devAccelerationMatrix(devAccelerationMatrix)
    , _devTempX(devTempX)
    , _devTempV(devTempV)
{
    gpuErrchk(cudaMalloc((void**)&_devBigS, 3 * 9 * _numSatellites * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&_devSmallS, 3 * 9 * _numSatellites * sizeof(double)));

    gpu_constants::createCoefficients(&_devA, &_devB);

    CalculateAccelerationMatrix(true);
}

void StartupCorrector::Correct()
{
    // Step 3.(a)
    CalculateS0();
    // Step 3.(b).i
    CalculateAllSKernel<<<(_numSatellites + 255) / 256, 256>>>(
        _devBigS, _devSmallS, _devAccelerationMatrix, _numSatellites);
    gpuErrchk(cudaGetLastError());

    // Move epoch state as not to override it
    MoveStateToTempKernel<<<(_numSatellites + 255) / 256, 256>>>(
        _devTempX, _devTempV, _devPositionMatrix, _devVelocityMatrix, 4, _numSatellites);
    gpuErrchk(cudaGetLastError());
    double alpha = 1.0;
    double beta = 0.0;

    // Calculate sum from 3.(b).ii
    cublasErrchk(cublasDgemmStridedBatched(*_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 3, 9, 9,
        &alpha, _devAccelerationMatrix, 3, 3 * 9, _devB, 10, 0, &beta, _devVelocityMatrix, 3, 3 * 9,
        _numSatellites));
    // Calculate sum from 3.(b).iii
    cublasErrchk(cublasDgemmStridedBatched(*_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 3, 9, 9,
        &alpha, _devAccelerationMatrix, 3, 3 * 9, _devA, 10, 0, &beta, _devPositionMatrix, 3, 3 * 9,
        _numSatellites));
    // Step 3.(b).iv equation (74)
    // Add small s to sum from ii
    cublasErrchk(cublasDaxpy(
        *_cublasHandle, 3 * 9 * _numSatellites, &alpha, _devSmallS, 1, _devVelocityMatrix, 1));
    // Multiply by h
    cublasErrchk(
        cublasDscal(*_cublasHandle, 3 * 9 * _numSatellites, &_timeStep, _devVelocityMatrix, 1));
    // Step 3.(b).iv equation (87)
    // Add big S to sum from iii
    cublasErrchk(cublasDaxpy(
        *_cublasHandle, 3 * 9 * _numSatellites, &alpha, _devBigS, 1, _devPositionMatrix, 1));
    // Multiply by h^2
    double timeStepSquared = _timeStep * _timeStep;
    cublasErrchk(cublasDscal(
        *_cublasHandle, 3 * 9 * _numSatellites, &timeStepSquared, _devPositionMatrix, 1));

    // Move epoch state back to matrices
    MoveStateToMatrixKernel<<<(_numSatellites + 255) / 256, 256>>>(
        _devTempX, _devTempV, _devPositionMatrix, _devVelocityMatrix, 4, _numSatellites);
    gpuErrchk(cudaDeviceSynchronize());

    // Calculate all accelerations for updated states
    CalculateAccelerationMatrix(false);
}
FullStartupData StartupCorrector::getData()
{
    return FullStartupData { _devBigS, _devSmallS, _devAccelerationMatrix, _devPositionMatrix,
        _devVelocityMatrix, _devA, _devB };
}

void GpuStartup::Predict()
{
    StartupPredictor predictor;
    StartupPredictionData prediction = predictor.Predict(_numSatellites, _devPositions,
        _devVelocities, _devTempX, _devTempV, _forceModel, _timeStep, _cublasHandle);

    _devPositionMatrix = prediction.devPositionMatrix;
    _devVelocityMatrix = prediction.devVelocityMatrix;

    // TODO: add correct afterEpoch
    _forceModel.InitializePropagation(4, _maxPropagationSteps);
}

GpuStartup::~GpuStartup()
{
    cudaFree(_devTempX);
    cudaFree(_devTempV);
}