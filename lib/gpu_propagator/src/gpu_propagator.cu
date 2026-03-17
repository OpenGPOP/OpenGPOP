#include "gpu_propagator/device/copy_s.cuh"
#include "gpu_propagator/device/dev_functions.cuh"
#include "gpu_propagator/device/update_smalls.cuh"
#include "gpu_propagator/gpu_propagator.cuh"

#include "gpu_utils/gpu_utils.cuh"

#include <codecvt>
#include <cstdlib>
#include <cublas_v2.h>

void GpuPropagator::Step(
    double* hostPositions, double* hostVelocities, double* devAccelerationVector)
{
    if (_stepNumber == 1) {
        _startupData = _startup.Startup();
        gpuErrchk(cudaMalloc((void**)&_smallS, _numSatellites * 3 * sizeof(double)));
        gpuErrchk(cudaMalloc((void**)&_bigS, _numSatellites * 3 * sizeof(double)));
        gpuErrchk(cudaMalloc((void**)&_prevSmallS, _numSatellites * 3 * sizeof(double)));
        gpuErrchk(cudaMalloc((void**)&_tempSum1, _numSatellites * 3 * sizeof(double)));
        gpuErrchk(cudaMalloc((void**)&_tempSum2, _numSatellites * 3 * sizeof(double)));

        CopySKernel<<<(_numSatellites + 255) / 256, 256>>>(
            _smallS, _bigS, _startupData.devSmallS, _startupData.devBigS, _numSatellites);
        gpuErrchk(cudaGetLastError());

        _predictor.SetPointers(_positions, _velocities, _startupData.devAccelerationMatrix,
            _startupData.devA, _startupData.devB, _smallS, _bigS);

        gpuErrchk(cudaDeviceSynchronize());
    }
    if (_stepNumber <= 4) {
        CopyStartupStateKernel<<<(_numSatellites + 255) / 256, 256>>>(_positions, _velocities,
            _startupData.devPositionMatrix, _startupData.devVelocityMatrix, _stepNumber + 4,
            _numSatellites);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    } else {
        _predictor.Predict(_stepNumber);
        Correct();
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    if (hostPositions != nullptr) {
        gpuErrchk(cudaMemcpy(hostPositions, _positions, _numSatellites * 3 * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (hostVelocities != nullptr) {
        gpuErrchk(cudaMemcpy(hostVelocities, _velocities, _numSatellites * 3 * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    if (devAccelerationVector != nullptr) {

        MoveColumnToVectorKernel<<<(_numSatellites + 255) / 256, 256>>>(
            _startupData.devAccelerationMatrix, devAccelerationVector, 8, _numSatellites);
    }

    gpuErrchk(cudaDeviceSynchronize());

    _stepNumber++;
}

void GpuPropagator::Correct()
{
    // Copy previous small s
    cublasErrchk(cublasDcopy(*_cublasHandle, 3 * _numSatellites, _smallS, 1, _prevSmallS, 1));
    double one = 1.0;
    double zero = 0.0;
    // Calculate sums without last term
    // Step 10.(b).i
    cublasErrchk(cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 8, &one,
        _startupData.devAccelerationMatrix, 3, 3 * 9, _startupData.devB + 8, 10, 0, &zero,
        _tempSum1, 1, 3, _numSatellites));
    // Step 10.(b).ii
    cublasErrchk(cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 8, &one,
        _startupData.devAccelerationMatrix, 3, 3 * 9, _startupData.devA + 8, 10, 0, &zero,
        _tempSum2, 1, 3, _numSatellites));

    for (int i = 0; i < _numCorrections; i++) {
        _forceModel.ComputePropagationAcceleration(
            _positions, _velocities, _startupData.devAccelerationMatrix, _stepNumber + 4, 8);
        UpdateSmallSKernel<<<(_numSatellites + 255) / 256, 256>>>(
            _smallS, _startupData.devAccelerationMatrix, _prevSmallS, _numSatellites);
        gpuErrchk(cudaGetLastError());

        // Move part of the sum
        cublasErrchk(cublasDcopy(*_cublasHandle, 3 * _numSatellites, _tempSum1, 1, _velocities, 1));
        cublasErrchk(cublasDcopy(*_cublasHandle, 3 * _numSatellites, _tempSum2, 1, _positions, 1));

        // Step 10.(d)
        FinalizeCorrectionKernel<<<(_numSatellites + 255) / 256, 256>>>(_positions, _velocities,
            _startupData.devAccelerationMatrix, _smallS, _bigS, _startupData.devA,
            _startupData.devB, _timeStep, _numSatellites);
        gpuErrchk(cudaGetLastError());
    }
}

GpuPropagator::GpuPropagator(double* epochPositions, double* epochVelocities, uint numSatellites,
    cublasHandle_t* cublasHandle, double timeStep, FullPerturbationParameters* params,
    IForceModel& forceModel, int maxSteps, int numCorrections)
    : _startup(epochPositions, epochVelocities, numSatellites, cublasHandle, timeStep, forceModel,
          maxSteps, numCorrections)
    , _predictor(forceModel, cublasHandle, numSatellites, timeStep)
    , _numSatellites(numSatellites)
    , _positions(epochPositions)
    , _velocities(epochVelocities)
    , _timeStep(timeStep)
    , _forceModel(forceModel)
    , _forceModelParams(params)
    , _cublasHandle(cublasHandle)
    , _numCorrections(numCorrections)
{
}

GpuPropagator::~GpuPropagator() { FreeAllMemory(); }

void GpuPropagator::FreeAllMemory()
{
    gpuErrchk(cudaFree((void*)_smallS));
    gpuErrchk(cudaFree((void*)_tempSum1));
    gpuErrchk(cudaFree((void*)_tempSum2));
    gpuErrchk(cudaFree((void*)_prevSmallS));
    gpuErrchk(cudaFree((void*)_bigS));
    gpuErrchk(cudaFree((void*)_startupData.devBigS));
    gpuErrchk(cudaFree((void*)_startupData.devSmallS));
    gpuErrchk(cudaFree((void*)_startupData.devAccelerationMatrix));
    gpuErrchk(cudaFree((void*)_startupData.devPositionMatrix));
    gpuErrchk(cudaFree((void*)_startupData.devVelocityMatrix));
    gpuErrchk(cudaFree((void*)_startupData.devA));
    gpuErrchk(cudaFree((void*)_startupData.devB));
}
