#include "gpu_integrator/prediction.cuh"

#include "../include/gpu_integrator/device/finalize_prediction.cuh"
#include "../include/gpu_integrator/device/move_accelerations.cuh"
#include "gpu_utils/gpu_utils.cuh"

Predictor::Predictor(
    IForceModel& forceModel, cublasHandle_t* cublasHandle, int numSatellites, double stepSize)
    : _devPositionVector(nullptr)
    , _devVelocityVector(nullptr)
    , _devAccelerationMatrix(nullptr)
    , _devA(nullptr)
    , _devB(nullptr)
    , _devSmallSVector(nullptr)
    , _devBigSVector(nullptr)
    , _forceModel(forceModel)
    , _cublasHandle(cublasHandle)
    , _numSatellites(numSatellites)
    , _stepSize(stepSize)
{
}
void Predictor::SetPointers(double* devPositionVector, double* devVelocityVector,
    double* devAccelerationMatrix, double* devA, double* devB, double* devSmallSVector,
    double* devBigSVector)
{
    _devPositionVector = devPositionVector;
    _devVelocityVector = devVelocityVector;
    _devAccelerationMatrix = devAccelerationMatrix;
    _devA = devA;
    _devB = devB;
    _devSmallSVector = devSmallSVector;
    _devBigSVector = devBigSVector;
}
void Predictor::Predict(int stepNum)
{
    double one = 1.0;
    double beta = 0.0;
    cublasErrchk(
        cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 9, &one, _devAccelerationMatrix,
            3, 3 * 9, _devB + 9, 10, 0, &beta, _devVelocityVector, 1, 3, _numSatellites));

    cublasErrchk(
        cublasDgemvStridedBatched(*_cublasHandle, CUBLAS_OP_N, 3, 9, &one, _devAccelerationMatrix,
            3, 3 * 9, _devA + 9, 10, 0, &beta, _devPositionVector, 1, 3, _numSatellites));

    FinalizePredictionKernel<<<(_numSatellites + 255) / 256, 256>>>(_devPositionVector,
        _devVelocityVector, _devAccelerationMatrix, _devSmallSVector, _devBigSVector,
        _numSatellites, _stepSize);
    gpuErrchk(cudaGetLastError());

    MoveAccelerationsKernel<<<(_numSatellites + 255) / 256, 256>>>(
        _devAccelerationMatrix, _numSatellites);
    gpuErrchk(cudaGetLastError());

    _forceModel.ComputePropagationAcceleration(
        _devPositionVector, _devVelocityVector, _devAccelerationMatrix, stepNum + 4, 8);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
