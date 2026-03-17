#include "force_model/constants.cuh"
#include "force_model/force_model.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include <cstring>
#include <vector>

void ForceModel::PreInitialize()
{
    gpuErrchk(cudaSetDevice(0));

    _timePoints.Initialize();
    for (const auto& source : _accelerationSources) {
        source->Initialize();
    }

    // Allocate memory for summation matrix
    auto sourcesNum = _accelerationSources.size();
    gpuErrchk(
        cudaMalloc(&_devAccelerationSumMatrix, sizeof(double) * sourcesNum * 3 * _numEntities));
    gpuErrchk(
        cudaMemset(_devAccelerationSumMatrix, 0, sizeof(double) * sourcesNum * 3 * _numEntities));

    // Setup the ones vector for acceleration source summation
    std::vector<double> ones;
    ones.insert(ones.end(), sourcesNum, 1.0);
    gpuErrchk(cudaMalloc(&_devOnesVector, sizeof(double) * sourcesNum));
    gpuErrchk(cudaMemcpy(
        _devOnesVector, ones.data(), sizeof(double) * sourcesNum, cudaMemcpyHostToDevice));
}

void ForceModel::InitializeGeneration(int beforeEpoch, int afterEpoch)
{
    int arrayLength = (beforeEpoch + afterEpoch) * CFMConst::rk8StepCount;
    _timePoints.CalculateParameters(arrayLength, true, beforeEpoch);
}

void ForceModel::InitializePropagation(int beforeEpoch, int afterEpoch)
{
    int arrayLength = beforeEpoch + afterEpoch + 1; // +1 for epoch
    _timePoints.CalculateParameters(arrayLength, false, beforeEpoch);
}

void ForceModel::ComputeGenerationAcceleration(
    double* devPosition, double* devVelocity, double* devKMatrix, int stepNum, int kIndex)
{
    if (kIndex < 0 || kIndex >= CFMConst::rk8StepCount) {
        fprintf(stderr, "kIndex must be between 0 and %d", CFMConst::rk8StepCount - 1);
        return;
    }

    int kMatrixColumnCount = CFMConst::rk8StepCount;
    // This is an index of the time parameters array
    if (stepNum > 0)
        stepNum -= 1;
    int arrayIndex = (stepNum + 4) * kMatrixColumnCount + kIndex;

    double* devSumTempPointer = _devAccelerationSumMatrix;
    auto sourcesNum = _accelerationSources.size();
    for (const auto& source : _accelerationSources) {
        source->ComputeAcceleration(
            devPosition, devVelocity, arrayIndex, devSumTempPointer, sourcesNum);
        devSumTempPointer += 3;
    }

    // Calculate sum of all acceleration sources
    double alpha = 1.0;
    double beta = 0.0;
    cublasErrchk(cublasDgemvStridedBatched(*_handle, CUBLAS_OP_N, 3, sourcesNum, &alpha,
        _devAccelerationSumMatrix, 3, sourcesNum * 3, _devOnesVector, 1, 0, &beta,
        devKMatrix + kIndex * 3, 1, 3 * kMatrixColumnCount, _numEntities));
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void ForceModel::ComputePropagationAcceleration(double* devPosition, double* devVelocity,
    double* devAcceleration, int stepNum, int returnColumnIndex)
{
    double* devSumTempPointer = _devAccelerationSumMatrix;
    auto sourcesNum = _accelerationSources.size();
    for (const auto& source : _accelerationSources) {
        source->ComputeAcceleration(
            devPosition, devVelocity, stepNum, devSumTempPointer, sourcesNum * 3);
        devSumTempPointer += 3;
    }

    // Calculate sum of all acceleration sources
    double alpha = 1.0;
    double beta = 0.0;
    const int accelerationColumnsCount = 9;
    cublasErrchk(cublasDgemvStridedBatched(*_handle, CUBLAS_OP_N, 3, sourcesNum, &alpha,
        _devAccelerationSumMatrix, 3, sourcesNum * 3, _devOnesVector, 1, 0, &beta,
        devAcceleration + 3 * returnColumnIndex, 1, 3 * accelerationColumnsCount, _numEntities));
}

ForceModel::~ForceModel()
{
    CUDA_SAFE_FREE(_devAccelerationSumMatrix);
    CUDA_SAFE_FREE(_devOnesVector);
}
