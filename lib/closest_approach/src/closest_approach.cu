#include "closest_approach/closest_approach.cuh"
#include "closest_approach/utility/partition_parameters.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>

ClosestApproach::~ClosestApproach()
{
    // Free device memory
    CUDA_SAFE_FREE(_devPositions);
    CUDA_SAFE_FREE(_devVelocities);
    CUDA_SAFE_FREE(_devAccelerations);
    CUDA_SAFE_FREE(_devLastPositions);
    CUDA_SAFE_FREE(_devLastVelocities);
    CUDA_SAFE_FREE(_devLastAccelerations);

    CUDA_SAFE_FREE(_devPartitionParams.setIds);
    CUDA_SAFE_FREE(_devPartitionParams.setOffsets);
    CUDA_SAFE_FREE(_devPartitionParams.numHigherIds);
    CUDA_SAFE_FREE(_devPartitionParams.mapOffsets);

    _devApproachParams.cudaFree();
    _hostPartitionParams.hostFree();
}

void ClosestApproach::Initialize(std::vector<double>& positions, std::vector<double>& velocities,
    std::vector<unsigned>& setIndices)
{
    // Initialization should be called only once
    assert(_numEntities == 0);

    // Check input sanity
    assert(positions.size() > 0);
    assert(positions.size() % 3 == 0);
    assert(positions.size() == velocities.size());
    assert(positions.size() == setIndices.size() * 3);
    _numEntities = setIndices.size();

    // Calculate partition parameters, save to _hostPartitionParams
    assert(setIndices.front() != setIndices.back()); // Ensure at least 2 sets
    assert(_hostPartitionParams.Calculate(setIndices));
    _hostPartitionParams.setIds = new unsigned[setIndices.size()];
    memcpy(_hostPartitionParams.setIds, setIndices.data(), sizeof(unsigned) * setIndices.size());

    // Allocate device memory, copy partition parameters and setIds
    _devPartitionParams.setNum = _hostPartitionParams.setNum;
    _devPartitionParams.pairNum = _hostPartitionParams.pairNum;
    gpuErrchk(cudaMalloc(&_devPartitionParams.setIds, sizeof(unsigned) * setIndices.size()));
    gpuErrchk(cudaMalloc(
        &_devPartitionParams.setOffsets, sizeof(unsigned) * _hostPartitionParams.setNum));
    gpuErrchk(cudaMalloc(
        &_devPartitionParams.numHigherIds, sizeof(unsigned) * _hostPartitionParams.setNum));
    gpuErrchk(cudaMalloc(
        &_devPartitionParams.mapOffsets, sizeof(unsigned) * _hostPartitionParams.setNum));
    gpuErrchk(cudaMemcpy(_devPartitionParams.setIds, _hostPartitionParams.setIds,
        sizeof(unsigned) * setIndices.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devPartitionParams.setOffsets, _hostPartitionParams.setOffsets,
        sizeof(unsigned) * _hostPartitionParams.setNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devPartitionParams.numHigherIds, _hostPartitionParams.numHigherIds,
        sizeof(unsigned) * _hostPartitionParams.setNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devPartitionParams.mapOffsets, _hostPartitionParams.mapOffsets,
        sizeof(unsigned) * _hostPartitionParams.setNum, cudaMemcpyHostToDevice));

    // Allocate device memory and copy position and velocity data
    gpuErrchk(cudaMalloc(&_devPositions, sizeof(double) * positions.size()));
    gpuErrchk(cudaMalloc(&_devVelocities, sizeof(double) * velocities.size()));
    gpuErrchk(cudaMemcpy(_devPositions, positions.data(), sizeof(double) * positions.size(),
        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devVelocities, velocities.data(), sizeof(double) * velocities.size(),
        cudaMemcpyHostToDevice));

    // Allocate memory for results
    _devApproachParams.cudaAllocate(_hostPartitionParams.pairNum);

    // Initialize the closest distances to double max and closest steps to -1
    std::vector<double> closestDistancesHost(
        _hostPartitionParams.pairNum, std::numeric_limits<double>::max());
    std::vector<int> closestStepsHost(_hostPartitionParams.pairNum, -1);
    gpuErrchk(cudaMemcpy(_devApproachParams.closestDistances, closestDistancesHost.data(),
        sizeof(double) * _hostPartitionParams.pairNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devApproachParams.closestSteps, closestStepsHost.data(),
        sizeof(int) * _hostPartitionParams.pairNum, cudaMemcpyHostToDevice));

    // Allocate memory for the last step's positions, velocities and accelerations
    gpuErrchk(cudaMalloc(&_devAccelerations, sizeof(double) * _numEntities * 3));
    gpuErrchk(cudaMalloc(&_devLastPositions, sizeof(double) * _numEntities * 3));
    gpuErrchk(cudaMalloc(&_devLastVelocities, sizeof(double) * _numEntities * 3));
    gpuErrchk(cudaMalloc(&_devLastAccelerations, sizeof(double) * _numEntities * 3));
}

void ClosestApproach::Propagate(int numSteps)
{
    assert(_numEntities != 0); // Ensure initialized
    assert(numSteps > 0);

    // ForceModel forceModel { _cublasHandle, _numEntities, stepSize, _params };
    GpuPropagator propagator(_devPositions, _devVelocities, _numEntities, _cublasHandle, _stepSize,
        _params, _forceModel, numSteps);

    for (int step = 0; step < numSteps; step++) {
        propagator.Step(nullptr, nullptr, _devAccelerations);

        int threadsPerBlock = 256;
        int blockCount = _devPartitionParams.pairNum / threadsPerBlock + 1;
        UpdateClosestApproachKernel<<<blockCount, threadsPerBlock>>>(_devPartitionParams,
            _devPositions, _devVelocities, _devAccelerations, _devLastPositions, _devLastVelocities,
            _devLastAccelerations, _devApproachParams, _totalNumSteps);
        _totalNumSteps++;
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
}

void ClosestApproach::FindClosest(ClosestApproachData& closestApproaches)
{
    assert(_totalNumSteps > 0); // Ensure that Propagate has been called

    double* devClosestTimes;
    gpuErrchk(cudaMalloc(&devClosestTimes, sizeof(double) * _devPartitionParams.pairNum));

    int threadsPerBlock = 256;
    int blockCount = _devPartitionParams.pairNum / threadsPerBlock + 1;
    CalculateClosestApproachesKernel<<<blockCount, threadsPerBlock>>>(
        _devPartitionParams, _devApproachParams, _stepSize, _totalNumSteps, devClosestTimes);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy the closest distances back to the host
    double* closestDistances = new double[_devPartitionParams.pairNum];
    gpuErrchk(cudaMemcpy(closestDistances, _devApproachParams.closestDistances,
        sizeof(double) * _devPartitionParams.pairNum, cudaMemcpyDeviceToHost));

    double* closestTimes = new double[_devPartitionParams.pairNum];
    gpuErrchk(cudaMemcpy(closestTimes, devClosestTimes,
        sizeof(double) * _devPartitionParams.pairNum, cudaMemcpyDeviceToHost));

    CUDA_SAFE_FREE(devClosestTimes);

    // Make a deep copy of the host partition parameters to return with the results
    PartitionParameters hostPartitionParamsCopy;
    hostPartitionParamsCopy.setNum = _hostPartitionParams.setNum;
    hostPartitionParamsCopy.pairNum = _hostPartitionParams.pairNum;
    hostPartitionParamsCopy.setIds = new unsigned[hostPartitionParamsCopy.setNum];
    hostPartitionParamsCopy.setOffsets = new unsigned[hostPartitionParamsCopy.setNum];
    hostPartitionParamsCopy.numHigherIds = new unsigned[hostPartitionParamsCopy.setNum];
    hostPartitionParamsCopy.mapOffsets = new unsigned[hostPartitionParamsCopy.setNum];
    std::memcpy(hostPartitionParamsCopy.setIds, _hostPartitionParams.setIds,
        sizeof(unsigned) * hostPartitionParamsCopy.setNum);
    std::memcpy(hostPartitionParamsCopy.setOffsets, _hostPartitionParams.setOffsets,
        sizeof(unsigned) * hostPartitionParamsCopy.setNum);
    std::memcpy(hostPartitionParamsCopy.numHigherIds, _hostPartitionParams.numHigherIds,
        sizeof(unsigned) * hostPartitionParamsCopy.setNum);
    std::memcpy(hostPartitionParamsCopy.mapOffsets, _hostPartitionParams.mapOffsets,
        sizeof(unsigned) * hostPartitionParamsCopy.setNum);

    closestApproaches.Populate(hostPartitionParamsCopy, closestDistances, closestTimes);
}