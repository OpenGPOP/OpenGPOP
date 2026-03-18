#include "force_model/force_model.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "monte_carlo/monte_carlo.cuh"
#include "monte_carlo/utility.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
// #include <iostream>

// void PrintEntities(double* positionsHost, double* velocitiesHost, int numSatellites)
// {
//     for (int i = 0; i < numSatellites; i += 1) {
//         std::cout << "Satellite " << i << ": Position: (" << positionsHost[i * 3] << ", "
//                   << positionsHost[i * 3 + 1] << ", " << positionsHost[i * 3 + 2]
//                   << "), Velocity: (" << velocitiesHost[i * 3] << ", " << velocitiesHost[i * 3 +
//                   1]
//                   << ", " << velocitiesHost[i * 3 + 2] << ")\n";
//     }
// }

void ClosestApproachFinder::Initialize(std::vector<double>& positions,
    std::vector<double>& velocities, std::vector<unsigned>& setIndices)
{
    // Initialization should be called only once
    assert(_numEntities == 0);

    // Check input sanity
    assert(positions.size() > 0);
    assert(positions.size() % 3 == 0);
    assert(positions.size() == velocities.size());
    assert(positions.size() == setIndices.size() * 3);
    _numEntities = setIndices.size();

    // Calculate partition parameters
    assert(CalculatePartitionParameters(setIndices, _hostPartitionParams));
    _hostPartitionParams.setIds = new unsigned[setIndices.size()];
    memcpy(_hostPartitionParams.setIds, setIndices.data(), sizeof(unsigned) * setIndices.size());

    // Allocate device memory, copy partition parameters and setIds and free host memory
    gpuErrchk(cudaMalloc(&_devPartitionParams.setIds, sizeof(unsigned) * setIndices.size()));
    gpuErrchk(
        cudaMalloc(&_devPartitionParams.setOffsets, sizeof(unsigned) * _devPartitionParams.setNum));
    gpuErrchk(cudaMalloc(
        &_devPartitionParams.numHigherIds, sizeof(unsigned) * _devPartitionParams.setNum));
    gpuErrchk(
        cudaMalloc(&_devPartitionParams.mapOffsets, sizeof(unsigned) * _devPartitionParams.setNum));
    gpuErrchk(cudaMemcpy(_devPartitionParams.setIds, _hostPartitionParams.setIds,
        sizeof(unsigned) * setIndices.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devPartitionParams.setOffsets, _hostPartitionParams.setOffsets,
        sizeof(unsigned) * _devPartitionParams.setNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devPartitionParams.numHigherIds, _hostPartitionParams.numHigherIds,
        sizeof(unsigned) * _devPartitionParams.setNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devPartitionParams.mapOffsets, _hostPartitionParams.mapOffsets,
        sizeof(unsigned) * _devPartitionParams.setNum, cudaMemcpyHostToDevice));

    // Allocate device memory and copy position and velocity data
    gpuErrchk(cudaMalloc(&_devPositions, sizeof(double) * positions.size()));
    gpuErrchk(cudaMalloc(&_devVelocities, sizeof(double) * velocities.size()));
    gpuErrchk(cudaMemcpy(_devPositions, positions.data(), sizeof(double) * positions.size(),
        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devVelocities, velocities.data(), sizeof(double) * velocities.size(),
        cudaMemcpyHostToDevice));

    // Allocate memory for results
    _devApproachParams.cudaAllocate(_devPartitionParams.pairNum);

    // Initialize the closest distances to double max and closest steps to -1
    std::vector<double> closestDistancesHost(
        _devPartitionParams.pairNum, std::numeric_limits<double>::max());
    std::vector<int> closestStepsHost(_devPartitionParams.pairNum, -1);
    gpuErrchk(cudaMemcpy(_devApproachParams.closestDistances, closestDistancesHost.data(),
        sizeof(double) * _devPartitionParams.pairNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devApproachParams.closestSteps, closestStepsHost.data(),
        sizeof(int) * _devPartitionParams.pairNum, cudaMemcpyHostToDevice));

    // Allocate memory for acceleration vectors
    gpuErrchk(cudaMalloc(&_devAccelerations, sizeof(double) * positions.size()));
}

void ClosestApproachFinder::Propagate(double stepSize, int numSteps)
{
    assert(_numEntities != 0); // Ensure initialized
    assert(stepSize > 0);
    assert(numSteps > 0);
    _lastStepSize = stepSize;

    ForceModel forceModel { _cublasHandle, _numEntities, stepSize, _params };
    GpuPropagator propagator(_devPositions, _devVelocities, _numEntities, _cublasHandle, stepSize,
        _params, forceModel, numSteps);

    for (int step = 0; step < numSteps; step++) {
        propagator.Step(nullptr, nullptr, _devAccelerations);

        int threadsPerBlock = 256;
        int blockCount = _devPartitionParams.pairNum / threadsPerBlock + 1;
        UpdateClosestApproachKernel<<<blockCount, threadsPerBlock>>>(_devPartitionParams,
            _devPositions, _devVelocities, _devAccelerations, _devLastPositions, _devLastVelocities,
            _devLastAccelerations, _devApproachParams, step);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
}

void ClosestApproachFinder::FindClosest(ClosestApproaches& closestApproaches)
{
    assert(_lastStepSize > 0); // Ensure that Propagate has been called

    double* devClosestTimes;
    gpuErrchk(cudaMalloc(&devClosestTimes, sizeof(double) * _devPartitionParams.pairNum));

    int threadsPerBlock = 256;
    int blockCount = _devPartitionParams.pairNum / threadsPerBlock + 1;
    CalculateClosestApproachesKernel<<<blockCount, threadsPerBlock>>>(
        _devPartitionParams, _devApproachParams, _lastStepSize, devClosestTimes);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy the closest distances back to the host
    double* closestDistances = new double[_devPartitionParams.pairNum];
    gpuErrchk(cudaMemcpy(closestDistances, _devApproachParams.closestDistances,
        sizeof(double) * _devPartitionParams.pairNum, cudaMemcpyDeviceToHost));

    double* closestTimes = new double[_devPartitionParams.pairNum];
    gpuErrchk(cudaMemcpy(closestTimes, devClosestTimes,
        sizeof(double) * _devPartitionParams.pairNum, cudaMemcpyDeviceToHost));

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