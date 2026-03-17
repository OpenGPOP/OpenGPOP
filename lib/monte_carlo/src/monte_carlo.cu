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

    // Calculate partition parameters and make a local copy to preserve host pointers
    assert(CalculatePartitionParameters(setIndices, _partitionParams));
    auto paramsCopy = _partitionParams;

    gpuErrchk(cudaMalloc(&_partitionParams.setIds, sizeof(unsigned) * setIndices.size()));
    gpuErrchk(cudaMemcpy(_partitionParams.setIds, setIndices.data(),
        sizeof(unsigned) * setIndices.size(), cudaMemcpyHostToDevice));

    // Allocate device memory, copy partition parameters and setIds and free host memory
    gpuErrchk(cudaMalloc(&_partitionParams.setOffsets, sizeof(unsigned) * _partitionParams.setNum));
    gpuErrchk(
        cudaMalloc(&_partitionParams.numHigherIds, sizeof(unsigned) * _partitionParams.setNum));
    gpuErrchk(cudaMalloc(&_partitionParams.mapOffsets, sizeof(unsigned) * _partitionParams.setNum));
    gpuErrchk(cudaMemcpy(_partitionParams.setOffsets, paramsCopy.setOffsets,
        sizeof(unsigned) * _partitionParams.setNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_partitionParams.numHigherIds, paramsCopy.numHigherIds,
        sizeof(unsigned) * _partitionParams.setNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_partitionParams.mapOffsets, paramsCopy.mapOffsets,
        sizeof(unsigned) * _partitionParams.setNum, cudaMemcpyHostToDevice));
    delete[] paramsCopy.setOffsets;
    delete[] paramsCopy.numHigherIds;
    delete[] paramsCopy.mapOffsets;

    // Allocate device memory and copy position and velocity data
    gpuErrchk(cudaMalloc(&_devPositions, sizeof(double) * positions.size()));
    gpuErrchk(cudaMalloc(&_devVelocities, sizeof(double) * velocities.size()));
    gpuErrchk(cudaMemcpy(_devPositions, positions.data(), sizeof(double) * positions.size(),
        cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_devVelocities, velocities.data(), sizeof(double) * velocities.size(),
        cudaMemcpyHostToDevice));

    // Allocate memory for results
    _approachStates.cudaAllocate(_partitionParams.pairNum);

    // Initialize the closest distances to double max and closest steps to -1
    std::vector<double> closestDistancesHost(
        _partitionParams.pairNum, std::numeric_limits<double>::max());
    std::vector<int> closestStepsHost(_partitionParams.pairNum, -1);
    gpuErrchk(cudaMemcpy(_approachStates.closestDistances, closestDistancesHost.data(),
        sizeof(double) * _partitionParams.pairNum, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_approachStates.closestSteps, closestStepsHost.data(),
        sizeof(int) * _partitionParams.pairNum, cudaMemcpyHostToDevice));

    // Allocate memory for acceleration vectors
    gpuErrchk(cudaMalloc(&_devAccelerations, sizeof(double) * positions.size()));
}

void ClosestApproach::Propagate(double stepSize, int numSteps)
{
    assert(_numEntities != 0); // Ensure initialized
    assert(stepSize > 0);
    assert(numSteps > 0);

    ForceModel forceModel { _cublasHandle, _numEntities, stepSize, _params };
    GpuPropagator propagator(_devPositions, _devVelocities, _numEntities, _cublasHandle, stepSize,
        _params, forceModel, numSteps);

    for (int step = 0; step < numSteps; step++) {
        propagator.Step(nullptr, nullptr, _devAccelerations);

        int threadsPerBlock = 256;
        int blockCount = _partitionParams.pairNum / threadsPerBlock + 1;
        UpdateClosestApproachKernel<<<blockCount, threadsPerBlock>>>(_partitionParams,
            _devPositions, _devVelocities, _devAccelerations, _devLastPositions, _devLastVelocities,
            _devLastAccelerations, _approachStates, step);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
}

int ClosestApproach::CollectResults(double distanceThreshold, double* hostClosestDistances)
{
    return 0;
    // cudaError_t cudaStatus;

    // int threadsPerBlock = 256;
    // int blockCount = _numEntities / threadsPerBlock + 1;
    // FindClosestApproachKernel<<<blockCount, threadsPerBlock>>>(_numEntities, _devFinePhaseParams,
    //     _finePhaseParams.distances); // We reuse the distances array, since it is no logner
    //     needed

    // cudaStatus = cudaGetLastError();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "FindClosestApproachKernel launch failed: %s\n",
    //         cudaGetErrorString(cudaStatus));
    //     return -1;
    // }
    // cudaStatus = cudaDeviceSynchronize();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
    //     return -1;
    // }

    // // Copy the distances back to the host
    // bool returnDistances = hostClosestDistances != nullptr;
    // if (!returnDistances) {
    //     hostClosestDistances = new double[_numEntities];
    // }
    // cudaStatus = cudaMemcpy(hostClosestDistances, _finePhaseParams.distances,
    //     _numEntities * sizeof(double), cudaMemcpyDeviceToHost);
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.distances!");
    //     return -1;
    // }

    // int count = 0;
    // for (int i = 0; i < _numEntities; ++i) {
    //     if (hostClosestDistances[i] < distanceThreshold) {
    //         count++;
    //     }
    // }

    // if (!returnDistances) {
    //     delete[] hostClosestDistances;
    // }

    // return count;
}