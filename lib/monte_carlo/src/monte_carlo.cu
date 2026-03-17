#include "force_model/force_model.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "monte_carlo/kernels.cuh"
#include "monte_carlo/monte_carlo.cuh"
#include "types/Parameters.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

void PrintEntities(double* positionsHost, double* velocitiesHost, int numSatellites)
{
    for (int i = 0; i < numSatellites; i += 1) {
        std::cout << "Satellite " << i << ": Position: (" << positionsHost[i * 3] << ", "
                  << positionsHost[i * 3 + 1] << ", " << positionsHost[i * 3 + 2]
                  << "), Velocity: (" << velocitiesHost[i * 3] << ", " << velocitiesHost[i * 3 + 1]
                  << ", " << velocitiesHost[i * 3 + 2] << ")\n";
    }
}

void MonteCarlo::CoarsePropagationPhase()
{
    cudaError_t cudaStatus;

    AllocateCoarsePhaseMemory();

    // Initialize closest approach index to -1
    int* hostClosestApproachIndex = new int[_numEntities];
    for (int i = 0; i < _numEntities; ++i) {
        hostClosestApproachIndex[i] = -1;
    }
    cudaStatus = cudaMemcpy(_coarsePhaseParams.stepNums, hostClosestApproachIndex,
        _numEntities * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _closestApproachParams.indexes!");
        return;
    }
    delete[] hostClosestApproachIndex;

    // Initialize closest approach distance to double max
    double* hostClosestApproachDistance = new double[_numEntities];
    for (int i = 0; i < _numEntities; ++i) {
        hostClosestApproachDistance[i] = std::numeric_limits<double>::max();
    }
    cudaStatus = cudaMemcpy(_coarsePhaseParams.distances, hostClosestApproachDistance,
        _numEntities * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _closestApproachParams.distances!");
        return;
    }
    delete[] hostClosestApproachDistance;

    double* hostPositions = new double[(_numEntities + 1 /* +1 for object */) * 3];
    double* hostVelocities = new double[(_numEntities + 1 /* +1 for object */) * 3];

    ForceModel forceModel(
        _cublasHandle, _numEntities + 1 /*+1 for object*/, _coarseStepSize, _params);
    GpuPropagator propagator(_devPosition, _devVelocity, _numEntities + 1 /*+1 for object*/,
        _cublasHandle, _coarseStepSize, _params, forceModel, _coarseNumSteps);

    for (int j = 0; j < _coarseNumSteps; j++) {
        std::cout << "Coarse step: " << j + 1 << std::endl;
        propagator.Step(hostPositions, hostVelocities);
        PrintEntities(hostPositions, hostVelocities, _numEntities);
        UpdateCoarseApproach(j);
    }

    delete[] hostPositions;
    delete[] hostVelocities;
}

void MonteCarlo::UpdateCoarseApproach(int stepNum)
{
    cudaError_t cudaStatus;

    // We skip the frist step, because we do not have previous positions
    // If the first step was the closest approach, we will still find it later (since the next step
    // will be considered the closest, we will save the first step positions in the 'before' array)
    //
    // For the coarse phase, we omit the last step, because we don't need the "after" positions
    if (stepNum != 0 && stepNum != _coarseNumSteps - 1) {
        int threadsPerBlock = 256;
        int blockCount = _numEntities / threadsPerBlock + 1;
        UpdateCoarseApproachKernel<<<threadsPerBlock, blockCount>>>(
            _devPosition, _numEntities, _devCoarsePhaseParams, stepNum, _coarseNumSteps);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "ClosestApproachKernel launch failed: %s\n",
                cudaGetErrorString(cudaStatus));
            return;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }
    }

    // Owerwrite last positions with current positions
    cudaStatus = cudaMemcpy(_coarsePhaseParams.lastPositions, _devPosition,
        (_numEntities + 1 /* +1 for object */) * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _devLastPositions!");
        return;
    }
    // Owerwrite last velocities with current velocities
    cudaStatus = cudaMemcpy(_coarsePhaseParams.lastVelocities, _devVelocity,
        (_numEntities + 1 /* +1 for object */) * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _devLastVelocities!");
        return;
    }
}

// The idea is to check which step was the closest for each satellite, and then apply fine
// propagation to both the entities and the versions of the object at that step. To avoid computing
// object propagation from the same time step multiple times, we find the number of distinct closest
// approach steps.
//
// The whole approach assumes that closest approaches for all satellites are happening roughly at
// the same time.
void MonteCarlo::FinePropagationPhase()
{
    cudaError_t cudaStatus;

    int* hostStepNums = new int[_numEntities];
    cudaStatus = cudaMemcpy(hostStepNums, _coarsePhaseParams.stepNums, _numEntities * sizeof(int),
        cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _closestApproachParams.stepNums!");
        return;
    }
    double* hostObjPosBefore = new double[_numEntities * 3];
    cudaStatus = cudaMemcpy(hostObjPosBefore, _coarsePhaseParams.objectPositionsBefore,
        _numEntities * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _closestApproachParams.objectPositionsBefore!");
        return;
    }
    double* hostObjVelBefore = new double[_numEntities * 3];
    cudaStatus = cudaMemcpy(hostObjVelBefore, _coarsePhaseParams.objectVelocitiesBefore,
        _numEntities * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _closestApproachParams.objectVelocitiesBefore!");
        return;
    }

    std::vector<int> distinctStepNums;
    std::vector<double> objPosBefore;
    std::vector<double> objVelBefore;
    int* objectVersions = new int[_numEntities];

    int firstClosestApproachStep = std::numeric_limits<int>::max();
    for (int i = 0; i < _numEntities; ++i) {
        int stepNum = hostStepNums[i];
        if (stepNum < firstClosestApproachStep) {
            firstClosestApproachStep = stepNum;
        }

        objectVersions[i] = distinctStepNums.size() - 1;
        bool found = false;
        for (long unsigned int j = 0; j < distinctStepNums.size(); ++j) {
            if (stepNum == distinctStepNums[j]) {
                found = true;
                break;
            }
        }

        if (found)
            continue;

        objectVersions[i]++;
        distinctStepNums.push_back(stepNum);
        objPosBefore.push_back(hostObjPosBefore[i * 3]);
        objPosBefore.push_back(hostObjPosBefore[i * 3 + 1]);
        objPosBefore.push_back(hostObjPosBefore[i * 3 + 2]);
        objVelBefore.push_back(hostObjVelBefore[i * 3]);
        objVelBefore.push_back(hostObjVelBefore[i * 3 + 1]);
        objVelBefore.push_back(hostObjVelBefore[i * 3 + 2]);
    }

    delete[] hostStepNums;
    delete[] hostObjPosBefore;
    delete[] hostObjVelBefore;

    AllocateFinePhaseMemory(distinctStepNums.size());

    cudaStatus = cudaMemcpy(_finePhaseParams.objectVersion, objectVersions,
        _numEntities * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.objectVersion!");
        return;
    }
    delete[] objectVersions;

    // Construct new arrays with distinct versions of the object at the end
    cudaStatus = cudaMemcpy(_finePhaseParams.positions, _coarsePhaseParams.positionsBefore,
        _numEntities * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.positions!");
        return;
    }
    cudaStatus = cudaMemcpy(_finePhaseParams.positions + _numEntities * 3, objPosBefore.data(),
        objPosBefore.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.positions!");
        return;
    }
    cudaStatus = cudaMemcpy(_finePhaseParams.velocities, _coarsePhaseParams.velocitiesBefore,
        _numEntities * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.velocities!");
        return;
    }
    cudaStatus = cudaMemcpy(_finePhaseParams.velocities + _numEntities * 3, objVelBefore.data(),
        objVelBefore.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.velocities!");
        return;
    }

    FreeCoarsePhaseMemory();

    // Initialize closest approach index to -1
    int* hostClosestApproachIndex = new int[_numEntities];
    for (int i = 0; i < _numEntities; ++i) {
        hostClosestApproachIndex[i] = -1;
    }
    cudaStatus = cudaMemcpy(_finePhaseParams.stepNums, hostClosestApproachIndex,
        _numEntities * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.indexes!");
        return;
    }
    delete[] hostClosestApproachIndex;

    // Initialize closest approach distance to double max
    double* hostClosestApproachDistance = new double[_numEntities];
    for (int i = 0; i < _numEntities; ++i) {
        hostClosestApproachDistance[i] = std::numeric_limits<double>::max();
    }
    cudaStatus = cudaMemcpy(_finePhaseParams.distances, hostClosestApproachDistance,
        _numEntities * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.distances!");
        return;
    }
    delete[] hostClosestApproachDistance;

    // Change the start time
    FullPerturbationParameters newParams = *_params;
    newParams.StartTimeUTC.Sec
        += (firstClosestApproachStep - 1) * _coarseStepSize; // -1 because we start from 'before'

    ForceModel forceModel(
        _cublasHandle, _numEntities + distinctStepNums.size(), _fineStepSize, &newParams);
    GpuPropagator propagator(_finePhaseParams.positions, _finePhaseParams.velocities,
        _numEntities + distinctStepNums.size(), _cublasHandle, _fineStepSize, &newParams,
        forceModel, _fineNumSteps);

    double* hostPositions = new double[(_numEntities + distinctStepNums.size()) * 3];
    double* hostVelocities = new double[(_numEntities + distinctStepNums.size()) * 3];

    for (int stepNum = 0; stepNum < _fineNumSteps; ++stepNum) {
        std::cout << "Fine step: " << stepNum + 1 << std::endl;
        propagator.Step(hostPositions, hostVelocities);
        PrintEntities(hostPositions, hostVelocities, _numEntities);
        UpdateFineApproach(stepNum, distinctStepNums.size());
    }
}

void MonteCarlo::UpdateFineApproach(int stepNum, int objectVersionsCount)
{
    cudaError_t cudaStatus;

    // We skip the first step, because we do not have previous positions
    // If the first step was the closest approach, we will still find it later (since the next step
    // will be considered the closest, we will save the first step positions in the 'before' array)
    if (stepNum != 0) {
        int threadsPerBlock = 256;
        int blockCount = _numEntities / threadsPerBlock + 1;
        UpdateFineApproachKernel<<<threadsPerBlock, blockCount>>>(
            _finePhaseParams.positions, _numEntities, _devFinePhaseParams, stepNum, _fineNumSteps);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "UpdateFineApproachKernel launch failed: %s\n",
                cudaGetErrorString(cudaStatus));
            return;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }
    }

    // Owerwrite last positions with current positions
    cudaStatus = cudaMemcpy(_finePhaseParams.lastPositions, _finePhaseParams.positions,
        (_numEntities + objectVersionsCount) * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.lastPositions!");
        return;
    }
}

int MonteCarlo::CollectResults(double distanceThreshold, double* hostClosestDistances)
{
    cudaError_t cudaStatus;

    int threadsPerBlock = 256;
    int blockCount = _numEntities / threadsPerBlock + 1;
    FindClosestApproachKernel<<<blockCount, threadsPerBlock>>>(_numEntities, _devFinePhaseParams,
        _finePhaseParams.distances); // We reuse the distances array, since it is no logner needed

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FindClosestApproachKernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
        return -1;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    // Copy the distances back to the host
    bool returnDistances = hostClosestDistances != nullptr;
    if (!returnDistances) {
        hostClosestDistances = new double[_numEntities];
    }
    cudaStatus = cudaMemcpy(hostClosestDistances, _finePhaseParams.distances,
        _numEntities * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _finePhaseParams.distances!");
        return -1;
    }

    int count = 0;
    for (int i = 0; i < _numEntities; ++i) {
        if (hostClosestDistances[i] < distanceThreshold) {
            count++;
        }
    }

    if (!returnDistances) {
        delete[] hostClosestDistances;
    }

    return count;
}