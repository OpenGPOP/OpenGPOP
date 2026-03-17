#include "monte_carlo/monte_carlo.cuh"
#include <cstdio>

void MonteCarlo::AllocateCoarsePhaseMemory()
{
    // Allocate memory for closest approach index
    cudaError_t cudaStatus = cudaMalloc(&_coarsePhaseParams.stepNums, _numEntities * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.indexes!");
        return;
    }
    // Allocate memory for closest approach distance
    cudaStatus = cudaMalloc(&_coarsePhaseParams.distances, _numEntities * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.distances!");
        return;
    }
    // Allocate memory for closest approach positions before and after
    cudaStatus = cudaMalloc(&_coarsePhaseParams.positionsBefore, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.positionsBefore!");
        return;
    }
    cudaStatus
        = cudaMalloc(&_coarsePhaseParams.objectPositionsBefore, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.objectPositionsBefore!");
        return;
    }

    // Allocate memory for velocities before and after
    cudaStatus
        = cudaMalloc(&_coarsePhaseParams.velocitiesBefore, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.velocitiesBefore!");
        return;
    }
    cudaStatus
        = cudaMalloc(&_coarsePhaseParams.objectVelocitiesBefore, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.objectVelocitiesBefore!");
        return;
    }

    // Allocate memory for last velocities
    cudaStatus = cudaMalloc(&_coarsePhaseParams.lastVelocities,
        (_numEntities + 1 /* +1 for object */) * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.lastVelocities!");
        return;
    }
    // Allocate memory for last positions
    cudaStatus = cudaMalloc(&_coarsePhaseParams.lastPositions,
        (_numEntities + 1 /* +1 for object */) * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _closestApproachParams.lastPositions!");
        return;
    }

    // Allocate memory for the device structure
    cudaStatus = cudaMalloc(&_devCoarsePhaseParams, sizeof(CoarsePhaseParams));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _devClosestApproachParams!");
        return;
    }

    // Copy the structure to the device
    cudaStatus = cudaMemcpy(_devCoarsePhaseParams, &_coarsePhaseParams, sizeof(CoarsePhaseParams),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _devClosestApproachParams!");
        return;
    }
}

void MonteCarlo::AllocateFinePhaseMemory(int objectVersionsCount)
{
    cudaError_t cudaStatus;

    // Allocate memory for fine phase parameters
    cudaStatus = cudaMalloc(
        &_finePhaseParams.positions, (_numEntities + objectVersionsCount) * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.positions!");
        return;
    }
    cudaStatus = cudaMalloc(
        &_finePhaseParams.velocities, (_numEntities + objectVersionsCount) * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.velocities!");
        return;
    }

    cudaStatus = cudaMalloc(&_finePhaseParams.objectVersion, _numEntities * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.objectVersion!");
        return;
    }
    cudaStatus = cudaMalloc(&_finePhaseParams.stepNums, _numEntities * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.stepNums!");
        return;
    }
    cudaStatus = cudaMalloc(&_finePhaseParams.distances, _numEntities * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.distances!");
        return;
    }

    cudaStatus = cudaMalloc(&_finePhaseParams.positionsBefore, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.positionsBefore!");
        return;
    }
    cudaStatus = cudaMalloc(&_finePhaseParams.positionsAfter, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.positionsAfter!");
        return;
    }

    cudaStatus
        = cudaMalloc(&_finePhaseParams.objectPositionsBefore, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.objectPositionsBefore!");
        return;
    }
    cudaStatus
        = cudaMalloc(&_finePhaseParams.objectPositionsAfter, _numEntities * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.objectPositionsAfter!");
        return;
    }

    // Allocate memory for last positions and velocities
    cudaStatus = cudaMalloc(
        &_finePhaseParams.lastPositions, (_numEntities + objectVersionsCount) * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _finePhaseParams.lastPositions!");
        return;
    }

    // Allocate memory for the device structure
    cudaStatus = cudaMalloc(&_devFinePhaseParams, sizeof(FinePhaseParams));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for _devFinePhaseParams!");
        return;
    }

    // Copy the structure to the device
    cudaStatus = cudaMemcpy(
        _devFinePhaseParams, &_finePhaseParams, sizeof(FinePhaseParams), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for _devFinePhaseParams!");
        return;
    }
}

void MonteCarlo::FreeCoarsePhaseMemory()
{
    cudaFree(_coarsePhaseParams.stepNums);
    cudaFree(_coarsePhaseParams.distances);
    cudaFree(_coarsePhaseParams.positionsBefore);
    cudaFree(_coarsePhaseParams.velocitiesBefore);
    cudaFree(_coarsePhaseParams.objectPositionsBefore);
    cudaFree(_coarsePhaseParams.objectVelocitiesBefore);
    cudaFree(_coarsePhaseParams.lastPositions);
    cudaFree(_coarsePhaseParams.lastVelocities);

    cudaFree(_devCoarsePhaseParams);
}

void MonteCarlo::FreeFinePhaseMemory()
{
    cudaFree(_finePhaseParams.positions);
    cudaFree(_finePhaseParams.velocities);
    cudaFree(_finePhaseParams.objectVersion);
    cudaFree(_finePhaseParams.stepNums);
    cudaFree(_finePhaseParams.distances);
    cudaFree(_finePhaseParams.positionsBefore);
    cudaFree(_finePhaseParams.positionsAfter);
    cudaFree(_finePhaseParams.objectPositionsBefore);
    cudaFree(_finePhaseParams.objectPositionsAfter);
    cudaFree(_finePhaseParams.lastPositions);

    cudaFree(_devFinePhaseParams);
}

void MonteCarlo::FreeAllMemory()
{
    FreeCoarsePhaseMemory();
    FreeFinePhaseMemory();
}