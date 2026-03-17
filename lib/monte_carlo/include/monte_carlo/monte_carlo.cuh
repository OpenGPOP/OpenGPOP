#pragma once

#include "cublas_v2.h"
#include "monte_carlo/types.cuh"
#include "types/Parameters.hpp"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

class MonteCarlo {
public:
    /**
     * @brief Constructs a MonteCarlo object with device pointers and entity count.
     *
     * @param cublasHandle Pointer to the cuBLAS handle for matrix operations.
     * @param params Full perturbation parameters for the Monte Carlo simulation.
     * @param devPosition Pointer to the device memory for positions of entities (with the object
     * position at the end).
     * @param devVelocity Pointer to the device memory for velocities of entities (with the object
     * velocity at the end).
     * @param numEntities Number of entities in the simulation (not including the object).
     * @param coarseStepSize Step size for coarse propagation.
     * @param numSteps Number of steps for coarse propagation.
     * @param fineStepSize Step size for fine propagation (should evenly divide the coarse step
     * size).
     */
    MonteCarlo(cublasHandle_t* cublasHandle, FullPerturbationParameters* params,
        double* devPosition, double* devVelocity, int numEntities, double coarseStepSize,
        int numSteps, double fineStepSize)
        : _cublasHandle(cublasHandle)
        , _params(params)
        , _numEntities(numEntities)
        , _devPosition(devPosition)
        , _devVelocity(devVelocity)
        , _coarseStepSize(coarseStepSize)
        , _coarseNumSteps(numSteps)
        , _fineStepSize(fineStepSize)
        , _fineNumSteps(static_cast<int>(coarseStepSize / fineStepSize)
              * 2) // * 2 because we propagate from step-1 to step+1
    {
        if (std::fmod(coarseStepSize, fineStepSize) != 0.0) {
            std::cerr << "MonteCarlo: fineStepSize should evenly divide coarseStepSize!"
                      << std::endl;
        }
    }
    ~MonteCarlo() { FreeAllMemory(); }

    void CoarsePropagationPhase();

    void FinePropagationPhase();

    // The function finds the number of satellites that are closer than the given distance. If
    // hostClosestDistances is provided, it will be filled with the distances.
    int CollectResults(double distanceThreshold, double* hostClosestDistances = nullptr);

private:
    cublasHandle_t* _cublasHandle;
    FullPerturbationParameters* _params;

    int _numEntities;
    double* _devPosition;
    double* _devVelocity;
    double _coarseStepSize;
    int _coarseNumSteps;
    double _fineStepSize;
    int _fineNumSteps;

    CoarsePhaseParams _coarsePhaseParams;
    CoarsePhaseParams* _devCoarsePhaseParams;

    FinePhaseParams _finePhaseParams;
    FinePhaseParams* _devFinePhaseParams;

    void UpdateCoarseApproach(int stepNum);
    void UpdateFineApproach(int stepNum, int objectVersionsCount);

    void AllocateCoarsePhaseMemory();
    void AllocateFinePhaseMemory(int objectVersionsCount);

    void FreeCoarsePhaseMemory();
    void FreeFinePhaseMemory();

    void FreeAllMemory();
};