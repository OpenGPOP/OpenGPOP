#include "monte_carlo/monte_carlo.cuh"

// Position and velocity arrays should include the object at the end (the total length should be
// entityCount + 1)
__global__ void UpdateCoarseApproachKernel(
    double* positions, int entityCount, CoarsePhaseParams* params, int stepNum, int numSteps);

// Position and velocity arrays should include the different object versions at the end (the total
// length should be entityCount + objectVersionsCount)
__global__ void UpdateFineApproachKernel(
    double* positions, int entityCount, FinePhaseParams* params, int stepNum, int numSteps);

__global__ void FindClosestApproachKernel(
    int entityCount, FinePhaseParams* params, double* closestDistances);