#include "monte_carlo/kernels.cuh"
#include "monte_carlo/types.cuh"

__global__ void UpdateCoarseApproachKernel(
    double* positions, int entityCount, CoarsePhaseParams* params, int stepNum, int numSteps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= entityCount || stepNum == 0 || stepNum >= numSteps)
        return;

    double* entityPosition = positions + idx * 3;
    double* objectPosition = positions + entityCount * 3; // Last 3 elements are the object position

    double posX = entityPosition[0];
    double posY = entityPosition[1];
    double posZ = entityPosition[2];

    double objPosX = objectPosition[0];
    double objPosY = objectPosition[1];
    double objPosZ = objectPosition[2];

    double dx = posX - objPosX;
    double dy = posY - objPosY;
    double dz = posZ - objPosZ;
    double distance = sqrt(dx * dx + dy * dy + dz * dz);

    if (distance >= params->distances[idx])
        return;

    params->distances[idx] = distance;
    params->stepNums[idx] = stepNum;

    // Update entity position before the closest approach
    double* posBefore = params->positionsBefore + idx * 3;
    double* posLast = params->lastPositions + idx * 3;
    posBefore[0] = posLast[0];
    posBefore[1] = posLast[1];
    posBefore[2] = posLast[2];

    // Update entity velocity before the closest approach
    double* velBefore = params->velocitiesBefore + idx * 3;
    double* velLast = params->lastVelocities + idx * 3;
    velBefore[0] = velLast[0];
    velBefore[1] = velLast[1];
    velBefore[2] = velLast[2];

    // Update object position before the closest approach
    double* objPosBefore = params->objectPositionsBefore + idx * 3;
    double* objPosLast
        = params->lastPositions + entityCount * 3; // Last 3 elements are the object position
    objPosBefore[0] = objPosLast[0];
    objPosBefore[1] = objPosLast[1];
    objPosBefore[2] = objPosLast[2];

    // Update object velocity before the closest approach
    double* objVelBefore = params->objectVelocitiesBefore + idx * 3;
    double* objVelLast
        = params->lastVelocities + entityCount * 3; // Last 3 elements are the object velocity
    objVelBefore[0] = objVelLast[0];
    objVelBefore[1] = objVelLast[1];
    objVelBefore[2] = objVelLast[2];
}

__global__ void UpdateFineApproachKernel(
    double* positions, int entityCount, FinePhaseParams* params, int stepNum, int numSteps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= entityCount || stepNum == 0)
        return;

    int objectVersion = params->objectVersion[idx];

    double* entityPosition = positions + idx * 3;
    double* objectPosition = positions + (entityCount + objectVersion) * 3;

    double posX = entityPosition[0];
    double posY = entityPosition[1];
    double posZ = entityPosition[2];

    double objPosX = objectPosition[0];
    double objPosY = objectPosition[1];
    double objPosZ = objectPosition[2];

    double dx = posX - objPosX;
    double dy = posY - objPosY;
    double dz = posZ - objPosZ;
    double distance = sqrt(dx * dx + dy * dy + dz * dz);

    // If this is the closest approach so far and it's not the last step
    if (distance < params->distances[idx] && stepNum != numSteps - 1) {
        params->distances[idx] = distance;
        params->stepNums[idx] = stepNum;

        // Update entity position before the closest approach
        double* posBefore = params->positionsBefore + idx * 3;
        double* posLast = params->lastPositions + idx * 3;
        posBefore[0] = posLast[0];
        posBefore[1] = posLast[1];
        posBefore[2] = posLast[2];

        // Update object position before the closest approach
        double* objPosBefore = params->objectPositionsBefore + idx * 3;
        double* objPosLast = params->lastPositions + (entityCount + objectVersion) * 3;
        objPosBefore[0] = objPosLast[0];
        objPosBefore[1] = objPosLast[1];
        objPosBefore[2] = objPosLast[2];
    }

    // If the closest approach was one step ago
    else if (params->stepNums[idx] == stepNum - 1) {
        // Update entity position after the closest approach
        double* posAfter = params->positionsAfter + idx * 3;
        posAfter[0] = posX;
        posAfter[1] = posY;
        posAfter[2] = posZ;

        // Update object position after the closest approach
        double* objPosAfter = params->objectPositionsAfter + idx * 3;
        objPosAfter[0] = objPosX;
        objPosAfter[1] = objPosY;
        objPosAfter[2] = objPosZ;
    }
}

__global__ void FindClosestApproachKernel(
    int entityCount, FinePhaseParams* params, double* closestDistances)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= entityCount)
        return;

    double* posBefore = params->positionsBefore + idx * 3;
    double xA0 = posBefore[0];
    double yA0 = posBefore[1];
    double zA0 = posBefore[2];

    double* posAfter = params->positionsAfter + idx * 3;
    double xA1 = posAfter[0];
    double yA1 = posAfter[1];
    double zA1 = posAfter[2];

    double* objPosBefore = params->objectPositionsBefore + idx * 3;
    double xB0 = objPosBefore[0];
    double yB0 = objPosBefore[1];
    double zB0 = objPosBefore[2];

    double* objPosAfter = params->objectPositionsAfter + idx * 3;
    double xB1 = objPosAfter[0];
    double yB1 = objPosAfter[1];
    double zB1 = objPosAfter[2];

    // Distance between satellite and object at t = 0 (start) and t = 1 (end)
    double xAB0 = xA0 - xB0;
    double yAB0 = yA0 - yB0;
    double zAB0 = zA0 - zB0;

    double xAB1 = xA1 - xB1;
    double yAB1 = yA1 - yB1;
    double zAB1 = zA1 - zB1;

    double dX = xAB1 - xAB0;
    double dY = yAB1 - yAB0;
    double dZ = zAB1 - zAB0;

    double a = dX * xAB0 + dY * yAB0 + dZ * zAB0;
    double b = dX * dX + dY * dY + dZ * dZ;
    double t = -a / b;

    // Clamp t to the range [0, 1]
    if (t < 0.0) {
        t = 0.0;
    } else if (t > 1.0) {
        t = 1.0;
    }

    // Interpolate positions
    double xClosest = xA0 + t * (xB1 - xA0);
    double yClosest = yA0 + t * (yB1 - yA0);
    double zClosest = zA0 + t * (zB1 - zA0);

    double xObjClosest = xB0 + t * (xB1 - xB0);
    double yObjClosest = yB0 + t * (yB1 - yB0);
    double zObjClosest = zB0 + t * (zB1 - zB0);

    // Calculate the distance at the closest approach
    double distX = xClosest - xObjClosest;
    double distY = yClosest - yObjClosest;
    double distZ = zClosest - zObjClosest;

    closestDistances[idx] = sqrt(distX * distX + distY * distY + distZ * distZ);
}