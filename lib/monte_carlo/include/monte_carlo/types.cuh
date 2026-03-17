#pragma once

struct CoarsePhaseParams {
    int* stepNums = nullptr;
    double* distances = nullptr;

    double* positionsBefore = nullptr;
    double* velocitiesBefore = nullptr;

    double* objectPositionsBefore = nullptr;
    double* objectVelocitiesBefore = nullptr;

    // These are copies of positions and velocities from the last step (including object at the end)
    double* lastPositions = nullptr;
    double* lastVelocities = nullptr;
};

struct FinePhaseParams {
    double* positions = nullptr;
    double* velocities = nullptr;

    int* objectVersion = nullptr; // Which version of the object should be used by the given entity
    int* stepNums = nullptr;
    double* distances = nullptr;

    double* positionsBefore = nullptr;
    double* positionsAfter = nullptr;
    double* objectPositionsBefore = nullptr;
    double* objectPositionsAfter = nullptr;

    // These are copies of positions and velocities from the last step (including different object
    // versions at the end)
    double* lastPositions = nullptr;
};