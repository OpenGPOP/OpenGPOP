#pragma once
#include "types/Parameters.hpp"
#include <array>
#include <vector>

constexpr double IdealStepSize = 30;

struct PropagationInput {
    TimeData propagationStart;
    int propagationSeconds;
    std::vector<std::array<double, 3>> positions;
    std::vector<std::array<double, 3>> velocities;
};

struct PropagationOutput {
    TimeData propagationStart;
    int propagationSeconds;
    std::vector<std::array<double, 3>> positionsStart;
    std::vector<std::array<double, 3>> velocitiesStart;
    std::vector<std::array<double, 3>> positionsEnd;
    std::vector<std::array<double, 3>> velocitiesEnd;
};

struct ClosestApproachInput {
    TimeData propagationStart;
    int propagationSeconds;
    std::vector<std::array<double, 3>> positions;
    std::vector<std::array<double, 3>> velocities;
    std::vector<unsigned> setIndices;
};

struct ClosestApproachRecord {
    int objectIndex1, objectIndex2;
    double distance, time;
};

struct ClosestApproachOutput {
    TimeData propagationStart;
    int propagationSeconds;
    std::vector<std::array<double, 3>> positionsStart;
    std::vector<std::array<double, 3>> velocitiesStart;
    std::vector<ClosestApproachRecord> closestApproaches;
};

PropagationOutput propagate(const PropagationInput& input);

ClosestApproachOutput closestApproach(const ClosestApproachInput& input);
