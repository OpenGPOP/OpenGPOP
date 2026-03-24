#include "runner.cuh"

#include "closest_approach/closest_approach.cuh"
#include "force_model/force_model.cuh"
#include "gpu_propagator/gpu_propagator.cuh"

#include <cuda/std/detail/libcxx/include/chrono>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>

PropagationOutput propagate(const PropagationInput& input)
{
    int numStepsToFirst = input.propagationSeconds / IdealStepSize;
    double stepSize = input.propagationSeconds / static_cast<double>(numStepsToFirst);
    int totalSteps = numStepsToFirst;

    FullPerturbationParameters parameters { input.propagationStart };

    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));
    ForceModel forceModel(&handle, 1, stepSize, &parameters);

    double* devPosition;
    double* devVelocity;

    cudaMallocManaged(&devPosition, input.positions.size() * 3 * sizeof(double));
    cudaMallocManaged(&devVelocity, input.velocities.size() * 3 * sizeof(double));

    cudaMemcpy(devPosition, input.positions.data(), input.positions.size() * 3 * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, input.velocities.data(), input.velocities.size() * 3 * sizeof(double),
        cudaMemcpyHostToDevice);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    GpuPropagator propagator(
        devPosition, devVelocity, 1, &handle, stepSize, &parameters, forceModel, totalSteps, 7);

    indicators::BlockProgressBar bar {
        indicators::option::BarWidth(80),
        indicators::option::PrefixText { "Propagating..." },
        indicators::option::Start { "[" },
        indicators::option::End { "]" },
        indicators::option::ShowPercentage { true },
        indicators::option::ShowElapsedTime { true },
        indicators::option::ShowRemainingTime { true },
    };

    for (int i = 0; i < totalSteps; i++) {
        bar.set_progress(static_cast<int>(static_cast<double>(i) * 100 / totalSteps));
        propagator.Step(nullptr, nullptr);
    }
    bar.set_progress(100);

    std::cout << std::endl;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

    std::vector<std::array<double, 3>> positionsEnd(input.positions.size());
    std::vector<std::array<double, 3>> velocitiesEnd(input.positions.size());
    propagator.CopyStateToHost(reinterpret_cast<double*>(positionsEnd.data()),
        reinterpret_cast<double*>(velocitiesEnd.data()));

    cublasDestroy(handle);
    cudaFree(devPosition);
    cudaFree(devVelocity);

    return { input.propagationStart, input.propagationSeconds, input.positions, input.velocities,
        positionsEnd, velocitiesEnd };
}

ClosestApproachOutput closestApproach(const ClosestApproachInput& input)
{
    int numStepsToFirst = input.propagationSeconds / IdealStepSize;
    double stepSize = input.propagationSeconds / static_cast<double>(numStepsToFirst);
    int totalSteps = numStepsToFirst;

    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));

    FullPerturbationParameters params { .StartTimeUTC = input.propagationStart };

    ForceModel forceModel(&handle, input.positions.size(), 1.0, &params);
    ClosestApproach closestApproach(&handle, &params, forceModel, stepSize);

    // initialize vectors with data
    closestApproach.Initialize(input.positions, input.velocities, input.setIndices);

    closestApproach.Propagate(totalSteps, true);
    ClosestApproachData data;
    closestApproach.FindClosest(data);
    std::vector<ClosestApproachRecord> records;

    // Iterate through all possible pairs
    for (size_t i = 0; i < input.positions.size(); ++i) {
        for (size_t j = i + 1; j < input.positions.size(); ++j) {
            // Only print pairs that belong to different sets
            if (input.setIndices[i] != input.setIndices[j]) {
                records.emplace_back(i, j, data.Get(i, j).distance, data.Get(i, j).time);
            }
        }
    }

    cublasDestroy(handle);

    return {
        .propagationStart = input.propagationStart,
        .propagationSeconds = input.propagationSeconds,
        .positionsStart = input.positions,
        .velocitiesStart = input.velocities,
        .closestApproaches = records,
    };
}
