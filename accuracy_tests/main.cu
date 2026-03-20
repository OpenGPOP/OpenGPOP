// Open GPU-Parallelised Orbit Propagator (OpenGPOP)
// Copyright (C) 2026 Igor Januszkiewicz, Michał Kiedrzyński
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// INFO
// format of the test case line in test cases file:
// <epoch x> <epoch y> <epoch z> <epoch dx> <epoch dy> <epoch dz> <year> <month>
// <day> <minute> <second> <delta seconds> <second x> <second y> <second z>
// <orekit error>

#include "force_model/force_model.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "progressbar/progressbar.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

constexpr double IdealStepSize = 30;

struct TestCase {
    double epochPosition[3];
    double epochVelocity[3];
    double expectedPosition[3];

    int year, month, day, hour, minute, second;
    double secondsSinceEpoch;
    double orekitError;
};

struct PlatformTestResult {
    double error;
    double calculationTime;
};

struct FullTestCaseResult {
    PlatformTestResult gpuResult;
    double secondsSinceEpoch;
    double orekitError;
};

double distance(double position1[3], double position2[3])
{
    return std::sqrt(std::pow(position1[0] - position2[0], 2)
        + std::pow(position1[1] - position2[1], 2) + std::pow(position1[2] - position2[2], 2));
}

void PrintSingleResult(const FullTestCaseResult& result)
{
    std::cout << "Second since epoch:       " << result.secondsSinceEpoch << std::endl;
    std::cout << "Orekit Error:             " << result.orekitError << std::endl;
    std::cout << "GPU Error:                " << result.gpuResult.error << std::endl;
    std::cout << "GPU calculation time [s]: " << result.gpuResult.calculationTime << std::endl;
}

void PrintResults(const std::vector<FullTestCaseResult>& testResults)
{
    std::cout << "RESULTS:" << std::endl;
    for (unsigned long int i = 0; i < testResults.size(); i++) {
        std::cout << "Test case " << i << ":" << std::endl;
        PrintSingleResult(testResults[i]);
        std::cout << std::endl;
    }
}

PlatformTestResult PerformGpuTest(
    TestCase testCase, FullPerturbationParameters* parameters, double stepSize, int totalSteps)
{
    cublasHandle_t handle;
    cublasErrchk(cublasCreate(&handle));
    ForceModel forceModel(&handle, 1, stepSize, parameters);

    double* devPosition;
    double* devVelocity;

    cudaMallocManaged(&devPosition, 3 * sizeof(double));
    cudaMallocManaged(&devVelocity, 3 * sizeof(double));

    cudaMemcpy(devPosition, testCase.epochPosition, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocity, testCase.epochVelocity, 3 * sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "Initializing propagator..." << std::endl;
    std::cout.flush();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    GpuPropagator propagator(
        devPosition, devVelocity, 1, &handle, stepSize, parameters, forceModel, totalSteps, 7);

    double currentPosition[3];
    double currentVelocity[3];

    double minDistance = std::numeric_limits<double>::max();
    std::cout << "Starting propagation..." << std::endl;
    std::cout.flush();
    progressbar bar(totalSteps);
    for (int i = 0; i < totalSteps; i++) {
        bar.update();
        propagator.Step(currentPosition, currentVelocity);
        double currentDistance = distance(currentPosition, testCase.expectedPosition);
        if (currentDistance < minDistance) {
            minDistance = currentDistance;
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::endl;
    std::cout << "Propagation done." << std::endl;

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    cublasDestroy(handle);
    cudaFree(devPosition);
    cudaFree(devVelocity);
    return { minDistance, static_cast<double>(elapsed) };
}

FullTestCaseResult PerformTest(TestCase testCase)
{
    int numStepsToFirst = testCase.secondsSinceEpoch / IdealStepSize;
    double timeStep = testCase.secondsSinceEpoch / static_cast<double>(numStepsToFirst);
    int totalSteps = numStepsToFirst + 2;

    std::cout << "Test case info: " << std::endl;
    std::cout << "Step size [s]: " << timeStep << std::endl;
    std::cout << "Total steps:   " << totalSteps << std::endl;

    FullPerturbationParameters params;
    params.StartTimeUTC.Year = testCase.year;
    params.StartTimeUTC.Month = testCase.month;
    params.StartTimeUTC.Day = testCase.day;
    params.StartTimeUTC.Hour = testCase.hour;
    params.StartTimeUTC.Min = testCase.minute;
    params.StartTimeUTC.Sec = testCase.second;

    params.Area_drag = 55.64;
    params.Area_solar = 88.4;
    params.mass = 2128; // [kg]
    params.CR = 1.0;
    params.CD = 2.7;
    params.n = 70;
    params.m = 70;
    params.Drag = false;

    std::cout << "Performing gpu test..." << std::endl;
    PlatformTestResult gpuResult = PerformGpuTest(testCase, &params, timeStep, totalSteps);
    std::cout << "Gpu test done" << std::endl;
    FullTestCaseResult fullResult = { gpuResult, testCase.secondsSinceEpoch, testCase.orekitError };
    PrintSingleResult(fullResult);
    return fullResult;
}

int main()
{
    double x, y, z, xd, yd, zd, secondsSinceEpoch, secondX, secondY, secondZ, orekitError;
    int yr, mo, d, hr, mi, sec;

    std::vector<TestCase> testCases;
    std::ifstream in("../../data/state_vectors.txt");
    while (in >> x >> y >> z >> xd >> yd >> zd >> yr >> mo >> d >> hr >> mi >> sec
        >> secondsSinceEpoch >> secondX >> secondY >> secondZ >> orekitError) {
        testCases.push_back({ { x, y, z }, { xd, yd, zd }, { secondX, secondY, secondZ }, yr, mo, d,
            hr, mi, sec, secondsSinceEpoch, orekitError });
    }
    std::cout << "Read " << testCases.size() << " test cases" << std::endl;
    std::cout << "Testing..." << std::endl;
    std::vector<FullTestCaseResult> results(testCases.size());
    for (unsigned long int i = 0; i < testCases.size(); i++) {
        std::cout << "Testing case: " << i << std::endl;
        results[i] = PerformTest(testCases[i]);
        std::cout << "Test case " << i << " done" << std::endl << std::endl;
    }
    std::cout << "All test cases done" << std::endl;
    PrintResults(results);
}
