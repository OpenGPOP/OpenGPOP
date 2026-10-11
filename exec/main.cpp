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

#include "closest_approach/closest_approach.cuh"
#include "gpu_propagator/gpu_propagator.cuh"
#include "runner.cuh"
#include <CLI/CLI.hpp>
#include <fstream>
#include <indicators/progress_bar.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TimeData, Year, Month, Day, Hour, Min, Sec);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    PropagationInput, propagationStart, propagationSeconds, positions, velocities);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(PropagationOutput, propagationStart,
    propagationSeconds, positionsStart, velocitiesStart, positionsEnd, velocitiesEnd);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ClosestApproachRecord, objectIndex1, objectIndex2, distance,
    time, positionAx, positionAy, positionAz, positionBx, positionBy, positionBz);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    ClosestApproachInput, propagationStart, propagationSeconds, positions, velocities, setIndices);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(ClosestApproachOutput, propagationStart,
    propagationSeconds, positionsStart, velocitiesStart, closestApproaches);

void PrintSatellites(double* positionsHost, double* velocitiesHost, int numSatellites)
{
    for (int i = 0; i < numSatellites; i += 1) {
        std::cout << "Satellite " << i << ": Position: (" << positionsHost[i * 3] << ", "
                  << positionsHost[i * 3 + 1] << ", " << positionsHost[i * 3 + 2]
                  << "), Velocity: (" << velocitiesHost[i * 3] << ", " << velocitiesHost[i * 3 + 1]
                  << ", " << velocitiesHost[i * 3 + 2] << ")\n";
    }
}

int main(int argc, char** argv)
{
    CLI::App app { "Open GPU-Parallelised Orbit Propagator" };
    app.require_subcommand(1); // Force user to choose either propagate or closest-approach

    std::string inputFile;
    std::string propagationOutputFile;
    std::string closestOutputFile;

    DatasetFilePaths dsFiles;
    auto registerDatasetFlags = [](CLI::App* sub, DatasetFilePaths& paths) {
        auto* group
            = sub->add_option_group("Data Files", "Astronomical and gravitational model datasets");
        group->add_option("--gravity-model", paths.gravityModel, "Path to gravity model (.gfc)")
            ->default_val(paths.gravityModel)
            ->check(CLI::ExistingFile);
        group
            ->add_option(
                "--nutation-lon", paths.nutationLon, "Path to IAU2000 nutation longitude file")
            ->default_val(paths.nutationLon)
            ->check(CLI::ExistingFile);
        group
            ->add_option(
                "--nutation-obl", paths.nutationObl, "Path to IAU2000 nutation obliquity file")
            ->default_val(paths.nutationObl)
            ->check(CLI::ExistingFile);
        group->add_option("--eop", paths.eop, "Path to Earth Orientation Parameters (EOP) file")
            ->default_val(paths.eop)
            ->check(CLI::ExistingFile);
        group->add_option("--leap-seconds", paths.leapSeconds, "Path to leap seconds file")
            ->default_val(paths.leapSeconds)
            ->check(CLI::ExistingFile);
    };

    CLI::App* cmdPropagate = app.add_subcommand("propagate", "Run orbit propagation");
    cmdPropagate->add_option("-i,--input", inputFile, "Input JSON file")
        ->required()
        ->check(CLI::ExistingFile); // CLI11 automatically checks if file exists!
    cmdPropagate->add_option("-o,--output", propagationOutputFile, "Output JSON file")
        ->default_val("propagation_out.json");
    registerDatasetFlags(cmdPropagate, dsFiles);

    CLI::App* cmdClosest
        = app.add_subcommand("closest-approach", "Calculate closest approaches between objects");
    cmdClosest->add_option("-i,--input", inputFile, "Input JSON file")
        ->required()
        ->check(CLI::ExistingFile);
    cmdClosest->add_option("-o,--output", closestOutputFile, "Output JSON file")
        ->default_val("closest_approach_out.json");
    registerDatasetFlags(cmdClosest, dsFiles);

    CLI11_PARSE(app, argc, argv);

    if (cmdPropagate->parsed()) {
        std::cout << "Running Propagation...\n";

        std::ifstream ifs(inputFile);
        json jInput = json::parse(ifs);
        PropagationInput inputData = jInput.get<PropagationInput>();

        PropagationOutput outputData = propagate(inputData, dsFiles);

        std::ofstream ofs(propagationOutputFile);
        json jOutput = outputData;
        ofs << jOutput.dump(4);
        std::cout << "Saved to: " << propagationOutputFile << "\n";
    } else if (cmdClosest->parsed()) {
        std::cout << "Running Closest Approach Calculation...\n";

        std::ifstream ifs(inputFile);
        json jInput = json::parse(ifs);
        ClosestApproachInput inputData = jInput.get<ClosestApproachInput>();

        ClosestApproachOutput outputData = closestApproach(inputData, dsFiles);

        std::ofstream ofs(closestOutputFile);
        json jOutput = outputData;
        ofs << jOutput.dump(4);
        std::cout << "Saved to: " << closestOutputFile << "\n";
    }

    return 0;
}
