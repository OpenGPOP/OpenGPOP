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

#pragma once

#include "cublas_v2.h"
#include "gpu_utils/gpu_utils.cuh"
#include <cuda_runtime.h>
#include <tuple>

class PerturbationGenerator {
    double epochPositionHost[3];
    double epochVelocityHost[3];

public:
    PerturbationGenerator(double epochPosition[3], double epochVelocity[3])
    {
        for (int i = 0; i < 3; ++i) {
            this->epochPositionHost[i] = epochPosition[i];
            this->epochVelocityHost[i] = epochVelocity[i];
        }
    }

    ~PerturbationGenerator() { }

    // In the object pointers are not null, they will be added to the end of the perturbed
    // satellites. If they are null, the perturbed satellites will be generated without them.
    std::tuple<double*, double*> generatePerturbedSatellites(int numSatellites,
        double perturbationStrength, double* objectPosition = nullptr,
        double* objectVelocity = nullptr);
};
