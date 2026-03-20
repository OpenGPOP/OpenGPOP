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
#include "force_model/acceleration/gravity_source.cuh"
#include "force_model/acceleration/iacceleration_source.cuh"
#include "force_model/iforce_model.cuh"
#include "force_model/time_points/time_points.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

class ForceModel : public IForceModel {
public:
    ForceModel(cublasHandle_t* handle, unsigned numEntities, double stepSize,
        FullPerturbationParameters* params)
        : _handle(handle)
        , _params(params)
        , _numEntities(numEntities)
        , _stepSize(stepSize)
        , _timePoints(handle, stepSize, params->StartTimeUTC)
    {
        _accelerationSources.push_back(
            std::make_unique<GravitySource>(handle, numEntities, &_timePoints));
    };
    ~ForceModel();

    // This method should be called before any other methods of this class.
    // It initializes essential parameters and allocates global constants to device memory.
    void PreInitialize() override;

    // ----------------------------
    // Generation phase

    // This method should be called before the initial point generation phase.
    void InitializeGeneration(int beforeEpoch, int afterEpoch) override;

    // This method computes the acceleration for the initial point generation phase
    //
    // IMPORTANT: kIndex is indexed from 0
    //
    // Device pointers passed here are not managed by this class and should be allocated and freed
    // by the caller.
    void ComputeGenerationAcceleration(double* devPosition, double* devVelocity, double* devKMatrix,
        int stepNum, int kIndex) override;

    // ----------------------------
    // Correction and propagation phases

    // This method should be called after the generation phase and before the correction and
    // propagation phases.
    void InitializePropagation(int beforeEpoch, int afterEpoch) override;

    // This method overwrites the last column of the acceleration matrix. The device pointers passed
    // here are not managed by this class and should be allocated and freed by the caller.
    void ComputePropagationAcceleration(double* devPosition, double* devVelocity,
        double* devAcceleration, int stepNum, int returnColumnIndex) override;

private:
    cublasHandle_t* _handle;
    FullPerturbationParameters* _params;
    unsigned _numEntities;
    double _stepSize;

    // Matrices used to store acceleration from different sources, which are then summed
    double* _devAccelerationSumMatrix = nullptr;
    double* _devOnesVector = nullptr;

    std::vector<std::unique_ptr<IAccelerationSource>> _accelerationSources;

    FMTimePoints _timePoints;
};