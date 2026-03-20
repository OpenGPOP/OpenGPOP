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

struct IForceModel {
    virtual void PreInitialize() = 0;

    virtual void InitializeGeneration(int beforeEpoch, int afterEpoch) = 0;

    virtual void ComputeGenerationAcceleration(
        double* devPosition, double* devVelocity, double* devKMatrix, int stepNum, int kIndex)
        = 0;

    virtual void InitializePropagation(int beforeEpoch, int afterEpoch) = 0;

    virtual void ComputePropagationAcceleration(double* devPosition, double* devVelocity,
        double* devAcceleration, int stepNum, int returnColumnIndex)
        = 0;
};