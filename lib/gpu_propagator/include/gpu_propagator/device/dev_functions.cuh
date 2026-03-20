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

#include <cuda_runtime.h>

__global__ void CopyStartupStateKernel(double* positions, double* velocities,
    double* positionMatrix, double* velocityMatrix, int stepNum, uint numSatellites);

__global__ void FinalizeCorrectionKernel(double* positions, double* velocities,
    double* accelerationMatrix, double* smallS, double* bigS, double* a, double* b, double timeStep,
    uint numSatellites);

__global__ void MoveColumnToVectorKernel(
    double* matrix, double* vector, int columnIndex, uint numSatellites);