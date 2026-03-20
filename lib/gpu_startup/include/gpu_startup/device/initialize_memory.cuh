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

#define MATRIX_INDEX(sat, row, col) ((sat) * 3 * 9 + (row) * 3 + (col))
#define VECTOR_INDEX(sat, col) ((sat) * 3 + (col))
#define KMATRIX_INDEX(sat, row, col) ((sat) * 3 * 10 + (row) * 3 + (col))

__global__ void initialize_memory(double* positions, double* velocities, double* positionMatrix,
    double* velocityMatrix, double* temp_x, double* temp_v, uint numSatellites);