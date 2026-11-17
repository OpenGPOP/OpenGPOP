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
#include "force_model/acceleration/iacceleration_source.cuh"
#include "force_model/time_points/itime_points.cuh"
#include "force_model/types.cuh"
#include "types/Parameters.hpp"
#include <string>

// Computes xi(n,m), eta(n,m), and zeta(n,m) tables of auxiliary coefficients (eq. 7-10, 7-19)
// Run with TRIANGULAR_SIZE(N) threads.
__global__ void ComputeAuxiliaryCoefficientsKernel(GravityDataset* dataset);

// Copies the Pbar diagonal elements from the first to all other entities
// Run with numEntities - 1 threads.
__global__ void PopulatePbarDiagonalKernel(GravityDataset* dataset, int numEntities);

// Computes the table of normalized derived Legendre polynomials (Pbar)
// Run with numEntities * N threads.
__global__ void ComputePbarKernel(GravityDataset* dataset, double* positions, int numEntities);

// Computes the Cm_bar and Sm_bar tables
// Run with numEntities threads.
__global__ void ComputeCmSmKernel(GravityDataset* dataset, double* positions, int numEntities);

// Prepares summation elements for Gn, Hn, Jn, and Kn (eq. 4-11)
// Run with numEntities * TRIANGULAR_SIZE(N) threads.
__global__ void PrepareMSumKernel(GravityDataset* dataset, int numEntities, double* mjd);

// Prepares summation elements for G, H, J, and K (eq. 4-12)
// Run with numEntities threads.
__global__ void PrepareNSumKernel(GravityDataset* dataset, double* positions, int numEntities);

// Calculates accelerations of the entities
// Run with numEntities threads.
__global__ void CalculateAcceleration(
    GravityDataset* dataset, double* positions, int numEntities, double* outAccelerations);

// Harmonic gravity acceleration
//
// Based on:
// Gottlieb, R. G. (1993). Fast gravity, gravity partials, normalized gravity, gravity gradient
// torque and magnetic field: derivation, code and data (No. NAS 1.26: 188243).
// https://ntrs.nasa.gov/api/citations/19940025085/downloads/19940025085.pdf
class GravitySource : public IAccelerationSource {
public:
    GravitySource(cublasHandle_t* handle, int numEntities, IFMTimePoints* time_points,
        DatasetFilePaths& dsFiles)
        : _handle(handle)
        , _numEntities(numEntities)
        , _time_points(time_points)
        , _dsFiles(dsFiles)
    {
    }
    ~GravitySource() { FreeDataset(); }

    virtual void Initialize() override;
    virtual void ComputeAcceleration(
        double* devPosition, double* devVelocity, int stepNum, double* out, int outStride) override;

private:
    cublasHandle_t* _handle;
    int _numEntities;
    IFMTimePoints* _time_points;
    DatasetFilePaths& _dsFiles;

    GravityDataset _dataset;
    GravityDataset* _devDataset = nullptr;
    void FreeDataset();

    double* _devPositionInITRS = nullptr;
    double* _devAccelerationInITRS = nullptr;

    void SumM(double* devMjd);
    void SumN();
};