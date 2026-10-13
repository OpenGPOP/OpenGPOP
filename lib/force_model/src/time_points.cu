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

#include "force_model/constants.cuh"
#include "force_model/time_points/time_points.cuh"
#include "force_model/types.cuh"
#include "force_model/utility/dataset_loaders.cuh"
#include "force_model/utility/ref_sys_trans.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "types/Parameters.hpp"
#include <cstring>
#include <cuda_runtime_api.h>

__global__ void TimePointParamsKernel(TimePointDatasets* datasets,
    TimePointTempParameters* tempParams, TimeData startTime, double stepSize, int arrayLength,
    bool useMidpoints, int beforeEpoch, double* mjd)
{
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= arrayLength)
        return;

    double timeFromStart;
    if (useMidpoints) {
        int midpointCount = CFMConst::rk8StepCount;
        int midpointIndex = tIdx % midpointCount;
        timeFromStart
            = ((double)(tIdx / midpointCount) + CFMConst::rk8Midpoints[midpointIndex]) * stepSize;
    } else {
        timeFromStart = tIdx * stepSize;
    }

    double mjd_UTC
        = RefSysTrans::CalendarToMJD(startTime.Year, startTime.Month, startTime.Day, startTime.Hour,
            startTime.Min, startTime.Sec - (double)beforeEpoch * stepSize + timeFromStart);
    mjd[tIdx] = mjd_UTC;

    // 60063.9581249998
    EOPRecord eop_record = datasets->eopDataset.getRecord(mjd_UTC);
    double mjd_UT1 = RefSysTrans::MJD_UTCtoUT1(mjd_UTC, eop_record);
    double mjd_TT = RefSysTrans::MJD_UTCtoTT(mjd_UTC, datasets->lsDataset);

    double* Q_R1 = tempParams->Q_R1.at(tIdx);
    double* Q_R2 = tempParams->Q_R2.at(tIdx);
    double* Q_R3 = tempParams->Q_R3.at(tIdx);
    double* Q_R4 = tempParams->Q_R4.at(tIdx);
    double* W_R1 = tempParams->W_R1.at(tIdx);
    double* W_R2 = tempParams->W_R2.at(tIdx);
    double* W_R3 = tempParams->W_R3.at(tIdx);
    double* R = tempParams->R.at(tIdx);

    double F[14];
    double delta_psi, eps_A;

    RefSysTrans::CreateQMatrixComponents(
        Q_R1, Q_R2, Q_R3, Q_R4, mjd_TT, datasets, F, &delta_psi, &eps_A);
    RefSysTrans::CreateWMatrixComponents(W_R1, W_R2, W_R3, mjd_TT, eop_record);
    RefSysTrans::CreateRMatrix(R, mjd_UT1, mjd_TT, delta_psi, eps_A, F);
}

void FMTimePoints::Initialize()
{
    // Read datasets and place them in device memory
    // Nutation in longitude
    DatasetLoaders::LoadNutationData(_dsFiles.nutationLon, _datasets.structure.nutDataset.lon);
    _datasets.structure.nutDataset.lon.j0.moveToDev();
    _datasets.structure.nutDataset.lon.j1.moveToDev();

    // Nutation in obliquity
    DatasetLoaders::LoadNutationData(_dsFiles.nutationObl, _datasets.structure.nutDataset.obl);
    _datasets.structure.nutDataset.obl.j0.moveToDev();
    _datasets.structure.nutDataset.obl.j1.moveToDev();

    // EOP
    DatasetLoaders::LoadEOPData(_dsFiles.eop, _datasets.structure.eopDataset);
    _datasets.structure.eopDataset.records.moveToDev();

    // Leap seconds
    DatasetLoaders::LoadLeapSeconds(_dsFiles.leapSeconds, _datasets.structure.lsDataset);
    _datasets.structure.lsDataset.leap_seconds.moveToDev();
    _datasets.structure.lsDataset.mjd_utc.moveToDev();

    _datasets.copyToDev();
}

void FMTimePoints::CalculateParameters(int arrayLength, bool useMidpoints, int beforeEpoch)
{
    // Free potentially allocated parameters
    _timePointParams.emplace(arrayLength);
    GpopStruct<TimePointTempParameters> tempParams(arrayLength);

    int threadsPerBlock = 256;
    int blockCount = arrayLength / threadsPerBlock + 1;
    TimePointParamsKernel<<<blockCount, threadsPerBlock>>>(_datasets.ptr, tempParams.ptr,
        _startTime, _stepSize, arrayLength, useMidpoints, beforeEpoch,
        _timePointParams.structure.mjd.get());
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    ComputeITRStoGCRS(arrayLength, tempParams.structure);
}

void FMTimePoints::MultiplyMatrices(int arrayLength, GpopMatrixArray<double>& A,
    GpopMatrixArray<double>& B, GpopMatrixArray<double>& out)
{
    double alpha = 1.0;
    double beta = 0.0;
    cublasErrchk(cublasDgemmStridedBatched(*_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3,
        &alpha, A.get(), 3, 9, B.get(), 3, 9, &beta, out.get(), 3, 9, arrayLength));
}

// Multiplies rotation matrices to create the final ITRS to GCRS transformation matrix (eq. 5.1)
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
void FMTimePoints::ComputeITRStoGCRS(int arrayLength, TimePointTempParameters& tempParams)
{
    // Calculate Q(t) matrix (ch. 5.4.5)
    MultiplyMatrices(arrayLength, //
        tempParams.Q_R1, //
        tempParams.Q_R2, //
        _timePointParams.structure.ITRS_to_GCRS // Store temporarily in ITRS_to_GCRS
    );
    MultiplyMatrices(arrayLength, //
        tempParams.Q_R3, //
        tempParams.Q_R4, //
        tempParams.Q_R2 // Store temporarily in Q_R2, since it's no longer needed
    );
    MultiplyMatrices(arrayLength, //
        _timePointParams.structure.ITRS_to_GCRS, // Q_R1 * Q_R2
        tempParams.Q_R2, // Q_R3 * Q_R4
        tempParams.Q_R1 // Q_R1 now contains the final Q(t)
    );

    // Calculate W(t) matrix (ch. 5.4.1)
    MultiplyMatrices(arrayLength, //
        tempParams.W_R1, //
        tempParams.W_R2, //
        _timePointParams.structure.ITRS_to_GCRS // Store temporarily in ITRS_to_GCRS
    );
    MultiplyMatrices(arrayLength, //
        _timePointParams.structure.ITRS_to_GCRS, // W_R1 * W_R2
        tempParams.W_R3, //
        tempParams.W_R1 // W_R1 now contains the final W(t)
    );
    // Calculate the final transformation matrix Q(t) * R(t) * W(t)
    MultiplyMatrices(arrayLength, //
        tempParams.Q_R1, // Q(t)
        tempParams.R, //
        tempParams.Q_R2 // Q(t) * R(t)
    );
    MultiplyMatrices(arrayLength, //
        tempParams.Q_R2, // Q(t) * R(t)
        tempParams.W_R1, // W(t)
        _timePointParams.structure.ITRS_to_GCRS // Final transformation matrix
    );
}

TimePointParametersSlice FMTimePoints::GetParametersDev(int stepNum)
{
    return { .ITRS_to_GCRS = _timePointParams.structure.ITRS_to_GCRS.get() + stepNum * 9,
        .mjd = _timePointParams.structure.mjd.get() + stepNum };
}

TimeData FMTimePoints::GetStartTime() { return _startTime; }
