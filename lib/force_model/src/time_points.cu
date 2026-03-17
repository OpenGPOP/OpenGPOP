#include "force_model/constants.cuh"
#include "force_model/time_points/time_points.cuh"
#include "force_model/types.cuh"
#include "force_model/utility/dataset_loaders.cuh"
#include "force_model/utility/ref_sys_trans.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include <cstring>

__global__ void TimePointParamsKernel(TimePointDatasets* datasets,
    TimePointTempParameters* tempParams, StartTime startTime, double stepSize, int arrayLength,
    bool useMidpoints, int beforeEpoch)
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

    EOPRecord eop_record = datasets->eopDataset.getRecord(mjd_UTC);
    double mjd_UT1 = RefSysTrans::MJD_UTCtoUT1(mjd_UTC, eop_record);
    double mjd_TT = RefSysTrans::MJD_UTCtoTT(mjd_UTC, datasets->lsDataset);

    int offset = tIdx * 9;
    double* Q_R1 = tempParams->Q_R1 + offset;
    double* Q_R2 = tempParams->Q_R2 + offset;
    double* Q_R3 = tempParams->Q_R3 + offset;
    double* Q_R4 = tempParams->Q_R4 + offset;
    double* W_R1 = tempParams->W_R1 + offset;
    double* W_R2 = tempParams->W_R2 + offset;
    double* W_R3 = tempParams->W_R3 + offset;
    double* R = tempParams->R + offset;

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
    auto nutLon = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutLon.txt");
    _datasets.nutDataset.lon = nutLon; // Create a shallow copy of non-array parameters
    gpuErrchk(cudaMalloc(&_datasets.nutDataset.lon.j0, nutLon.j0_size * sizeof(NutationTerm)));
    gpuErrchk(cudaMalloc(&_datasets.nutDataset.lon.j1, nutLon.j1_size * sizeof(NutationTerm)));
    gpuErrchk(cudaMemcpy(_datasets.nutDataset.lon.j0, nutLon.j0,
        nutLon.j0_size * sizeof(NutationTerm), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_datasets.nutDataset.lon.j1, nutLon.j1,
        nutLon.j1_size * sizeof(NutationTerm), cudaMemcpyHostToDevice));
    delete[] nutLon.j0;
    delete[] nutLon.j1;

    // Nutation in obliquity
    auto nutObl = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutObl.txt");
    _datasets.nutDataset.obl = nutObl; // Create a shallow copy of non-array parameters
    gpuErrchk(cudaMalloc(&_datasets.nutDataset.obl.j0, nutObl.j0_size * sizeof(NutationTerm)));
    gpuErrchk(cudaMalloc(&_datasets.nutDataset.obl.j1, nutObl.j1_size * sizeof(NutationTerm)));
    gpuErrchk(cudaMemcpy(_datasets.nutDataset.obl.j0, nutObl.j0,
        nutObl.j0_size * sizeof(NutationTerm), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_datasets.nutDataset.obl.j1, nutObl.j1,
        nutObl.j1_size * sizeof(NutationTerm), cudaMemcpyHostToDevice));
    delete[] nutObl.j0;
    delete[] nutObl.j1;

    // EOP
    auto eop = DatasetLoaders::LoadEOPData("../../data/IAU2000A_EOP.txt");
    _datasets.eopDataset = eop; // Create a shallow copy of non-array parameters
    gpuErrchk(cudaMalloc(&_datasets.eopDataset.records, eop.size * sizeof(EOPRecord)));
    gpuErrchk(cudaMemcpy(_datasets.eopDataset.records, eop.records, eop.size * sizeof(EOPRecord),
        cudaMemcpyHostToDevice));
    delete[] eop.records;

    // Leap seconds
    auto ls = DatasetLoaders::LoadLeapSeconds("../../data/leap-seconds.list");
    _datasets.lsDataset = ls; // Create a shallow copy of non-array parameters
    gpuErrchk(cudaMalloc(&_datasets.lsDataset.leap_seconds, ls.size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_datasets.lsDataset.mjd_utc, ls.size * sizeof(double)));
    gpuErrchk(cudaMemcpy(_datasets.lsDataset.leap_seconds, ls.leap_seconds,
        ls.size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
        _datasets.lsDataset.mjd_utc, ls.mjd_utc, ls.size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] ls.leap_seconds;
    delete[] ls.mjd_utc;

    // Allocate space for the containing structure and copy it
    gpuErrchk(cudaMalloc(&_devDatasets, sizeof(TimePointDatasets)));
    gpuErrchk(
        cudaMemcpy(_devDatasets, &_datasets, sizeof(TimePointDatasets), cudaMemcpyHostToDevice));
}

void FMTimePoints::CalculateParameters(int arrayLength, bool useMidpoints, int beforeEpoch)
{
    // Free potentially allocated parameters
    FreeParameters();

    AllocateParameters(arrayLength);
    AllocateTempParameters(arrayLength);

    int threadsPerBlock = 256;
    int blockCount = arrayLength / threadsPerBlock + 1;
    TimePointParamsKernel<<<blockCount, threadsPerBlock>>>(_devDatasets, _devTempInitParameters,
        _startTime, _stepSize, arrayLength, useMidpoints, beforeEpoch);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    ComputeITRStoGCRS(arrayLength);

    FreeTempParameters();
}

void FMTimePoints::MultiplyMatrices(int arrayLength, double* A, double* B, double* out)
{
    double alpha = 1.0;
    double beta = 0.0;
    cublasErrchk(cublasDgemmStridedBatched(*_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3,
        &alpha, A, 3, 9, B, 3, 9, &beta, out, 3, 9, arrayLength));
}

// Multiplies rotation matrices to create the final ITRS to GCRS transformation matrix (eq. 5.1)
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
void FMTimePoints::ComputeITRStoGCRS(int arrayLength)
{
    // Calculate Q(t) matrix (ch. 5.4.5)
    MultiplyMatrices(arrayLength, //
        _tempInitParameters.Q_R1, //
        _tempInitParameters.Q_R2, //
        _timePointParameters.ITRS_to_GCRS // Store temporarily in ITRS_to_GCRS
    );
    MultiplyMatrices(arrayLength, //
        _tempInitParameters.Q_R3, //
        _tempInitParameters.Q_R4, //
        _tempInitParameters.Q_R2 // Store temporarily in Q_R2, since it's no longer needed
    );
    MultiplyMatrices(arrayLength, //
        _timePointParameters.ITRS_to_GCRS, // Q_R1 * Q_R2
        _tempInitParameters.Q_R2, // Q_R3 * Q_R4
        _tempInitParameters.Q_R1 // Q_R1 now contains the final Q(t)
    );

    // Calculate W(t) matrix (ch. 5.4.1)
    MultiplyMatrices(arrayLength, //
        _tempInitParameters.W_R1, //
        _tempInitParameters.W_R2, //
        _timePointParameters.ITRS_to_GCRS // Store temporarily in ITRS_to_GCRS
    );
    MultiplyMatrices(arrayLength, //
        _timePointParameters.ITRS_to_GCRS, // W_R1 * W_R2
        _tempInitParameters.W_R3, //
        _tempInitParameters.W_R1 // W_R1 now contains the final W(t)
    );
    // Calculate the final transformation matrix Q(t) * R(t) * W(t)
    MultiplyMatrices(arrayLength, //
        _tempInitParameters.Q_R1, // Q(t)
        _tempInitParameters.R, //
        _tempInitParameters.Q_R2 // Q(t) * R(t)
    );
    MultiplyMatrices(arrayLength, //
        _tempInitParameters.Q_R2, // Q(t) * R(t)
        _tempInitParameters.W_R1, // W(t)
        _timePointParameters.ITRS_to_GCRS // Final transformation matrix
    );
}

TimePointParameters FMTimePoints::GetParametersDev(int stepNum)
{
    return { .ITRS_to_GCRS = _timePointParameters.ITRS_to_GCRS + stepNum * 9 };
}

void FMTimePoints::FreeDatasets()
{
    CUDA_SAFE_FREE(_datasets.nutDataset.lon.j0);
    CUDA_SAFE_FREE(_datasets.nutDataset.lon.j1);

    CUDA_SAFE_FREE(_datasets.nutDataset.obl.j0);
    CUDA_SAFE_FREE(_datasets.nutDataset.obl.j1);

    CUDA_SAFE_FREE(_datasets.eopDataset.records);

    CUDA_SAFE_FREE(_datasets.lsDataset.leap_seconds);
    CUDA_SAFE_FREE(_datasets.lsDataset.mjd_utc);

    CUDA_SAFE_FREE(_devDatasets);
}

void FMTimePoints::AllocateParameters(int arrayLength)
{
    gpuErrchk(cudaMalloc(&_timePointParameters.ITRS_to_GCRS, sizeof(double) * 3 * 3 * arrayLength));
    gpuErrchk(cudaMalloc(&_devTimePointParameters, sizeof(TimePointParameters)));

    // Copy the struct to device
    gpuErrchk(cudaMemcpy(_devTimePointParameters, &_timePointParameters,
        sizeof(TimePointParameters), cudaMemcpyHostToDevice));
}

void FMTimePoints::FreeParameters()
{
    CUDA_SAFE_FREE(_timePointParameters.ITRS_to_GCRS);
    CUDA_SAFE_FREE(_devTimePointParameters);
}

void FMTimePoints::AllocateTempParameters(int arrayLength)
{
    gpuErrchk(cudaMalloc(&_tempInitParameters.Q_R1, sizeof(double) * 9 * arrayLength));
    gpuErrchk(cudaMalloc(&_tempInitParameters.Q_R2, sizeof(double) * 9 * arrayLength));
    gpuErrchk(cudaMalloc(&_tempInitParameters.Q_R3, sizeof(double) * 9 * arrayLength));
    gpuErrchk(cudaMalloc(&_tempInitParameters.Q_R4, sizeof(double) * 9 * arrayLength));

    gpuErrchk(cudaMalloc(&_tempInitParameters.W_R1, sizeof(double) * 9 * arrayLength));
    gpuErrchk(cudaMalloc(&_tempInitParameters.W_R2, sizeof(double) * 9 * arrayLength));
    gpuErrchk(cudaMalloc(&_tempInitParameters.W_R3, sizeof(double) * 9 * arrayLength));

    gpuErrchk(cudaMalloc(&_tempInitParameters.R, sizeof(double) * 9 * arrayLength));

    // Allocate struct
    gpuErrchk(cudaMalloc(&_devTempInitParameters, sizeof(TimePointTempParameters)));

    // Copy the struct to device
    gpuErrchk(cudaMemcpy(_devTempInitParameters, &_tempInitParameters,
        sizeof(TimePointTempParameters), cudaMemcpyHostToDevice));
}

void FMTimePoints::FreeTempParameters()
{
    CUDA_SAFE_FREE(_tempInitParameters.Q_R1);
    CUDA_SAFE_FREE(_tempInitParameters.Q_R2);
    CUDA_SAFE_FREE(_tempInitParameters.Q_R3);
    CUDA_SAFE_FREE(_tempInitParameters.Q_R4);

    CUDA_SAFE_FREE(_tempInitParameters.W_R1);
    CUDA_SAFE_FREE(_tempInitParameters.W_R2);
    CUDA_SAFE_FREE(_tempInitParameters.W_R3);

    CUDA_SAFE_FREE(_tempInitParameters.R);

    CUDA_SAFE_FREE(_devTempInitParameters);
}