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

#include "force_model/acceleration/gravity_source.cuh"
#include "force_model/utility/dataset_loaders.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include <cmath>
#include <cstring>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

//                end of segment -> |
// in  = 0 1 2 3 4 5 6 7 8 9 ... N-1 N N+1 N+2 ...
// out = 0 0 0 0 0 0 0 0 0 0 ...  0  1  1   1  ...
struct SegmentLambda {
    int N;
    SegmentLambda(int N)
        : N(N)
    {
    }
    __host__ __device__ int operator()(int i) const { return i / N; }
};

//                          end of cycle -> |
// in  = 0 1 2 3 4 5 6 7 8 9 ...  i  ...  j  j+1 j+2 j+3 j+4 ...
// out = 0 1 1 2 2 2 3 3 3 3 ... N-1 ... N-1  0   1   1   2  ...
// where j = i + N - 1
struct TriangularCyclicLambda {
    const int N;
    const int cycleLength;
    TriangularCyclicLambda(int N)
        : N(N)
        , cycleLength(N * (N + 1) / 2) // arithmetic series
    {
    }
    __host__ __device__ int operator()(int i) const
    {
        int cycleIndex = i % cycleLength;

        // since it's an arithmetic series, we're looking for a positive solution to x(x+1)/2 == N
        double x = (sqrt(8.0 * cycleIndex + 1.0) - 1.0) / 2.0;
        return (int)floor(x);
    }
};

void GravitySource::Initialize()
{
    auto data = DatasetLoaders::LoadGravityModel("../../data/EIGEN-6S4_(v2).gfc");
    _dataset = data; // make a shallow copy to fill non-array elements

    size_t size = TRI_SIZE(data.N);

    // Copy C and S coefficients
    gpuErrchk(cudaMalloc(&_dataset.Cnm, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.Snm, size * sizeof(double)));
    gpuErrchk(cudaMemcpy(_dataset.Cnm, data.Cnm, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_dataset.Snm, data.Snm, size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] data.Cnm;
    delete[] data.Snm;

    // Copy time-variable flag
    gpuErrchk(cudaMalloc(&_dataset.is_gfct, size * sizeof(bool)));
    gpuErrchk(
        cudaMemcpy(_dataset.is_gfct, data.is_gfct, size * sizeof(bool), cudaMemcpyHostToDevice));
    delete[] data.is_gfct;

    // Copy reference epoch
    gpuErrchk(cudaMalloc(&_dataset.t0, size * sizeof(double)));
    gpuErrchk(cudaMemcpy(_dataset.t0, data.t0, size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] data.t0;

    // Copy secular trends
    gpuErrchk(cudaMalloc(&_dataset.trnd_C, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.trnd_S, size * sizeof(double)));
    gpuErrchk(
        cudaMemcpy(_dataset.trnd_C, data.trnd_C, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(_dataset.trnd_S, data.trnd_S, size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] data.trnd_C;
    delete[] data.trnd_S;

    // Copy annual periodic coefficients (1.0 yr)
    gpuErrchk(cudaMalloc(&_dataset.acos1_C, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.asin1_C, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.acos1_S, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.asin1_S, size * sizeof(double)));
    gpuErrchk(
        cudaMemcpy(_dataset.acos1_C, data.acos1_C, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(_dataset.asin1_C, data.asin1_C, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(_dataset.acos1_S, data.acos1_S, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(_dataset.asin1_S, data.asin1_S, size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] data.acos1_C;
    delete[] data.asin1_C;
    delete[] data.acos1_S;
    delete[] data.asin1_S;

    // Copy semi-annual periodic coefficients (0.5 yr)
    gpuErrchk(cudaMalloc(&_dataset.acos05_C, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.asin05_C, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.acos05_S, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.asin05_S, size * sizeof(double)));
    gpuErrchk(cudaMemcpy(
        _dataset.acos05_C, data.acos05_C, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
        _dataset.asin05_C, data.asin05_C, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
        _dataset.acos05_S, data.acos05_S, size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(
        _dataset.asin05_S, data.asin05_S, size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] data.acos05_C;
    delete[] data.asin05_C;
    delete[] data.acos05_S;
    delete[] data.asin05_S;

    // Calculate the delta function and the diagonal of the normalized derived Legendre functions.
    auto delta = new double[data.N];
    auto P_bar = new double[size];
    memset(P_bar, 0xEE, sizeof(double) * size);
    P_bar[TRI_C2IDX(0, 0)] = 1;
    P_bar[TRI_C2IDX(1, 1)] = sqrt(3);
    // delta[0] stays invalid but it will never be used
    delta[1] = sqrt(3);
    for (int i = 2; i < data.N; i++) {
        auto previous = P_bar[TRI_C2IDX(i - 1, i - 1)];
        double tmp = 2.0 * (double)i + 1.0;
        P_bar[TRI_C2IDX(i, i)] = previous * sqrt(tmp / (2.0 * (double)i)); // eq. 7-8
        delta[i] = previous * sqrt(tmp); // eq. 7-15
    }
    gpuErrchk(cudaMalloc(&_dataset.delta, data.N * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.P_bar, size * sizeof(double) * _numEntities));
    gpuErrchk(cudaMemcpy(_dataset.delta, delta, data.N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_dataset.P_bar, P_bar, size * sizeof(double), cudaMemcpyHostToDevice));
    delete[] delta;
    delete[] P_bar;

    // Allocate and compute the rest of auxiliary coefficients
    gpuErrchk(cudaMalloc(&_dataset.xi, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.eta, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&_dataset.zeta, size * sizeof(double)));

    // Allocate temporary vectors and arrays used for computations
    gpuErrchk(cudaMalloc(&_dataset.Cm_bar, sizeof(double) * _numEntities * data.N));
    gpuErrchk(cudaMalloc(&_dataset.Sm_bar, sizeof(double) * _numEntities * data.N));

    size = sizeof(double) * _numEntities * TRI_SIZE(data.N);
    gpuErrchk(cudaMalloc(&_dataset.G, size));
    gpuErrchk(cudaMalloc(&_dataset.K, size));
    gpuErrchk(cudaMalloc(&_dataset.H, size));
    gpuErrchk(cudaMalloc(&_dataset.J, size));

    size = sizeof(double) * _numEntities * data.N;
    gpuErrchk(cudaMalloc(&_dataset.G_temp, size));
    gpuErrchk(cudaMalloc(&_dataset.K_temp, size));
    gpuErrchk(cudaMalloc(&_dataset.H_temp, size));
    gpuErrchk(cudaMalloc(&_dataset.J_temp, size));

    gpuErrchk(cudaMalloc(&_devDataset, sizeof(GravityDataset)));
    gpuErrchk(cudaMemcpy(_devDataset, &_dataset, sizeof(GravityDataset), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&_devPositionInITRS, sizeof(double) * _numEntities * 3));
    gpuErrchk(cudaMalloc(&_devAccelerationInITRS, sizeof(double) * _numEntities * 3));

    int threadsPerBlock = 256;
    int blockCount = TRI_SIZE(data.N) / threadsPerBlock + 1;
    ComputeAuxiliaryCoefficientsKernel<<<blockCount, threadsPerBlock>>>(_devDataset);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Make copies of Pbar diagonal for all entities, for simplicity
    blockCount = (_numEntities - 1) / threadsPerBlock + 1;
    PopulatePbarDiagonalKernel<<<blockCount, threadsPerBlock>>>(_devDataset, _numEntities);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void GravitySource::FreeDataset()
{
    CUDA_SAFE_FREE(_dataset.Cnm);
    CUDA_SAFE_FREE(_dataset.Snm);

    CUDA_SAFE_FREE(_dataset.is_gfct);
    CUDA_SAFE_FREE(_dataset.t0);

    CUDA_SAFE_FREE(_dataset.trnd_C);
    CUDA_SAFE_FREE(_dataset.trnd_S);

    CUDA_SAFE_FREE(_dataset.acos1_C);
    CUDA_SAFE_FREE(_dataset.asin1_C);
    CUDA_SAFE_FREE(_dataset.acos1_S);
    CUDA_SAFE_FREE(_dataset.asin1_S);

    CUDA_SAFE_FREE(_dataset.acos05_C);
    CUDA_SAFE_FREE(_dataset.asin05_C);
    CUDA_SAFE_FREE(_dataset.acos05_S);
    CUDA_SAFE_FREE(_dataset.asin05_S);

    CUDA_SAFE_FREE(_dataset.Cm_bar);
    CUDA_SAFE_FREE(_dataset.Sm_bar);

    CUDA_SAFE_FREE(_dataset.xi);
    CUDA_SAFE_FREE(_dataset.eta);
    CUDA_SAFE_FREE(_dataset.zeta);
    CUDA_SAFE_FREE(_dataset.delta);

    CUDA_SAFE_FREE(_dataset.P_bar);

    CUDA_SAFE_FREE(_dataset.G);
    CUDA_SAFE_FREE(_dataset.H);
    CUDA_SAFE_FREE(_dataset.J);
    CUDA_SAFE_FREE(_dataset.K);

    CUDA_SAFE_FREE(_dataset.G_temp);
    CUDA_SAFE_FREE(_dataset.H_temp);
    CUDA_SAFE_FREE(_dataset.J_temp);
    CUDA_SAFE_FREE(_dataset.K_temp);

    CUDA_SAFE_FREE(_devDataset);

    CUDA_SAFE_FREE(_devPositionInITRS);
    CUDA_SAFE_FREE(_devAccelerationInITRS);
}

void GravitySource::ComputeAcceleration(
    double* devPosition, double* devVelocity, int stepNum, double* out, int outStride)
{
    auto timePoint = _time_points->GetParametersDev(stepNum);

    // Change position frame from GCRS to ITRS (using the transpose)
    double alpha = 1.0;
    double beta = 0.0;
    cublasErrchk(
        cublasDgemvStridedBatched(*_handle, CUBLAS_OP_T, 3, 3, &alpha, timePoint.ITRS_to_GCRS, 3, 0,
            devPosition, 1, 3, &beta, _devPositionInITRS, 1, 3, _numEntities));

    // Compute the table of normalized derived Legendre polynomials (Pbar)
    int threadsPerBlock = 256;
    int numOfThreads = _numEntities * _dataset.N;
    int blockCount = numOfThreads / threadsPerBlock + 1;
    ComputePbarKernel<<<blockCount, threadsPerBlock>>>(
        _devDataset, _devPositionInITRS, _numEntities);

    // Compute Cm_bar and Sm_bar tables
    numOfThreads = _numEntities;
    blockCount = numOfThreads / threadsPerBlock + 1;
    ComputeCmSmKernel<<<blockCount, threadsPerBlock>>>(
        _devDataset, _devPositionInITRS, _numEntities);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    SumM();
    SumN();

    // Calculate acceleration
    numOfThreads = _numEntities;
    blockCount = numOfThreads / threadsPerBlock + 1;
    CalculateAcceleration<<<blockCount, threadsPerBlock>>>(
        _devDataset, _devPositionInITRS, _numEntities, _devAccelerationInITRS);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Change position frame from ITRS to GCRS save to the summation matrix
    cublasErrchk(
        cublasDgemvStridedBatched(*_handle, CUBLAS_OP_N, 3, 3, &alpha, timePoint.ITRS_to_GCRS, 3, 0,
            _devAccelerationInITRS, 1, 3, &beta, out, 1, outStride * 3, _numEntities));
}

double GravitySource::GetDecimalYear()
{
    auto start_time = _time_points->GetStartTime();
    const int days_before_month[] = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };
    int m_idx = (start_time.Month >= 1 && start_time.Month <= 12) ? (start_time.Month - 1) : 0;
    double day_frac = (start_time.Day - 1.0)
        + (start_time.Hour + (start_time.Min + start_time.Sec / 60.0) / 60.0) / 24.0;
    return start_time.Year + ((days_before_month[m_idx] + day_frac) / 365.25);
}

void GravitySource::SumM()
{
    // Prepare M summation
    int threadsPerBlock = 256;
    int numOfThreads = _numEntities * TRI_SIZE(_dataset.N);
    int blockCount = numOfThreads / threadsPerBlock + 1;
    PrepareMSumKernel<<<blockCount, threadsPerBlock>>>(_devDataset, _numEntities, GetDecimalYear());
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Sum M (eq. 4-11)
    auto counting_iter = thrust::make_counting_iterator(0);
    auto key_iter
        = thrust::make_transform_iterator(counting_iter, TriangularCyclicLambda(_dataset.N));

    auto arraySize = _numEntities * TRI_SIZE(_dataset.N);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.G,
        thrust::make_discard_iterator(), _dataset.G_temp);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.H,
        thrust::make_discard_iterator(), _dataset.H_temp);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.J,
        thrust::make_discard_iterator(), _dataset.J_temp);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.K,
        thrust::make_discard_iterator(), _dataset.K_temp);
}

void GravitySource::SumN()
{
    // Prepare N summation
    int threadsPerBlock = 256;
    int numOfThreads = _numEntities;
    int blockCount = numOfThreads / threadsPerBlock + 1;
    PrepareNSumKernel<<<blockCount, threadsPerBlock>>>(
        _devDataset, _devPositionInITRS, _numEntities);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Sum N (eq. 4-12)
    auto counting_iter = thrust::make_counting_iterator(0);
    auto key_iter = thrust::make_transform_iterator(counting_iter, SegmentLambda(_dataset.N));
    auto arraySize = _numEntities * _dataset.N;
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.G_temp,
        thrust::make_discard_iterator(), _dataset.G);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.H_temp,
        thrust::make_discard_iterator(), _dataset.H);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.J_temp,
        thrust::make_discard_iterator(), _dataset.J);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize, _dataset.K_temp,
        thrust::make_discard_iterator(), _dataset.K);
}