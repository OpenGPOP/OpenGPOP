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
    DatasetLoaders::LoadGravityModel(_dsFiles.gravityModel, _dataset.structure);
    auto& d = _dataset.structure;
    size_t size = TRI_SIZE(d.N);

    // C and S coefficients
    d.Cnm.moveToDev();
    d.Snm.moveToDev();

    // Time-variable flag
    d.is_gfct.moveToDev();

    // Reference epoch
    d.t0_mjd.moveToDev();

    // Secular trends
    d.trnd_C.moveToDev();
    d.trnd_S.moveToDev();

    // Annual periodic coefficients (1.0 yr)
    d.acos1_C.moveToDev();
    d.asin1_C.moveToDev();
    d.acos1_S.moveToDev();
    d.asin1_S.moveToDev();

    // Semi-annual periodic coefficients (0.5 yr)
    d.acos05_C.moveToDev();
    d.asin05_C.moveToDev();
    d.acos05_S.moveToDev();
    d.asin05_S.moveToDev();

    // Calculate the delta function and the diagonal of the normalized derived Legendre functions.
    d.delta.allocHost(d.N);
    std::vector<double> P_bar(size);
    memset(P_bar.data(), 0xEE, sizeof(double) * size);
    P_bar[TRI_C2IDX(0, 0)] = 1;
    P_bar[TRI_C2IDX(1, 1)] = sqrt(3);
    // delta[0] stays invalid but it will never be used
    d.delta[1] = sqrt(3);
    for (int i = 2; i < d.N; i++) {
        auto previous = P_bar[TRI_C2IDX(i - 1, i - 1)];
        double tmp = 2.0 * (double)i + 1.0;
        P_bar[TRI_C2IDX(i, i)] = previous * sqrt(tmp / (2.0 * (double)i)); // eq. 7-8
        d.delta[i] = previous * sqrt(tmp); // eq. 7-15
    }
    d.delta.moveToDev();
    d.P_bar.allocDev(size * _numEntities);
    d.P_bar.set(P_bar.data(), P_bar.size());

    // Allocate and compute the rest of auxiliary coefficients
    d.xi.allocDev(size);
    d.eta.allocDev(size);
    d.zeta.allocDev(size);

    // Allocate temporary vectors and arrays used for computations
    d.Cm_bar.allocDev(_numEntities * d.N);
    d.Sm_bar.allocDev(_numEntities * d.N);

    size = _numEntities * TRI_SIZE(d.N);
    d.G.allocDev(size);
    d.K.allocDev(size);
    d.H.allocDev(size);
    d.J.allocDev(size);

    size = _numEntities * d.N;
    d.G_temp.allocDev(size);
    d.K_temp.allocDev(size);
    d.H_temp.allocDev(size);
    d.J_temp.allocDev(size);

    _dataset.copyToDev();

    _positionInITRS.allocDev(_numEntities * 3);
    _accelerationInITRS.allocDev(_numEntities * 3);

    int threadsPerBlock = 256;
    int blockCount = TRI_SIZE(d.N) / threadsPerBlock + 1;
    ComputeAuxiliaryCoefficientsKernel<<<blockCount, threadsPerBlock>>>(_dataset.ptr);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Make copies of Pbar diagonal for all entities, for simplicity
    blockCount = (_numEntities - 1) / threadsPerBlock + 1;
    PopulatePbarDiagonalKernel<<<blockCount, threadsPerBlock>>>(_dataset.ptr, _numEntities);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
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
            devPosition, 1, 3, &beta, _positionInITRS.get(), 1, 3, _numEntities));

    // Compute the table of normalized derived Legendre polynomials (Pbar)
    int threadsPerBlock = 256;
    int numOfThreads = _numEntities * _dataset.structure.N;
    int blockCount = numOfThreads / threadsPerBlock + 1;
    ComputePbarKernel<<<blockCount, threadsPerBlock>>>(
        _dataset.ptr, _positionInITRS.get(), _numEntities);

    // Compute Cm_bar and Sm_bar tables
    numOfThreads = _numEntities;
    blockCount = numOfThreads / threadsPerBlock + 1;
    ComputeCmSmKernel<<<blockCount, threadsPerBlock>>>(
        _dataset.ptr, _positionInITRS.get(), _numEntities);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    SumM(timePoint.mjd);
    SumN();

    // Calculate acceleration
    numOfThreads = _numEntities;
    blockCount = numOfThreads / threadsPerBlock + 1;
    CalculateAcceleration<<<blockCount, threadsPerBlock>>>(
        _dataset.ptr, _positionInITRS.get(), _numEntities, _accelerationInITRS.get());
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Change position frame from ITRS to GCRS save to the summation matrix
    cublasErrchk(
        cublasDgemvStridedBatched(*_handle, CUBLAS_OP_N, 3, 3, &alpha, timePoint.ITRS_to_GCRS, 3, 0,
            _accelerationInITRS.get(), 1, 3, &beta, out, 1, outStride * 3, _numEntities));
}

void GravitySource::SumM(double* devMjd)
{
    // Prepare M summation
    int threadsPerBlock = 256;
    int numOfThreads = _numEntities * TRI_SIZE(_dataset.structure.N);
    int blockCount = numOfThreads / threadsPerBlock + 1;
    PrepareMSumKernel<<<blockCount, threadsPerBlock>>>(_dataset.ptr, _numEntities, devMjd);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Sum M (eq. 4-11)
    auto counting_iter = thrust::make_counting_iterator(0);
    auto key_iter = thrust::make_transform_iterator(
        counting_iter, TriangularCyclicLambda(_dataset.structure.N));

    auto arraySize = _numEntities * TRI_SIZE(_dataset.structure.N);
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.G.get(), thrust::make_discard_iterator(),
        _dataset.structure.G_temp.get());
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.H.get(), thrust::make_discard_iterator(),
        _dataset.structure.H_temp.get());
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.J.get(), thrust::make_discard_iterator(),
        _dataset.structure.J_temp.get());
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.K.get(), thrust::make_discard_iterator(),
        _dataset.structure.K_temp.get());
}

void GravitySource::SumN()
{
    // Prepare N summation
    int threadsPerBlock = 256;
    int numOfThreads = _numEntities;
    int blockCount = numOfThreads / threadsPerBlock + 1;
    PrepareNSumKernel<<<blockCount, threadsPerBlock>>>(
        _dataset.ptr, _positionInITRS.get(), _numEntities);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Sum N (eq. 4-12)
    auto counting_iter = thrust::make_counting_iterator(0);
    auto key_iter
        = thrust::make_transform_iterator(counting_iter, SegmentLambda(_dataset.structure.N));
    auto arraySize = _numEntities * _dataset.structure.N;
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.G_temp.get(), thrust::make_discard_iterator(),
        _dataset.structure.G.get());
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.H_temp.get(), thrust::make_discard_iterator(),
        _dataset.structure.H.get());
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.J_temp.get(), thrust::make_discard_iterator(),
        _dataset.structure.J.get());
    thrust::reduce_by_key(thrust::device, key_iter, key_iter + arraySize,
        _dataset.structure.K_temp.get(), thrust::make_discard_iterator(),
        _dataset.structure.K.get());
}