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
#include "force_model/constants.cuh"
#include "gpu_utils/gpu_utils.cuh"

__global__ void ComputeAuxiliaryCoefficientsKernel(GravityDataset* dataset)
{
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= TRI_SIZE(dataset->N))
        return;

    auto [i, j] = TRI_IDX2C(tIdx);
    if (i == j)
        return;
    double n = i, m = j;

    double* xi = dataset->xi.get();
    double* eta = dataset->eta.get();
    double* zeta = dataset->zeta.get();

    // eq. 7-10
    xi[tIdx] = sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)));
    eta[tIdx] = sqrt(
        (n + m - 1.0) * (2.0 * n + 1.0) * (n - m - 1.0) / ((n + m) * (n - m) * (2.0 * n - 3.0)));

    // eq. 7-19
    if (m == 0) {
        zeta[tIdx] = sqrt(n * (n + 1.0) / 2.0);
    } else {
        zeta[tIdx] = sqrt((n - m) * (n + m + 1.0));
    }
}

// TODO: Optimize memory accesses
__global__ void PopulatePbarDiagonalKernel(GravityDataset* dataset, int numEntities)
{
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= numEntities - 1)
        return;

    auto N = dataset->N;
    auto P_bar = dataset->P_bar.get() + (tIdx + 1) * TRI_SIZE(N);
    for (int i = 0; i < N; i++) {
        auto index = TRI_C2IDX(i, i);
        P_bar[index] = dataset->P_bar[index];
    }
}

__global__ void ComputePbarKernel(GravityDataset* dataset, double* positions, int numEntities)
{
    auto tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    auto N = dataset->N;
    if (tIdx >= numEntities * N)
        return;

    auto entity = tIdx / N;
    auto m = tIdx % N;
    if (m == N - 1) // Last column has nothing to calculate
        return;

    auto pos = positions + entity * 3;
    double x = pos[0], y = pos[1], z = pos[2];
    double r = sqrt(x * x + y * y + z * z);
    double eps = z / r; // eq. 3-3

    auto P_bar = dataset->P_bar.get() + TRI_SIZE(N) * entity;
    auto xi = dataset->xi.get();
    auto eta = dataset->eta.get();
    auto n = m + 1;

    // Calculate the term below the diagonal
    P_bar[TRI_C2IDX(n, n - 1)] = eps * dataset->delta[n]; // eq. 7-16

    // Calculate the other terms
    while (n != N - 1) {
        ++n;
        auto n_m = TRI_C2IDX(n, m);
        P_bar[n_m] = xi[n_m] * eps * P_bar[TRI_C2IDX(n - 1, m)]
            - eta[n_m] * P_bar[TRI_C2IDX(n - 2, m)]; // eq. 7-12
    }
}

__global__ void ComputeCmSmKernel(GravityDataset* dataset, double* positions, int numEntities)
{
    auto tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= numEntities)
        return;

    auto pos = positions + tIdx * 3;
    double x = pos[0], y = pos[1], z = pos[2];
    double r = sqrt(x * x + y * y + z * z);

    // Compute Cm_bar and Sm_bar recursively to avoid singularity at pole (eq. 3-18)
    // TODO: Optimize to avoid global memory accesses every time
    auto N = dataset->N;
    double* Cm = dataset->Cm_bar.get() + tIdx * N;
    double* Sm = dataset->Sm_bar.get() + tIdx * N;
    Cm[0] = 1;
    Sm[0] = 0;
    Cm[1] = x / r;
    Sm[1] = y / r;
    for (int i = 2; i < N; i++) {
        Cm[i] = Cm[1] * Cm[i - 1] - Sm[1] * Sm[i - 1];
        Sm[i] = Sm[1] * Cm[i - 1] + Cm[1] * Sm[i - 1];
    }
}

__device__ double GetCSCoefficient(GravityDataset* dataset, bool C, int index, double* mjd)
{
    double val = C ? dataset->Cnm[index] : dataset->Snm[index];

    if (!dataset->is_gfct[index])
        return val;

    double dt = (*mjd - dataset->t0_mjd[index]) / 365.25;
    double cos1 = cos(2.0 * CFMConst::PI * dt);
    double sin1 = sin(2.0 * CFMConst::PI * dt);
    double cos05 = cos(4.0 * CFMConst::PI * dt);
    double sin05 = sin(4.0 * CFMConst::PI * dt);

    if (C) {
        val += dataset->trnd_C[index] * dt;
        val += dataset->acos1_C[index] * cos1 + dataset->asin1_C[index] * sin1;
        val += dataset->acos05_C[index] * cos05 + dataset->asin05_C[index] * sin05;
    } else {
        val += dataset->trnd_S[index] * dt;
        val += dataset->acos1_S[index] * cos1 + dataset->asin1_S[index] * sin1;
        val += dataset->acos05_S[index] * cos05 + dataset->asin05_S[index] * sin05;
    }

    return val;
}

__global__ void PrepareMSumKernel(GravityDataset* dataset, int numEntities, double* mjd)
{
    auto tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    auto N = dataset->N;
    auto size = TRI_SIZE(N);
    if (tIdx >= numEntities * size)
        return;

    auto entity = tIdx / size;
    auto nm_index = tIdx % size;
    auto [n, m] = TRI_IDX2C(nm_index);
    auto n0_index = TRI_C2IDX(n, 0);

    auto P_bar = dataset->P_bar.get() + TRI_SIZE(N) * entity;
    double Pnm = P_bar[nm_index];
    double Pnm1;
    if (n > m)
        Pnm1 = P_bar[TRI_C2IDX(n, m + 1)];
    else
        Pnm1 = 0;

    double Cnm = GetCSCoefficient(dataset, true, nm_index, mjd);
    double Snm = GetCSCoefficient(dataset, false, nm_index, mjd);

    if (n < 2) {
        dataset->J[tIdx] = 0;
        dataset->K[tIdx] = 0;
        dataset->G[tIdx] = 0;
        dataset->H[tIdx] = 0;
        return;
    }

    // eq. 4-11
    if (m == 0) {
        // sums start from m = 1 but G and H have additional elements outside of the sums
        // for simplicity's sake, we made all these arrays the same size
        double Pn0 = P_bar[n0_index];
        double Pn1 = P_bar[TRI_C2IDX(n, 1)];
        double Cn0 = GetCSCoefficient(dataset, true, n0_index, mjd);
        dataset->J[tIdx] = 0;
        dataset->K[tIdx] = 0;
        dataset->G[tIdx] = Cn0 * (n + 1) * Pn0;
        dataset->H[tIdx] = dataset->zeta[n0_index] * Cn0 * Pn1; // zeta from eq. 7-17
        return;
    }

    auto Cm_arr = dataset->Cm_bar.get() + entity * N;
    auto Sm_arr = dataset->Sm_bar.get() + entity * N;
    double Cm = Cm_arr[m];
    double Sm = Sm_arr[m];
    double Cm1 = Cm_arr[m - 1];
    double Sm1 = Sm_arr[m - 1];

    // eq. 4-11
    // division by r^m already accounted for in Cm and Sm (eq. 3-18)
    dataset->J[tIdx] = (double)m * Pnm * (Cnm * Cm1 + Snm * Sm1);
    dataset->K[tIdx] = -(double)m * Pnm * (Cnm * Sm1 - Snm * Cm1);
    dataset->G[tIdx] = (double)(1 + n + m) * Pnm * (Cnm * Cm + Snm * Sm);
    dataset->H[tIdx] = dataset->zeta[nm_index] * Pnm1 * (Cnm * Cm + Snm * Sm); // zeta from eq. 7-17
}

__global__ void PrepareNSumKernel(GravityDataset* dataset, double* positions, int numEntities)
{
    auto tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= numEntities)
        return;

    auto N = dataset->N;
    auto offset = tIdx * N;
    auto G_temp = dataset->G_temp.get() + offset;
    auto H_temp = dataset->H_temp.get() + offset;
    auto K_temp = dataset->K_temp.get() + offset;
    auto J_temp = dataset->J_temp.get() + offset;

    auto pos = positions + tIdx * 3;
    double x = pos[0], y = pos[1], z = pos[2];
    double r = sqrt(x * x + y * y + z * z);

    // eq. 4-12
    double base = dataset->R / r;
    double coeff = base * base;
    for (int n = 2; n < N; n++) {
        G_temp[n] *= coeff;
        H_temp[n] *= coeff;
        K_temp[n] *= coeff;
        J_temp[n] *= coeff;
        coeff *= base;
    }
}

__global__ void CalculateAcceleration(
    GravityDataset* dataset, double* positions, int numEntities, double* outAccelerations)
{
    auto tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tIdx >= numEntities)
        return;

    double G = dataset->G[tIdx] + 1; // +1 from eq. 4-12
    double H = dataset->H[tIdx];
    double J = dataset->J[tIdx];
    double K = dataset->K[tIdx];

    auto pos = positions + tIdx * 3;
    double x = pos[0], y = pos[1], z = pos[2];
    double r = sqrt(x * x + y * y + z * z);

    // eq. 4-13
    double Lambda = G + z * H / r;
    double ax = Lambda * x / r - J;
    double ay = Lambda * y / r - K;
    double az = Lambda * z / r - H;

    double coeff = -dataset->GM / (r * r);
    auto g = outAccelerations + tIdx * 3;
    g[0] = coeff * ax;
    g[1] = coeff * ay;
    g[2] = coeff * az;
}