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
#include "gpu_utils/gpu_utils.cuh"
#include "types/Parameters.hpp"
#include <array>
#include <cstddef>
#include <cstring>
#include <cuda_runtime_api.h>
#include <utility>

template <typename T> struct DevStruct {
    T structure;
    T* ptr = nullptr;

    template <typename... Args>
    DevStruct(Args&&... args)
        : structure(std::forward<Args>(args)...)
    {
        gpuErrchk(cudaMalloc(&ptr, sizeof(T)));
        gpuErrchk(cudaMemcpy(ptr, &structure, sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DevStruct() { CUDA_SAFE_FREE(ptr); }

    // Destroys the structure and constructs a new one in its place
    template <typename... Args> void emplace(Args&&... args)
    {
        structure.~T();
        new (&structure) T(std::forward<Args>(args)...);
        gpuErrchk(cudaMemcpy(ptr, &structure, sizeof(T), cudaMemcpyHostToDevice));
    }

    DevStruct(const DevStruct&) = delete;
    DevStruct& operator=(const DevStruct&) = delete;
};

template <typename T> struct DevArray {
    DevArray() { }
    DevArray(uint N) { alloc(N); }
    ~DevArray() { CUDA_SAFE_FREE(ptr); }

    void alloc(uint N) { gpuErrchk(cudaMalloc(&ptr, N * sizeof(T))); }
    __host__ __device__ T* data() const { return ptr; }
    __device__ T& operator[](size_t i) const { return ptr[i]; }

    DevArray(const DevArray&) = delete;
    DevArray& operator=(const DevArray&) = delete;

private:
    T* ptr = nullptr;
};

template <typename T> struct DevMatrixArray {
    DevMatrixArray() { }
    DevMatrixArray(uint matrix_count)
        : array(matrix_count * dim * dim)
    {
    }

    void alloc(uint matrix_count) { array.alloc(matrix_count * dim * dim); }
    __host__ __device__ T* data() const { return array.data(); }
    __device__ T& operator[](size_t i) const { return array.data()[i * dim * dim]; }

    DevMatrixArray(const DevMatrixArray&) = delete;
    DevMatrixArray& operator=(const DevMatrixArray&) = delete;

private:
    static constexpr int dim = 3;
    DevArray<T> array;
};

struct TimePointParametersSlice {
    double* ITRS_to_GCRS;
    double* mjd;
};

struct TimePointParameters {
    // International Terrestrial Reference System (ITRS) to Geocentric Celestial Reference System
    // (GCRS) transformation matrix. The inverse transformation (GCRS to ITRS) is the transpose of
    // this matrix, since it's orthogonal.
    DevMatrixArray<double> ITRS_to_GCRS;
    DevArray<double> mjd;

    TimePointParameters() { }
    TimePointParameters(uint N)
        : ITRS_to_GCRS(N)
        , mjd(N)
    {
    }
};

struct TimePointTempParameters {
    // 4 matrices, which multiplied give the Q(t) matrix
    DevMatrixArray<double> Q_R1;
    DevMatrixArray<double> Q_R2;
    DevMatrixArray<double> Q_R3;
    DevMatrixArray<double> Q_R4;

    // 3 matrices, which multiplied give the W(t) matrix
    DevMatrixArray<double> W_R1;
    DevMatrixArray<double> W_R2;
    DevMatrixArray<double> W_R3;

    // The R(t) matrix
    DevMatrixArray<double> R;

    TimePointTempParameters(uint N)
        : Q_R1(N)
        , Q_R2(N)
        , Q_R3(N)
        , Q_R4(N)
        , W_R1(N)
        , W_R2(N)
        , W_R3(N)
        , R(N)
    {
    }
};

struct NutationTerm {
    double sine_coeff_uas; // A_i (lon) or B"_i (obl)
    double cosine_coeff_uas; // A"_i (lon) or B_i (obl)
    std::array<int, 14> args;
};

struct NutationTables {
    NutationTerm* j0 = nullptr;
    NutationTerm* j1 = nullptr;
    size_t j0_size;
    size_t j1_size;
};

struct NutationDataset {
    NutationTables lon;
    NutationTables obl;
};

// Struct representing a single extracted row
struct EOPRecord {
    double mjd_utc;
    double xp;
    double yp;
    double ut1_utc;
};

struct EOPDataset {
    EOPRecord* records = nullptr;
    size_t size = 0;

    // Find records by MJD_UTC using binary search and perform linear interpolation.
    // Clamps to the first or last value if out of bounds.
    __host__ __device__ EOPRecord getRecord(double target_mjd) const
    {
        // Safety check for empty dataset
        if (size == 0) {
            return EOPRecord { -1.0, 0.0, 0.0, 0.0 };
        }

        // --- Out-of-bounds checks (Clamp to first/last) ---
        if (target_mjd <= records[0].mjd_utc) {
            return records[0];
        }
        if (target_mjd >= records[size - 1].mjd_utc) {
            return records[size - 1];
        }

        // --- Binary Search ---
        int left = 0;
        int right = (int)size - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            double current_mjd = records[mid].mjd_utc;

            if (current_mjd == target_mjd) {
                return records[mid]; // Exact match found
            } else if (current_mjd < target_mjd) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        // --- Linear Interpolation ---
        // At this point, exact match wasn't found, and target_mjd is strictly between
        // records[right] and records[left].
        const EOPRecord& r0 = records[right];
        const EOPRecord& r1 = records[left];

        double t0 = r0.mjd_utc;
        double t1 = r1.mjd_utc;

        // Calculate the interpolation factor [0.0, 1.0]
        double fraction = (target_mjd - t0) / (t1 - t0);

        EOPRecord result;
        result.mjd_utc = target_mjd;
        result.xp = r0.xp + fraction * (r1.xp - r0.xp);
        result.yp = r0.yp + fraction * (r1.yp - r0.yp);
        result.ut1_utc = r0.ut1_utc + fraction * (r1.ut1_utc - r0.ut1_utc);

        return result;
    }
};

struct LeapSecondDataset {
    double* mjd_utc = nullptr; // Array of MJD_UTC values when leap seconds occurred
    double* leap_seconds
        = nullptr; // Array of cumulative leap seconds corresponding to the MJD_UTC values
    size_t size; // Number of leap second entries

    // Finds the active leap second offset (TAI - UTC) for a given MJD
    __host__ __device__ double getLeapSeconds(double target_mjd) const
    {
        if (size == 0 || mjd_utc == nullptr || leap_seconds == nullptr) {
            return 0.0;
        }

        for (size_t i = size; i-- > 0;) {
            if (target_mjd >= mjd_utc[i]) {
                return leap_seconds[i];
            }
        }
        return leap_seconds[0];
    }
};

struct TimePointDatasets {
    NutationDataset nutDataset;
    EOPDataset eopDataset;
    LeapSecondDataset lsDataset;
};

struct GravityDataset {
    unsigned N; // Max degree + 1 (because indexed from 0)
    double GM; // Earth gravity constant
    double R; // Earth radius

    // Triangular tables of harmonic coefficients (of size TRIANGULAR_SIZE(N))
    double* Cnm = nullptr;
    double* Snm = nullptr;
    bool* is_gfct = nullptr;
    double* trnd_C = nullptr;
    double* acos1_C = nullptr;
    double* asin1_C = nullptr;
    double* acos05_C = nullptr;
    double* asin05_C = nullptr;
    double* trnd_S = nullptr;
    double* acos1_S = nullptr;
    double* asin1_S = nullptr;
    double* acos05_S = nullptr;
    double* asin05_S = nullptr;
    double* t0_mjd = nullptr;

    // Sin and Cos values used for computation (eq. 3-18)
    double* Cm_bar = nullptr; // Linear, of size numEntities * (N - 1)
    double* Sm_bar = nullptr; // Linear, of size numEntities * (N - 1)

    // Tables of auxiliary coefficients
    double* xi = nullptr; // Triangular, of size TRIANGULAR_SIZE(N)
    double* eta = nullptr; // Triangular, of size TRIANGULAR_SIZE(N)
    double* zeta = nullptr; // Triangular, of size TRIANGULAR_SIZE(N)
    double* delta = nullptr; // Linear, of size N

    // Values of the normalized derived Legendre functions
    double* P_bar = nullptr; // Triangular for each entity, of size TRIANGULAR_SIZE(N) * numEntities

    // Intermediate tables for calculating the acceleration (eq. 4-11, 4-12)
    // Size TRIANGULAR_SIZE(N) * numEntities
    double* G = nullptr;
    double* H = nullptr;
    double* J = nullptr;
    double* K = nullptr;
    // Size N * numEntities
    double* G_temp = nullptr;
    double* H_temp = nullptr;
    double* J_temp = nullptr;
    double* K_temp = nullptr;
};
