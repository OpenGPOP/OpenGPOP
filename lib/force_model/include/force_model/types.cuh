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

template <typename T> struct GpopStruct {
    T structure;
    T* ptr = nullptr;

    template <typename... Args>
    GpopStruct(Args&&... args)
        : structure(std::forward<Args>(args)...)
    {
        gpuErrchk(cudaMalloc(&ptr, sizeof(T)));
        gpuErrchk(cudaMemcpy(ptr, &structure, sizeof(T), cudaMemcpyHostToDevice));
    }
    ~GpopStruct() { CUDA_SAFE_FREE(ptr); }

    void copyToDev() { gpuErrchk(cudaMemcpy(ptr, &structure, sizeof(T), cudaMemcpyHostToDevice)); }

    // Destroys the structure and constructs a new one in its place
    template <typename... Args> void emplace(Args&&... args)
    {
        structure.~T();
        new (&structure) T(std::forward<Args>(args)...);
        gpuErrchk(cudaMemcpy(ptr, &structure, sizeof(T), cudaMemcpyHostToDevice));
    }

    GpopStruct(const GpopStruct&) = delete;
    GpopStruct& operator=(const GpopStruct&) = delete;
};

template <typename T> struct GpopArray {
    GpopArray() { }
    ~GpopArray()
    {
        if (_isDev) {
            CUDA_SAFE_FREE(_ptr);
        } else {
            delete[] _ptr;
        }
    }

    __host__ void allocDev(uint N)
    {
        if (_ptr)
            return;
        gpuErrchk(cudaMalloc(&_ptr, N * sizeof(T)));
        _N = N;
        _isDev = true;
    }
    __host__ void allocHost(uint N)
    {
        if (_ptr)
            return;
        _ptr = new T[N]();
        _N = N;
    };

    __host__ void moveToDev()
    {
        if (!_ptr || _isDev)
            return;
        T* hostPtr = _ptr;
        gpuErrchk(cudaMalloc(&_ptr, _N * sizeof(T)));
        gpuErrchk(cudaMemcpy(_ptr, hostPtr, _N * sizeof(T), cudaMemcpyHostToDevice));
        delete[] hostPtr;
        _isDev = true;
    }

    __host__ void set(void* hostPtr, size_t count)
    {
        if (!_ptr)
            return;
        if (_isDev) {
            gpuErrchk(cudaMemcpy(_ptr, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
        } else
            std::memcpy(_ptr, hostPtr, count * sizeof(T));
    }

    __host__ __device__ bool isDev() const { return _isDev; }
    __host__ __device__ T* get() const { return _ptr; }
    __host__ __device__ T& operator[](size_t i) const { return _ptr[i]; }
    __host__ __device__ uint size() const { return _N; }

    GpopArray(const GpopArray&) = delete;
    GpopArray& operator=(const GpopArray&) = delete;

private:
    bool _isDev = false;
    T* _ptr = nullptr;
    uint _N = 0;
};

template <typename T> struct GpopMatrixArray {
    GpopMatrixArray() { }

    __host__ void allocDev(uint matrix_count) { array.allocDev(matrix_count * dim * dim); }
    __host__ void allocHost(uint matrix_count) { array.allocHost(matrix_count * dim * dim); }

    __host__ __device__ bool isDev() { return array.isDev(); }
    __host__ __device__ T* get() { return array.get(); }

    __host__ __device__ T* at(uint i)
    {
        T* ptr = array.get();
        if (!ptr)
            return nullptr;
        return ptr + i * dim * dim;
    }

    GpopMatrixArray(const GpopMatrixArray&) = delete;
    GpopMatrixArray& operator=(const GpopMatrixArray&) = delete;

private:
    static constexpr int dim = 3;
    GpopArray<T> array;
};

struct TimePointParametersSlice {
    double* ITRS_to_GCRS;
    double* mjd;
};

struct TimePointParameters {
    // International Terrestrial Reference System (ITRS) to Geocentric Celestial Reference System
    // (GCRS) transformation matrix. The inverse transformation (GCRS to ITRS) is the transpose of
    // this matrix, since it's orthogonal.
    GpopMatrixArray<double> ITRS_to_GCRS;
    GpopArray<double> mjd;

    TimePointParameters() { }
    TimePointParameters(uint N)
    {
        ITRS_to_GCRS.allocDev(N);
        mjd.allocDev(N);
    }
};

struct TimePointTempParameters {
    // 4 matrices, which multiplied give the Q(t) matrix
    GpopMatrixArray<double> Q_R1;
    GpopMatrixArray<double> Q_R2;
    GpopMatrixArray<double> Q_R3;
    GpopMatrixArray<double> Q_R4;

    // 3 matrices, which multiplied give the W(t) matrix
    GpopMatrixArray<double> W_R1;
    GpopMatrixArray<double> W_R2;
    GpopMatrixArray<double> W_R3;

    // The R(t) matrix
    GpopMatrixArray<double> R;

    TimePointTempParameters(uint N)
    {
        Q_R1.allocDev(N);
        Q_R2.allocDev(N);
        Q_R3.allocDev(N);
        Q_R4.allocDev(N);
        W_R1.allocDev(N);
        W_R2.allocDev(N);
        W_R3.allocDev(N);
        R.allocDev(N);
    }
};

struct NutationTerm {
    double sine_coeff_uas; // A_i (lon) or B"_i (obl)
    double cosine_coeff_uas; // A"_i (lon) or B_i (obl)
    std::array<int, 14> args;
};

struct NutationTables {
    GpopArray<NutationTerm> j0;
    GpopArray<NutationTerm> j1;
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
    GpopArray<EOPRecord> records;
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
    GpopArray<double> mjd_utc; // Array of MJD_UTC values when leap seconds occurred
    GpopArray<double>
        leap_seconds; // Array of cumulative leap seconds corresponding to the MJD_UTC values
    size_t size; // Number of leap second entries

    // Finds the active leap second offset (TAI - UTC) for a given MJD
    __host__ __device__ double getLeapSeconds(double target_mjd) const
    {
        if (size == 0 || mjd_utc.get() == nullptr || leap_seconds.get() == nullptr) {
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
    GpopArray<double> Cnm;
    GpopArray<double> Snm;
    GpopArray<bool> is_gfct;
    GpopArray<double> trnd_C;
    GpopArray<double> acos1_C;
    GpopArray<double> asin1_C;
    GpopArray<double> acos05_C;
    GpopArray<double> asin05_C;
    GpopArray<double> trnd_S;
    GpopArray<double> acos1_S;
    GpopArray<double> asin1_S;
    GpopArray<double> acos05_S;
    GpopArray<double> asin05_S;
    GpopArray<double> t0_mjd;

    // Sin and Cos values used for computation (eq. 3-18)
    GpopArray<double> Cm_bar; // Linear, of size numEntities * (N - 1)
    GpopArray<double> Sm_bar; // Linear, of size numEntities * (N - 1)

    // Tables of auxiliary coefficients
    GpopArray<double> xi; // Triangular, of size TRIANGULAR_SIZE(N)
    GpopArray<double> eta; // Triangular, of size TRIANGULAR_SIZE(N)
    GpopArray<double> zeta; // Triangular, of size TRIANGULAR_SIZE(N)
    GpopArray<double> delta; // Linear, of size N

    // Values of the normalized derived Legendre functions
    GpopArray<double> P_bar; // Triangular for each entity, of size TRIANGULAR_SIZE(N) * numEntities

    // Intermediate tables for calculating the acceleration (eq. 4-11, 4-12)
    // Size TRIANGULAR_SIZE(N) * numEntities
    GpopArray<double> G;
    GpopArray<double> H;
    GpopArray<double> J;
    GpopArray<double> K;
    // Size N * numEntities
    GpopArray<double> G_temp;
    GpopArray<double> H_temp;
    GpopArray<double> J_temp;
    GpopArray<double> K_temp;
};
