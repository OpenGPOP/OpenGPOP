#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <utility>

#define gpuErrchk(ans)                                                                             \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define cublasErrchk(ans)                                                                          \
    {                                                                                              \
        cublasAssert((ans), __FILE__, __LINE__);                                                   \
    }
inline void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true)
{
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLASassert: %s %s %d\n", cublasGetStatusString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T> inline void CUDA_SAFE_FREE(T*& ptr)
{
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

// Converts (row,col) coordinates of a square matrix of order N to a column-major index in a
// flattened 1D array
constexpr size_t SQ_C2IDX(size_t row, size_t col, size_t N) { return (col * N) + row; }

// Converts (row,col) coordinates of a lower-triangular matrix (row>=col) to an index in a
// flattened 1D array
constexpr size_t TRI_C2IDX(size_t row, size_t col) { return (row * (row + 1) / 2) + col; }

// Calculates the size of a flattened 1D array representing a triangular matrix of order N
constexpr size_t TRI_SIZE(size_t N) { return (N * (N + 1) / 2); }

constexpr size_t INTEGER_SQRT(size_t n)
{
    if (n <= 1)
        return n;
    size_t x = n, y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

// Converts the index in a flattened 1D array representing a lower-triangular matrix to (row,col)
// coordinates in said matrix
constexpr std::pair<size_t, size_t> TRI_IDX2C(size_t index)
{
    size_t i = (INTEGER_SQRT(8 * index + 1) - 1) / 2;
    size_t j = index - (i * (i + 1) / 2);
    return { i, j };
}