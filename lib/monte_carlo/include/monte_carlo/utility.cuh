#pragma once

#include "monte_carlo/types.cuh"
#include <cstring>
#include <vector>

inline __host__ __device__ int GetPartitionMapIdx(
    PartitionParameters params, unsigned idxA, unsigned idxB)
{
    if (idxA > idxB) {
        auto tmp = idxA;
        idxA = idxB;
        idxB = tmp;
    }

    auto setA = params.setIds[idxA];
    if (setA == params.setNum - 1)
        return -1;

    int offset1 = (int)idxB - (int)params.setOffsets[setA + 1];
    if (offset1 < 0)
        return -1;

    auto offset2 = params.mapOffsets[setA];
    auto offset3 = (idxA - params.setOffsets[setA]) * params.numHigherIds[setA];
    return offset1 + offset2 + offset3;
}

inline __host__ __device__ bool GetPartitionMapPair(
    PartitionParameters params, unsigned idx, unsigned& idxA, unsigned& idxB)
{
    // Find the A set (it cannot be the last one)
    unsigned setA = 0;
    while (setA < params.setNum - 1 && params.mapOffsets[setA + 1] <= idx)
        setA++;
    if (setA >= params.setNum - 1) {
        return false;
    }

    auto mapOffset = params.mapOffsets[setA];
    auto numHigherIds = params.numHigherIds[setA];

    idxA = (idx - mapOffset) / numHigherIds;
    idxA += params.setOffsets[setA];

    idxB = (idx - mapOffset) % numHigherIds;
    idxB += params.setOffsets[setA + 1];
    return true;
}

// Calculates the PartitionParameters structure and fills it, apart from the setIds array.
inline __host__ bool CalculatePartitionParameters(
    std::vector<unsigned> setIds, PartitionParameters& params)
{
    std::vector<unsigned> setOffsets;
    std::vector<unsigned> sizes;
    std::vector<unsigned> numHigherIds;
    std::vector<unsigned> mapOffsets;

    setOffsets.push_back(0);
    mapOffsets.push_back(0);

    unsigned prevId = setIds[0];
    unsigned prevChangeIdx = 0;

    for (unsigned i = 1; i < setIds.size(); i++) {
        if (setIds[i] < prevId || setIds[i] > prevId + 1) {
            return false;
        }
        if (setIds[i] > prevId) {
            setOffsets.push_back(i);
            sizes.push_back(i - prevChangeIdx);
            numHigherIds.push_back(setIds.size() - i);
            mapOffsets.push_back(mapOffsets.back() + sizes.back() * numHigherIds.back());
            prevChangeIdx = i;
        }
        prevId = setIds[i];
    }

    sizes.push_back(setIds.size() - prevChangeIdx);
    numHigherIds.push_back(0);

    params.setNum = static_cast<unsigned>(setOffsets.size());
    params.pairNum = mapOffsets.back();
    params.setOffsets = new unsigned[params.setNum];
    params.numHigherIds = new unsigned[params.setNum];
    params.mapOffsets = new unsigned[params.setNum];
    std::memcpy(params.setOffsets, setOffsets.data(), params.setNum * sizeof(unsigned));
    std::memcpy(params.numHigherIds, numHigherIds.data(), params.setNum * sizeof(unsigned));
    std::memcpy(params.mapOffsets, mapOffsets.data(), params.setNum * sizeof(unsigned));
    return true;
}

// A helper struct for calculating the Hermite interpolation polynomial coefficients and evaluating
// its value at a given point. The polynomial is defined by the positions, velocities and
// accelerations at three time points: 0, dt and 2*dt.
//
// Based on:
// Kincaid, D. R., & Cheney, E. W. (2009). Numerical analysis: mathematics of scientific
// computing (Vol. 2). American Mathematical Soc.. Chapter 6.3
// https://books.google.pl/books?hl=pl&lr=&id=kPDtAp3UZtIC
struct HermitePolynomial {
    double coeffs[9];
    double dt;

    // Calculates the 9 Newton coefficients of a Hermite interpolation polynomial
    __host__ __device__ HermitePolynomial(double pos0, double vel0, double acc0, double pos1,
        double vel1, double acc1, double pos2, double vel2, double acc2, double dt)
        : dt(dt)
    {
        double t[9] = { 0.0, 0.0, 0.0, dt, dt, dt, 2.0 * dt, 2.0 * dt, 2.0 * dt };

        // Initialize the coeffs array with positions
        coeffs[0] = pos0;
        coeffs[1] = pos0;
        coeffs[2] = pos0;
        coeffs[3] = pos1;
        coeffs[4] = pos1;
        coeffs[5] = pos1;
        coeffs[6] = pos2;
        coeffs[7] = pos2;
        coeffs[8] = pos2;

        for (int j = 1; j < 9; j++) {
            for (int i = 8; i >= j; i--) {
                if (t[i] == t[i - j]) {
                    // Inject velocities and accelerations directly
                    if (j == 1) {
                        coeffs[i] = (i < 3) ? vel0 : (i < 6) ? vel1 : vel2;
                    } else if (j == 2) {
                        coeffs[i] = ((i < 3) ? acc0 : (i < 6) ? acc1 : acc2) / 2.0; // k = 2
                    }
                } else {
                    // Standard divided difference formula
                    coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (t[i] - t[i - j]);
                }
            }
        }
    }

    inline __host__ __device__ double operator()(double t)
    {
        double z[9] = { 0.0, 0.0, 0.0, dt, dt, dt, 2.0 * dt, 2.0 * dt, 2.0 * dt };

        double result = coeffs[0];
        double term = 1.0;
        for (int i = 1; i < 9; ++i) {
            term *= (t - z[i - 1]);
            result += coeffs[i] * term;
        }

        return result;
    }

    // Evaluates the derivative of the polynomial (Velocity)
    inline __host__ __device__ double deriv(double t_eval) const
    {
        double z[9] = { 0.0, 0.0, 0.0, dt, dt, dt, 2.0 * dt, 2.0 * dt, 2.0 * dt };

        double result = 0.0;
        double N = 1.0;
        double dN = 0.0;
        for (int i = 0; i < 9; ++i) {
            result += coeffs[i] * dN;
            if (i < 8) {
                dN = dN * (t_eval - z[i]) + N;
                N *= (t_eval - z[i]);
            }
        }

        return result;
    }
};
