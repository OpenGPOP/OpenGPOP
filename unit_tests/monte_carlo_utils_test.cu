#include "monte_carlo/utility.cuh"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

void FreePartitionParameters(PartitionParameters& params)
{
    delete[] params.setOffsets;
    delete[] params.numHigherIds;
    delete[] params.mapOffsets;
}

TEST(MonteCarloUtilsTest, PartitionTest_ParameterCalculation)
{
    std::vector<unsigned> setIds { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3 };
    PartitionParameters params;
    EXPECT_TRUE(CalculatePartitionParameters(setIds, params));

    ASSERT_EQ(params.setNum, 4u);
    unsigned expected_setOffsets[] = { 0, 3, 5, 8 };
    unsigned expected_numHigherIds[] = { 7, 5, 2, 0 };
    unsigned expected_mapOffsets[] = { 0, 21, 31, 37 };

    for (unsigned i = 0; i < params.setNum; ++i) {
        EXPECT_EQ(params.setOffsets[i], expected_setOffsets[i]);
        EXPECT_EQ(params.numHigherIds[i], expected_numHigherIds[i]);
        EXPECT_EQ(params.mapOffsets[i], expected_mapOffsets[i]);
    }

    FreePartitionParameters(params);
}

TEST(MonteCarloUtilsTest, PartitionTest_ParameterCalculationInvalid)
{
    PartitionParameters params;
    std::vector<unsigned> setIds1 { 0, 0, 0, 1, 1, 2, 2, 2, 3, 0 };
    EXPECT_FALSE(CalculatePartitionParameters(setIds1, params));

    std::vector<unsigned> setIds2 { 0, 0, 0, 1, 1, 2, 2, 2, 3, 5 };
    EXPECT_FALSE(CalculatePartitionParameters(setIds2, params));

    std::vector<unsigned> setIds3 { 0, 0, 0, 1, 1, 3, 2, 2, 3, 3 };
    EXPECT_FALSE(CalculatePartitionParameters(setIds3, params));
}

TEST(MonteCarloUtilsTest, PartitionTest_ValidPairs)
{
    std::vector<unsigned> setIds { 0, 0, 0, 1, 1, 1, 2 };
    PartitionParameters params;
    EXPECT_TRUE(CalculatePartitionParameters(setIds, params));
    params.setIds = setIds.data();

    EXPECT_EQ(GetPartitionMapIdx(params, 0, 3), 0);
    EXPECT_EQ(GetPartitionMapIdx(params, 1, 6), 7);
    EXPECT_EQ(GetPartitionMapIdx(params, 1, 3), 4);
    EXPECT_EQ(GetPartitionMapIdx(params, 3, 6), 12);
    EXPECT_EQ(GetPartitionMapIdx(params, 6, 5), 14); // Auto-swaps to (5, 6)

    unsigned idxA, idxB;
    EXPECT_TRUE(GetPartitionMapPair(params, 0, idxA, idxB));
    EXPECT_EQ(idxA, 0u);
    EXPECT_EQ(idxB, 3u);

    EXPECT_TRUE(GetPartitionMapPair(params, 7, idxA, idxB));
    EXPECT_EQ(idxA, 1u);
    EXPECT_EQ(idxB, 6u);

    EXPECT_TRUE(GetPartitionMapPair(params, 4, idxA, idxB));
    EXPECT_EQ(idxA, 1u);
    EXPECT_EQ(idxB, 3u);

    EXPECT_TRUE(GetPartitionMapPair(params, 12, idxA, idxB));
    EXPECT_EQ(idxA, 3u);
    EXPECT_EQ(idxB, 6u);

    EXPECT_TRUE(GetPartitionMapPair(params, 14, idxA, idxB));
    EXPECT_EQ(idxA, 5u);
    EXPECT_EQ(idxB, 6u);

    FreePartitionParameters(params);
}

TEST(MonteCarloUtilsTest, PartitionTest_OneSetNoPairs)
{
    std::vector<unsigned> setIds { 0, 0, 0, 0 };
    PartitionParameters params;
    EXPECT_TRUE(CalculatePartitionParameters(setIds, params));
    params.setIds = setIds.data();

    EXPECT_EQ(params.setNum, 1u);
    EXPECT_EQ(params.pairNum, 0u);
    EXPECT_EQ(params.setOffsets[0], 0u);
    EXPECT_EQ(params.numHigherIds[0], 0u);
    EXPECT_EQ(params.mapOffsets[0], 0u);

    FreePartitionParameters(params);
}

TEST(MonteCarloUtilsTest, PartitionTest_InvalidPairs)
{
    std::vector<unsigned> setIds { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3 };
    PartitionParameters params;
    EXPECT_TRUE(CalculatePartitionParameters(setIds, params));
    params.setIds = setIds.data();

    EXPECT_EQ(GetPartitionMapIdx(params, 0, 1), -1);
    EXPECT_EQ(GetPartitionMapIdx(params, 8, 9), -1);

    unsigned idxA, idxB;
    EXPECT_FALSE(GetPartitionMapPair(params, 37, idxA, idxB));

    FreePartitionParameters(params);
}

TEST(MonteCarloUtilsTest, HermitePolynomial_ConstantPosition)
{
    double dt = 1.0;
    HermitePolynomial poly(5.0, 0.0, 0.0, // t = 0
        5.0, 0.0, 0.0, // t = 1
        5.0, 0.0, 0.0, // t = 2
        dt);

    EXPECT_DOUBLE_EQ(poly.coeffs[0], 5.0);
    for (int i = 1; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(poly.coeffs[i], 0.0);
    }

    EXPECT_DOUBLE_EQ(poly(0.0), 5.0);
    EXPECT_DOUBLE_EQ(poly(0.5), 5.0);
    EXPECT_DOUBLE_EQ(poly(1.0), 5.0);
    EXPECT_DOUBLE_EQ(poly(1.5), 5.0);
    EXPECT_DOUBLE_EQ(poly(2.0), 5.0);

    EXPECT_DOUBLE_EQ(poly.deriv(0.0), 0.0);
    EXPECT_DOUBLE_EQ(poly.deriv(0.5), 0.0);
    EXPECT_DOUBLE_EQ(poly.deriv(1.0), 0.0);
    EXPECT_DOUBLE_EQ(poly.deriv(1.5), 0.0);
    EXPECT_DOUBLE_EQ(poly.deriv(2.0), 0.0);
}

TEST(MonteCarloUtilsTest, HermitePolynomial_LinearFunction)
{
    double dt = 1.0;
    // f(t) = 2t + 1  =>  f'(t) = 2
    HermitePolynomial poly(1.0, 2.0, 0.0, // t = 0
        3.0, 2.0, 0.0, // t = 1
        5.0, 2.0, 0.0, // t = 2
        dt);

    EXPECT_DOUBLE_EQ(poly.coeffs[0], 1.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[1], 2.0);
    for (int i = 2; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(poly.coeffs[i], 0.0);
    }

    EXPECT_DOUBLE_EQ(poly(0.0), 1.0);
    EXPECT_DOUBLE_EQ(poly(0.5), 2.0);
    EXPECT_DOUBLE_EQ(poly(1.0), 3.0);
    EXPECT_DOUBLE_EQ(poly(1.5), 4.0);
    EXPECT_DOUBLE_EQ(poly(2.0), 5.0);

    EXPECT_DOUBLE_EQ(poly.deriv(0.0), 2.0);
    EXPECT_DOUBLE_EQ(poly.deriv(0.5), 2.0);
    EXPECT_DOUBLE_EQ(poly.deriv(1.0), 2.0);
    EXPECT_DOUBLE_EQ(poly.deriv(1.5), 2.0);
    EXPECT_DOUBLE_EQ(poly.deriv(2.0), 2.0);
}

TEST(MonteCarloUtilsTest, HermitePolynomial_QuadraticFunction)
{
    double dt = 2.0;
    // f(t) = t^2  =>  f'(t) = 2t
    HermitePolynomial poly(0.0, 0.0, 2.0, // t = 0
        4.0, 4.0, 2.0, // t = 2
        16.0, 8.0, 2.0, // t = 4
        dt);

    EXPECT_DOUBLE_EQ(poly.coeffs[0], 0.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[1], 0.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[2], 1.0);
    for (int i = 3; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(poly.coeffs[i], 0.0);
    }

    EXPECT_DOUBLE_EQ(poly(1.0), 1.0);
    EXPECT_DOUBLE_EQ(poly(2.0), 4.0);
    EXPECT_DOUBLE_EQ(poly(3.0), 9.0);

    EXPECT_DOUBLE_EQ(poly.deriv(1.0), 2.0); // 2 * 1
    EXPECT_DOUBLE_EQ(poly.deriv(2.0), 4.0); // 2 * 2
    EXPECT_DOUBLE_EQ(poly.deriv(3.0), 6.0); // 2 * 3
}

TEST(MonteCarloUtilsTest, HermitePolynomial_MatchesConstraintsAtNodes)
{
    double dt = 1.0;
    // f(t) = t^3 - t => f'(t) = 3t^2 - 1
    HermitePolynomial poly(0.0, -1.0, 0.0, // t = 0
        0.0, 2.0, 6.0, // t = 1
        6.0, 11.0, 12.0, // t = 2
        dt);

    // Coefficients for f(t) = 0 - 1*t + 0*t^2 + 1*t^3
    EXPECT_DOUBLE_EQ(poly.coeffs[0], 0.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[1], -1.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[2], 0.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[3], 1.0);
    for (int i = 4; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(poly.coeffs[i], 0.0);
    }

    EXPECT_DOUBLE_EQ(poly(0.0), 0.0);
    EXPECT_DOUBLE_EQ(poly(1.0), 0.0);
    EXPECT_DOUBLE_EQ(poly(2.0), 6.0);
    EXPECT_DOUBLE_EQ(poly(0.5), -0.375);

    EXPECT_DOUBLE_EQ(poly.deriv(0.0), -1.0);
    EXPECT_DOUBLE_EQ(poly.deriv(1.0), 2.0);
    EXPECT_DOUBLE_EQ(poly.deriv(2.0), 11.0);
    EXPECT_DOUBLE_EQ(poly.deriv(0.5), -0.25);
}

TEST(MonteCarloUtilsTest, HermitePolynomial_ExtrapolationBehavesAsExpected)
{
    double dt = 1.0;
    // f(t) = t^2 => f'(t) = 2t
    HermitePolynomial poly(0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 4.0, 4.0, 2.0, dt);

    EXPECT_DOUBLE_EQ(poly.coeffs[0], 0.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[1], 0.0);
    EXPECT_DOUBLE_EQ(poly.coeffs[2], 1.0);
    for (int i = 3; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(poly.coeffs[i], 0.0);
    }

    EXPECT_DOUBLE_EQ(poly(-1.0), 1.0);
    EXPECT_DOUBLE_EQ(poly(3.0), 9.0);

    EXPECT_DOUBLE_EQ(poly.deriv(-1.0), -2.0); // 2 * (-1)
    EXPECT_DOUBLE_EQ(poly.deriv(3.0), 6.0); // 2 * 3
}

struct LinearFunc {
    __host__ __device__ double operator()(double x) const { return 3.0 * x - 6.0; }
};

struct QuadraticFunc {
    __host__ __device__ double operator()(double x) const { return x * x - 9.0; }
};

struct SquareFunc {
    __host__ __device__ double operator()(double x) const { return x * x; }
};

TEST(MonteCarloUtilsTest, FindRoot_LinearFunction)
{
    LinearFunc f;

    double root = FindRoot(f, 0.0, 1.0);
    EXPECT_NEAR(root, 2.0, 1e-7);
}

TEST(MonteCarloUtilsTest, FindRoot_QuadraticFunction)
{
    QuadraticFunc f;

    double root_pos = FindRoot(f, 1.0, 5.0);
    EXPECT_NEAR(root_pos, 3.0, 1e-7);

    double root_neg = FindRoot(f, -1.0, -5.0);
    EXPECT_NEAR(root_neg, -3.0, 1e-7);
}

TEST(MonteCarloUtilsTest, FindRoot_HorizontalSecantAvoidsDivisionByZero)
{
    SquareFunc f;

    double root = FindRoot(f, -2.0, 2.0);
    EXPECT_DOUBLE_EQ(root, 2.0);
}

TEST(MonteCarloUtilsTest, ApproachFunc_EvaluatesDotProductCorrectly)
{
    double dt = 1.0;

    // A(t) = [t, 2, 0]
    HermitePolynomial Ax(0, 1, 0, 1, 1, 0, 2, 1, 0, dt); // A_x(t) = t
    HermitePolynomial Ay(2, 0, 0, 2, 0, 0, 2, 0, 0, dt); // A_y(t) = 2
    HermitePolynomial Az(0, 0, 0, 0, 0, 0, 0, 0, 0, dt); // A_z(t) = 0
    HermitePolynomial3D objA { Ax, Ay, Az };

    // B'(t) = [1, t - 3, 0]
    // Integrating B' gives B(t) = [t, 0.5*t^2 - 3t, 0]
    HermitePolynomial Bx(0, 1, 0, 1, 1, 0, 2, 1, 0, dt);
    HermitePolynomial By(0, -3, 1, -2.5, -2, 1, -4, -1, 1, dt);
    HermitePolynomial Bz(0, 0, 0, 0, 0, 0, 0, 0, 0, dt);
    HermitePolynomial3D objB { Bx, By, Bz };

    ApproachFunc func(objA, objB);

    // f(t) = dx*dx' + dy*dy' + dz*dz'
    // dy = -0.5t^2 + 3t + 2
    // dy' = -t + 3
    // f(t) = (-0.5t^2 + 3t + 2) * (-t + 3)
    EXPECT_DOUBLE_EQ(func(0.0), 6.0); // (2)*(3) = 6
    EXPECT_DOUBLE_EQ(func(1.0), 9.0); // (4.5)*(2) = 9
    EXPECT_DOUBLE_EQ(func(2.0), 6.0); // (6)*(1) = 6
    EXPECT_DOUBLE_EQ(func(3.0), 0.0); // (6.5)*(0) = 0
    EXPECT_DOUBLE_EQ(func(4.0), -6.0); // (6)*(-1) = -6
}

TEST(MonteCarloUtilsTest, Integration_FindRootOfApproachFunc)
{
    double dt = 1.0;

    HermitePolynomial Ax(0, 1, 0, 1, 1, 0, 2, 1, 0, dt);
    HermitePolynomial Ay(2, 0, 0, 2, 0, 0, 2, 0, 0, dt);
    HermitePolynomial Az(0, 0, 0, 0, 0, 0, 0, 0, 0, dt);
    HermitePolynomial3D objA { Ax, Ay, Az };

    HermitePolynomial Bx(0, 1, 0, 1, 1, 0, 2, 1, 0, dt);
    HermitePolynomial By(0, -3, 1, -2.5, -2, 1, -4, -1, 1, dt);
    HermitePolynomial Bz(0, 0, 0, 0, 0, 0, 0, 0, 0, dt);
    HermitePolynomial3D objB { Bx, By, Bz };

    ApproachFunc func(objA, objB);

    double tol = 1e-12;
    double root = FindRoot(func, 2.0, 10.0, tol);
    EXPECT_NEAR(root, 3.0, tol);
}