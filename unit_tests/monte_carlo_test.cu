#include "monte_carlo/utility.cuh"
#include <gtest/gtest.h>

void FreePartitionParameters(PartitionParameters& params)
{
    delete[] params.setIds;
    delete[] params.setOffsets;
    delete[] params.numHigherIds;
    delete[] params.mapOffsets;
}

TEST(PartitionTest, ParameterCalculation)
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

TEST(PartitionTest, ParameterCalculationInvalid)
{
    PartitionParameters params;
    std::vector<unsigned> setIds1 { 0, 0, 0, 1, 1, 2, 2, 2, 3, 0 };
    EXPECT_FALSE(CalculatePartitionParameters(setIds1, params));

    std::vector<unsigned> setIds2 { 0, 0, 0, 1, 1, 2, 2, 2, 3, 5 };
    EXPECT_FALSE(CalculatePartitionParameters(setIds2, params));

    std::vector<unsigned> setIds3 { 0, 0, 0, 1, 1, 3, 2, 2, 3, 3 };
    EXPECT_FALSE(CalculatePartitionParameters(setIds3, params));
}

TEST(PartitionTest, ValidPairs)
{
    std::vector<unsigned> setIds { 0, 0, 0, 1, 1, 1, 2 };
    PartitionParameters params;
    EXPECT_TRUE(CalculatePartitionParameters(setIds, params));

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

TEST(PartitionTest, InvalidPairs)
{
    std::vector<unsigned> setIds { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3 };
    PartitionParameters params;
    EXPECT_TRUE(CalculatePartitionParameters(setIds, params));

    EXPECT_EQ(GetPartitionMapIdx(params, 0, 1), -1);
    EXPECT_EQ(GetPartitionMapIdx(params, 8, 9), -1);

    unsigned idxA, idxB;
    EXPECT_FALSE(GetPartitionMapPair(params, 37, idxA, idxB));

    FreePartitionParameters(params);
}