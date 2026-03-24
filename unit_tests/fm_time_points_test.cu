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

#include "cublas_v2.h"
#include "force_model/constants.cuh"
#include "force_model/time_points/time_points.cuh" // Make sure this includes the class definition
#include "force_model/types.cuh"
#include "force_model/utility/dataset_loaders.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "sofa.h"
#include "types/Parameters.hpp"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

extern void ExpectMatricesNear(
    const double* actual_matrix, double sofa_matrix[3][3], double tolerance = 1e-12);

void GetSofaITRStoGCRS(TimeData defaultStartTime, double result[3][3])
{
    double djm0, djm;
    iauCal2jd(defaultStartTime.Year, defaultStartTime.Month, defaultStartTime.Day, &djm0, &djm);
    double time_fraction
        = (defaultStartTime.Hour + defaultStartTime.Min / 60.0 + defaultStartTime.Sec / 3600.0)
        / 24.0;
    double mjd_UTC = djm + time_fraction;

    double u1 = 2400000.5;
    double u2 = mjd_UTC;
    double tai1, tai2, tt1, tt2, ut11, ut12;

    // SOFA: UTC -> TAI -> TT
    iauUtctai(u1, u2, &tai1, &tai2);
    iauTaitt(tai1, tai2, &tt1, &tt2);

    // SOFA: UTC -> UT1
    auto eop = DatasetLoaders::LoadEOPData("../../data/IAU2000A_EOP.txt");
    auto eop_record = eop.getRecord(mjd_UTC);
    iauUtcut1(u1, u2, eop_record.ut1_utc, &ut11, &ut12);

    double sofa_Q[3][3];
    iauPnm06a(2400000.5, tt2, sofa_Q);

    double sofa_W_raw[3][3];
    double sp = iauSp00(2400000.5, tt2);
    iauPom00(eop_record.xp, eop_record.yp, sp, sofa_W_raw);

    double gst = iauGst06a(2400000.5, ut12, 2400000.5, tt2);
    double sofa_C2T_raw[3][3];
    iauC2teqx(sofa_Q, gst, sofa_W_raw, sofa_C2T_raw);

    // Transpose (SOFA returns a matrix in the other direction)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[j][i] = sofa_C2T_raw[i][j];
        }
    }
}

class ForceModelTimePointsTest : public ::testing::Test {
protected:
    cublasHandle_t cublasHandle;
    TimeData defaultStartTime {
        .Year = 2025, .Month = 10, .Day = 22, .Hour = 11, .Min = 0, .Sec = 0.0
    };
    double defaultStepSize = 30;

    void SetUp() override
    {
        cudaSetDevice(0);
        cublasStatus_t stat = cublasCreate(&cublasHandle);
        ASSERT_EQ(stat, CUBLAS_STATUS_SUCCESS) << "CUBLAS initialization failed!";
    }

    void TearDown() override
    {
        if (cublasHandle != nullptr) {
            cublasDestroy(cublasHandle);
        }
        // Check for uncaught errors
        gpuErrchk(cudaGetLastError());
        cudaDeviceReset();
    }
};

TEST_F(ForceModelTimePointsTest, CanConstructAndDestruct)
{
    EXPECT_NO_THROW({ FMTimePoints timePoints(&cublasHandle, defaultStepSize, defaultStartTime); });
}

TEST_F(ForceModelTimePointsTest, InitializeCompletesSuccessfully)
{
    FMTimePoints timePoints(&cublasHandle, defaultStepSize, defaultStartTime);
    timePoints.Initialize();
}

TEST_F(ForceModelTimePointsTest, CalculateParameters)
{
    FMTimePoints timePoints(&cublasHandle, defaultStepSize, defaultStartTime);
    timePoints.Initialize();

    timePoints.CalculateParameters(100, false, 0);
    timePoints.CalculateParameters(10, true, -100);
    timePoints.CalculateParameters(1000, true, 0);
}

TEST_F(ForceModelTimePointsTest, GetParameters)
{
    FMTimePoints timePoints(&cublasHandle, defaultStepSize, defaultStartTime);
    timePoints.Initialize();

    timePoints.CalculateParameters(1, false, 0);

    TimePointParameters paramsDev = timePoints.GetParametersDev(0);
    double ITRS_to_GCRS[9];
    cudaMemcpy(&ITRS_to_GCRS, paramsDev.ITRS_to_GCRS, sizeof(ITRS_to_GCRS), cudaMemcpyDeviceToHost);

    // SOFA Equivalent
    double sofa[3][3];
    GetSofaITRStoGCRS(defaultStartTime, sofa);
    ExpectMatricesNear(ITRS_to_GCRS, sofa, 1e-10);
}

TEST_F(ForceModelTimePointsTest, GetMultipleParametersSequence)
{
    const int numSteps = 10;
    const int stepSize = 86400 * 50; // large step to check far-spread dates
    FMTimePoints timePoints(&cublasHandle, stepSize, defaultStartTime);
    timePoints.Initialize();

    timePoints.CalculateParameters(numSteps, false, 0);

    for (int i = 0; i < numSteps; ++i) {
        TimePointParameters paramsDev = timePoints.GetParametersDev(i);

        double ITRS_to_GCRS_host[9];
        cudaMemcpy(&ITRS_to_GCRS_host, paramsDev.ITRS_to_GCRS, sizeof(ITRS_to_GCRS_host),
            cudaMemcpyDeviceToHost);

        TimeData currentStartTime = defaultStartTime;
        currentStartTime.Sec += i * stepSize;

        double sofa_matrix[3][3];
        GetSofaITRStoGCRS(currentStartTime, sofa_matrix);

        SCOPED_TRACE("Failed at step index: " + std::to_string(i));
        ExpectMatricesNear(ITRS_to_GCRS_host, sofa_matrix, 1e-10);
    }
}

TEST_F(ForceModelTimePointsTest, GetMultipleParametersSequenceBeforeEpoch)
{
    const int beforeEpoch = 5;
    const int numSteps = 15;
    const int stepSize = 86400 * 50;

    FMTimePoints timePoints(&cublasHandle, stepSize, defaultStartTime);
    timePoints.Initialize();

    timePoints.CalculateParameters(numSteps, false, beforeEpoch);

    for (int i = 0; i < numSteps; ++i) {
        TimePointParameters paramsDev = timePoints.GetParametersDev(i);

        double ITRS_to_GCRS_host[9];
        cudaMemcpy(&ITRS_to_GCRS_host, paramsDev.ITRS_to_GCRS, sizeof(ITRS_to_GCRS_host),
            cudaMemcpyDeviceToHost);

        TimeData currentStartTime = defaultStartTime;
        currentStartTime.Sec += (i - beforeEpoch) * stepSize;

        double sofa_matrix[3][3];
        GetSofaITRStoGCRS(currentStartTime, sofa_matrix);

        SCOPED_TRACE("Failed at step index: " + std::to_string(i));
        ExpectMatricesNear(ITRS_to_GCRS_host, sofa_matrix, 1e-10);
    }
}

TEST_F(ForceModelTimePointsTest, GetMultipleParametersSequenceWithMidpoints)
{
    const int numSteps = 30;
    const int stepSize = 86400 * 50; // large step to check far-spread dates
    FMTimePoints timePoints(&cublasHandle, stepSize, defaultStartTime);
    timePoints.Initialize();

    timePoints.CalculateParameters(numSteps, true, 0);

    constexpr double host_rk8Midpoints[] = { 0.0, 4.0 / 27.0, 2.0 / 9.0, 1.0 / 3.0, 1.0 / 2.0,
        2.0 / 3.0, 1.0 / 6.0, 1.0, 5.0 / 6.0, 1.0 };

    for (int i = 0; i < numSteps; ++i) {
        TimePointParameters paramsDev = timePoints.GetParametersDev(i);

        double ITRS_to_GCRS_host[9];
        cudaMemcpy(&ITRS_to_GCRS_host, paramsDev.ITRS_to_GCRS, sizeof(ITRS_to_GCRS_host),
            cudaMemcpyDeviceToHost);

        int midpointCount = CFMConst::rk8StepCount;
        int midpointIndex = i % midpointCount;
        double timeFromStart
            = ((double)(i / midpointCount) + host_rk8Midpoints[midpointIndex]) * stepSize;

        TimeData currentStartTime = defaultStartTime;
        currentStartTime.Sec += timeFromStart;

        double sofa_matrix[3][3];
        GetSofaITRStoGCRS(currentStartTime, sofa_matrix);

        SCOPED_TRACE("Failed at step index: " + std::to_string(i));
        ExpectMatricesNear(ITRS_to_GCRS_host, sofa_matrix, 1e-10);
    }
}
