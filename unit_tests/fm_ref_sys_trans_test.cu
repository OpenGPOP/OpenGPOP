#include "force_model/constants.cuh"
#include "force_model/types.cuh"
#include "force_model/utility/dataset_loaders.cuh"
#include "force_model/utility/ref_sys_trans.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include "sofa.h"
#include <gtest/gtest.h>

// Helper to compare a flat 1D matrix (9 elements) against a SOFA 3x3 matrix
void ExpectMatricesNear(
    const double* actual_matrix, double sofa_matrix[3][3], double tolerance = 1e-12)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(actual_matrix[SQ_C2IDX(i, j, 3)], sofa_matrix[i][j], tolerance);
        }
    }
}

// Column-major
void MultiplyMatrix(double* A, double* B, double* C)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[j * 3 + i] = 0.0;
            for (int k = 0; k < 3; ++k) {
                C[j * 3 + i] += A[k * 3 + i] * B[j * 3 + k];
            }
        }
    }
}

TEST(ReferenceSystemTransformationsTest, RotX)
{
    double angle = 0.45;
    double actual_R[9] = { 0 };
    RefSysTrans::CreateRotX(angle, actual_R);

    double sofa_R[3][3];
    iauIr(sofa_R);
    iauRx(angle, sofa_R);

    ExpectMatricesNear(actual_R, sofa_R);
}

TEST(ReferenceSystemTransformationsTest, RotY)
{
    double angle = 0.45;
    double actual_R[9] = { 0 };
    RefSysTrans::CreateRotY(angle, actual_R);

    double sofa_R[3][3];
    iauIr(sofa_R);
    iauRy(angle, sofa_R);

    ExpectMatricesNear(actual_R, sofa_R);
}

TEST(ReferenceSystemTransformationsTest, RotZ)
{
    double angle = 0.45;
    double actual_R[9] = { 0 };
    RefSysTrans::CreateRotZ(angle, actual_R);

    double sofa_R[3][3];
    iauIr(sofa_R);
    iauRz(angle, sofa_R);

    ExpectMatricesNear(actual_R, sofa_R);
}

TEST(ReferenceSystemTransformationsTest, CalendarToMJD)
{
    int y = 2026, m = 3, d = 11, h = 16, min = 13;
    double sec = 30.5;

    double actual_mjd = RefSysTrans::CalendarToMJD(y, m, d, h, min, sec);

    double djm0, djm;
    int status = iauCal2jd(y, m, d, &djm0, &djm);
    EXPECT_EQ(status, 0) << "SOFA rejected the date.";

    // Add the time fraction to the MJD
    double time_fraction = (h + min / 60.0 + sec / 3600.0) / 24.0;
    double sofa_mjd = djm + time_fraction;

    EXPECT_NEAR(actual_mjd, sofa_mjd, 1e-11);
}

TEST(ReferenceSystemTransformationsTest, ComputeFTable)
{
    double t = 0.15;
    double actual_F[14];
    RefSysTrans::ComputeFTable(t, actual_F);

    double sofa_F[14];
    sofa_F[0] = iauFal03(t); // l  : Moon's mean anomaly
    sofa_F[1] = iauFalp03(t); // l' : Sun's mean anomaly
    sofa_F[2] = iauFaf03(t); // F  : Moon's mean argument of latitude
    sofa_F[3] = iauFad03(t); // D  : Mean elongation of the Moon from the Sun
    sofa_F[4] = iauFaom03(t); // Om : Mean longitude of Moon's ascending node
    sofa_F[5] = iauFame03(t); // L_Me : Mercury
    sofa_F[6] = iauFave03(t); // L_Ve : Venus
    sofa_F[7] = iauFae03(t); // L_E  : Earth
    sofa_F[8] = iauFama03(t); // L_Ma : Mars
    sofa_F[9] = iauFaju03(t); // L_J  : Jupiter
    sofa_F[10] = iauFasa03(t); // L_Sa : Saturn
    sofa_F[11] = iauFaur03(t); // L_U  : Uranus
    sofa_F[12] = iauFane03(t); // L_Ne : Neptune
    sofa_F[13] = iauFapa03(t); // p_A  : General accumulated precession

    for (int i = 0; i < 14; i++) {
        EXPECT_NEAR(actual_F[i], sofa_F[i], 1e-12);
    }
}

TEST(ReferenceSystemTransformationsTest, CalculatePrecessionQuantities)
{
    double t = 0.15;
    double actual_eps_A = 0, actual_gamma_bar = 0, actual_phi_bar = 0, actual_psi_bar = 0;
    RefSysTrans::CalculatePrecessionQuantities(
        t, actual_eps_A, actual_gamma_bar, actual_phi_bar, actual_psi_bar);

    // SOFA Equivalent
    double date1 = 2451545.0;
    double date2 = t * 36525.0;

    double sofa_gamma_bar, sofa_phi_bar, sofa_psi_bar, sofa_eps_A;
    iauPfw06(date1, date2, &sofa_gamma_bar, &sofa_phi_bar, &sofa_psi_bar, &sofa_eps_A);

    EXPECT_NEAR(actual_eps_A, sofa_eps_A, 1e-12);
    EXPECT_NEAR(actual_gamma_bar, sofa_gamma_bar, 1e-12);
    EXPECT_NEAR(actual_phi_bar, sofa_phi_bar, 1e-12);
    EXPECT_NEAR(actual_psi_bar, sofa_psi_bar, 1e-12);
}

TEST(ReferenceSystemTransformationsTest, NutationLongitudeObliquity)
{
    double mjd_TT = 60000.0;
    double t = (mjd_TT - 51544.5) / 36525.0;

    double F[14] = { 0 };
    RefSysTrans::ComputeFTable(t, F);

    double dpsi = 0, deps = 0;
    NutationTables psi_tables
        = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutLon.txt");
    dpsi += RefSysTrans::CalculateDelta(F, psi_tables.j0_size, psi_tables.j0);
    dpsi += RefSysTrans::CalculateDelta(F, psi_tables.j1_size, psi_tables.j1) * t;
    delete[] psi_tables.j0;
    delete[] psi_tables.j1;

    NutationTables eps_tables
        = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutObl.txt");
    deps += RefSysTrans::CalculateDelta(F, eps_tables.j0_size, eps_tables.j0);
    deps += RefSysTrans::CalculateDelta(F, eps_tables.j1_size, eps_tables.j1) * t;
    delete[] eps_tables.j0;
    delete[] eps_tables.j1;

    // SOFA Equivalent
    double sofa_dpsi, sofa_deps;
    iauNut06a(2400000.5, mjd_TT, &sofa_dpsi, &sofa_deps);

    EXPECT_NEAR(dpsi, sofa_dpsi, 1e-11);
    EXPECT_NEAR(deps, sofa_deps, 1e-11);
}

TEST(ReferenceSystemTransformationsTest, GAST_Calculation)
{
    double mjd_UT1 = 60000.1;
    double mjd_TT = 60000.1008;

    double F[14];
    double t = (mjd_TT - CFMConst::MJD_J2000) / 36525.0;
    RefSysTrans::ComputeFTable(t, F);

    double eps_A = iauObl06(2400000.5, mjd_TT);
    double dpsi, deps;
    iauNut06a(2400000.5, mjd_TT, &dpsi, &deps);

    double actual_gast = RefSysTrans::GAST(mjd_UT1, mjd_TT, dpsi, eps_A, F);

    // SOFA equivalent
    double sofa_gast = iauGst06a(2400000.5, mjd_UT1, 2400000.5, mjd_TT);

    EXPECT_NEAR(actual_gast, sofa_gast, 1e-11);
}

TEST(ReferenceSystemTransformationsTest, QMatrix_PrecessionNutation)
{
    TimePointDatasets datasets;
    datasets.nutDataset.lon = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutLon.txt");
    datasets.nutDataset.obl = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutObl.txt");
    datasets.eopDataset = DatasetLoaders::LoadEOPData("../../data/IAU2000A_EOP.txt");
    datasets.lsDataset = DatasetLoaders::LoadLeapSeconds("../../data/leap-seconds.list");

    double mjd_TT = 60000.0;
    double R1[9], R2[9], R3[9], R4[9];
    double F[14], delta_psi, eps_A;
    RefSysTrans::CreateQMatrixComponents(R1, R2, R3, R4, mjd_TT, &datasets, F, &delta_psi, &eps_A);

    delete[] datasets.nutDataset.lon.j0;
    delete[] datasets.nutDataset.lon.j1;
    delete[] datasets.nutDataset.obl.j0;
    delete[] datasets.nutDataset.obl.j1;
    delete[] datasets.eopDataset.records;
    delete[] datasets.lsDataset.leap_seconds;
    delete[] datasets.lsDataset.mjd_utc;

    double R12[9], R123[9], actual_Q[9];
    MultiplyMatrix(R1, R2, R12);
    MultiplyMatrix(R12, R3, R123);
    MultiplyMatrix(R123, R4, actual_Q);

    // SOFA equivalent
    double sofa_Q_T[3][3];
    iauPnm06a(2400000.5, mjd_TT, sofa_Q_T);

    // Transpose (SOFA returns a matrix in the other direction)
    double sofa_Q[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sofa_Q[j][i] = sofa_Q_T[i][j];
        }
    }

    ExpectMatricesNear(actual_Q, sofa_Q, 1e-11);
}

TEST(ReferenceSystemTransformationsTest, WMatrix_PolarMotion)
{
    double mjd_TT = 60000.0;

    EOPRecord eop_record;
    eop_record.xp = 5e-7;
    eop_record.yp = 3e-7;

    double R1[9], R2[9], R3[9];
    RefSysTrans::CreateWMatrixComponents(R1, R2, R3, mjd_TT, eop_record);

    double R12[9], actual_W[9];
    MultiplyMatrix(R1, R2, R12);
    MultiplyMatrix(R12, R3, actual_W);

    // SOFA Equivalent
    double sofa_W_T[3][3];
    double sp = iauSp00(2400000.5, mjd_TT);
    iauPom00(eop_record.xp, eop_record.yp, sp, sofa_W_T);

    // Transpose (SOFA returns a matrix in the other direction)
    double sofa_W[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sofa_W[j][i] = sofa_W_T[i][j];
        }
    }

    ExpectMatricesNear(actual_W, sofa_W, 1e-12);
}

TEST(ReferenceSystemTransformationsTest, RMatrix_EarthRotation)
{
    double mjd_UT1 = 60000.0;
    double mjd_TT = 60000.0008;

    double eps_A = iauObl06(2400000.5, mjd_TT);
    double dpsi, deps;
    iauNut06a(2400000.5, mjd_TT, &dpsi, &deps);

    double F[14];
    double t = (mjd_TT - CFMConst::MJD_J2000) / 36525.0;
    RefSysTrans::ComputeFTable(t, F);

    double actual_R[9];
    RefSysTrans::CreateRMatrix(actual_R, mjd_UT1, mjd_TT, dpsi, eps_A, F);

    // SOFA Equivalent
    double sofa_R_T[3][3];
    iauIr(sofa_R_T);
    double gast = iauGst06a(2400000.5, mjd_UT1, 2400000.5, mjd_TT);
    iauRz(gast, sofa_R_T);

    // Transpose (SOFA returns a matrix in the other direction)
    double sofa_R[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sofa_R[j][i] = sofa_R_T[i][j];
        }
    }

    ExpectMatricesNear(actual_R, sofa_R, 1e-12);
}

TEST(ReferenceSystemTransformationsTest, NTPToMJD_Conversion)
{
    // Test 1: The NTP Epoch (Jan 1, 1900 00:00:00)
    long long ntp_epoch = 0;
    double expected_mjd_epoch = 15020.0;
    EXPECT_NEAR(RefSysTrans::NTPToMJD(ntp_epoch), expected_mjd_epoch, 1e-10);

    // Test 2: Exactly one standard day later
    long long ntp_one_day = 86400;
    double expected_mjd_one_day = 15021.0;
    EXPECT_NEAR(RefSysTrans::NTPToMJD(ntp_one_day), expected_mjd_one_day, 1e-10);

    // Test 3: J2000 Epoch (Jan 1, 2000 12:00:00 TT/UTC)
    long long ntp_j2000_noon = 3155673600 + 43200;
    double expected_mjd_j2000 = 51544.5;
    EXPECT_NEAR(RefSysTrans::NTPToMJD(ntp_j2000_noon), expected_mjd_j2000, 1e-10);
}

TEST(ReferenceSystemTransformationsTest, FullTransformation_T2C)
{
    double mjd_TT = 60000.0;
    double mjd_UT1 = 60000.0008;

    EOPRecord eop_record;
    eop_record.xp = 5e-7;
    eop_record.yp = 3e-7;

    TimePointDatasets datasets;
    datasets.nutDataset.lon = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutLon.txt");
    datasets.nutDataset.obl = DatasetLoaders::LoadNutationData("../../data/IAU2000_R06_NutObl.txt");
    datasets.eopDataset = DatasetLoaders::LoadEOPData("../../data/IAU2000A_EOP.txt");
    datasets.lsDataset = DatasetLoaders::LoadLeapSeconds("../../data/leap-seconds.list");

    double qR1[9], qR2[9], qR3[9], qR4[9];
    double F[14], delta_psi, eps_A;
    RefSysTrans::CreateQMatrixComponents(
        qR1, qR2, qR3, qR4, mjd_TT, &datasets, F, &delta_psi, &eps_A);

    double actual_R[9];
    RefSysTrans::CreateRMatrix(actual_R, mjd_UT1, mjd_TT, delta_psi, eps_A, F);

    double wR1[9], wR2[9], wR3[9];
    RefSysTrans::CreateWMatrixComponents(wR1, wR2, wR3, mjd_TT, eop_record);

    delete[] datasets.nutDataset.lon.j0;
    delete[] datasets.nutDataset.lon.j1;
    delete[] datasets.nutDataset.obl.j0;
    delete[] datasets.nutDataset.obl.j1;
    delete[] datasets.eopDataset.records;
    delete[] datasets.lsDataset.leap_seconds;
    delete[] datasets.lsDataset.mjd_utc;

    double qR12[9], qR123[9], actual_Q[9];
    MultiplyMatrix(qR1, qR2, qR12);
    MultiplyMatrix(qR12, qR3, qR123);
    MultiplyMatrix(qR123, qR4, actual_Q);

    double wR12[9], actual_W[9];
    MultiplyMatrix(wR1, wR2, wR12);
    MultiplyMatrix(wR12, wR3, actual_W);

    double QR[9], ITRStoGCRS[9];
    MultiplyMatrix(actual_Q, actual_R, QR);
    MultiplyMatrix(QR, actual_W, ITRStoGCRS);

    // SOFA Equivalent
    double sofa_Q[3][3];
    iauPnm06a(2400000.5, mjd_TT, sofa_Q);

    double sofa_W_raw[3][3];
    double sp = iauSp00(2400000.5, mjd_TT);
    iauPom00(eop_record.xp, eop_record.yp, sp, sofa_W_raw);

    double gst = iauGst06a(2400000.5, mjd_UT1, 2400000.5, mjd_TT);
    double sofa_C2T_raw[3][3];
    iauC2teqx(sofa_Q, gst, sofa_W_raw, sofa_C2T_raw);

    // Transpose (SOFA returns a matrix in the other direction)
    double sofa_T2C[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            sofa_T2C[j][i] = sofa_C2T_raw[i][j];
        }
    }

    ExpectMatricesNear(ITRStoGCRS, sofa_T2C, 1e-11);
}

TEST(ReferenceSystemTransformationsTest, TimeScales_UTC_to_TT_UT1)
{
    LeapSecondDataset lsDataset = DatasetLoaders::LoadLeapSeconds("../../data/leap-seconds.list");
    EOPDataset eopDataset = DatasetLoaders::LoadEOPData("../../data/IAU2000A_EOP.txt");

    // Array of MJDs representing different leap-second eras
    // The last date is far into the future and should fail if SOFA and LeapSecondDataset don't
    // agree on a comon value (for example if one is updated and the other isn't)
    const double test_mjds[] = { 41317.0, 46066.0, 51544.5, 60000.0, 999999.0 };
    for (double mjd_UTC : test_mjds) {
        auto eop_record = eopDataset.getRecord(mjd_UTC);

        double actual_TT = RefSysTrans::MJD_UTCtoTT(mjd_UTC, lsDataset);
        double actual_UT1 = RefSysTrans::MJD_UTCtoUT1(mjd_UTC, eop_record);

        // SOFA Equivalent
        double u1 = 2400000.5;
        double u2 = mjd_UTC;
        double tai1, tai2, tt1, tt2, ut11, ut12;

        // SOFA: UTC -> TAI -> TT
        iauUtctai(u1, u2, &tai1, &tai2);
        iauTaitt(tai1, tai2, &tt1, &tt2);

        // SOFA: UTC -> UT1
        iauUtcut1(u1, u2, eop_record.ut1_utc, &ut11, &ut12);

        // Combine SOFA 2-part JDs back into a single MJD double
        double sofa_TT = (tt1 - 2400000.5) + tt2;
        double sofa_UT1 = (ut11 - 2400000.5) + ut12;

        EXPECT_DOUBLE_EQ(actual_TT, sofa_TT);
        EXPECT_DOUBLE_EQ(actual_UT1, sofa_UT1);
    }

    delete[] lsDataset.leap_seconds;
    delete[] lsDataset.mjd_utc;
    delete[] eopDataset.records;
}