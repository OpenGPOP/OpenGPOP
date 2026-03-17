#include "force_model/constants.cuh"
#include "force_model/utility/ref_sys_trans.cuh"
#include "gpu_utils/gpu_utils.cuh"

// Utility functions for reference system transformations
namespace RefSysTrans {

__host__ __device__ void CreateRotX(double angle, double* ptr)
{
    double c = cos(angle);
    double s = sin(angle);
    ptr[SQ_C2IDX(0, 0, 3)] = 1.0;
    ptr[SQ_C2IDX(0, 1, 3)] = 0.0;
    ptr[SQ_C2IDX(0, 2, 3)] = 0.0;
    ptr[SQ_C2IDX(1, 0, 3)] = 0.0;
    ptr[SQ_C2IDX(1, 1, 3)] = c;
    ptr[SQ_C2IDX(1, 2, 3)] = s;
    ptr[SQ_C2IDX(2, 0, 3)] = 0.0;
    ptr[SQ_C2IDX(2, 1, 3)] = -s;
    ptr[SQ_C2IDX(2, 2, 3)] = c;
}

__host__ __device__ void CreateRotY(double angle, double* ptr)
{
    double c = cos(angle);
    double s = sin(angle);
    ptr[SQ_C2IDX(0, 0, 3)] = c;
    ptr[SQ_C2IDX(0, 1, 3)] = 0.0;
    ptr[SQ_C2IDX(0, 2, 3)] = -s;
    ptr[SQ_C2IDX(1, 0, 3)] = 0.0;
    ptr[SQ_C2IDX(1, 1, 3)] = 1.0;
    ptr[SQ_C2IDX(1, 2, 3)] = 0.0;
    ptr[SQ_C2IDX(2, 0, 3)] = s;
    ptr[SQ_C2IDX(2, 1, 3)] = 0.0;
    ptr[SQ_C2IDX(2, 2, 3)] = c;
}

__host__ __device__ void CreateRotZ(double angle, double* ptr)
{
    double c = cos(angle);
    double s = sin(angle);
    ptr[SQ_C2IDX(0, 0, 3)] = c;
    ptr[SQ_C2IDX(0, 1, 3)] = s;
    ptr[SQ_C2IDX(0, 2, 3)] = 0.0;
    ptr[SQ_C2IDX(1, 0, 3)] = -s;
    ptr[SQ_C2IDX(1, 1, 3)] = c;
    ptr[SQ_C2IDX(1, 2, 3)] = 0.0;
    ptr[SQ_C2IDX(2, 0, 3)] = 0.0;
    ptr[SQ_C2IDX(2, 1, 3)] = 0.0;
    ptr[SQ_C2IDX(2, 2, 3)] = 1.0;
}

// Computes the Greenwich Apparent Sidereal Time (GAST) using the Equation of Equinoxes method.
// (p. B9)
//
// Based on:
// The Astronomical Almanac 2010
// https://archive.org/details/astronomicalalma0000unse_n8r6
__host__ __device__ double GAST(
    double mjd_UT1, double mjd_TT, double delta_psi, double eps_A, double* F)
{
    // p. B6
    double T = (mjd_TT - CFMConst::MJD_J2000) / 36525.0;
    double Du = mjd_UT1 - CFMConst::MJD_J2000;

    // Calculate GMST in seconds (p. B8)
    double Du_mod_1 = Du - floor(Du);
    double GMST = 86400.0 * (0.7790572732640 + 0.00273781191135448 * Du + Du_mod_1);
    const double coeffs[6]
        = { 0.00096707, 307.47710227, 0.092772113, -0.0000000293, -0.00000199707, -2.453e-9 };
    double poly = coeffs[5];
    for (int i = 4; i >= 0; --i) {
        poly = poly * T + coeffs[i];
    }
    GMST += poly;
    GMST *= (M_PI / 43200.0); // convert to radians

    // Calculate the Equation of Equinoxes in seconds (p. B10)
    double l = F[0];
    double l_prime = F[1];
    double f_arg = F[2];
    double D = F[3];
    double Omega = F[4];
    double EqEquinoxes = 0.0;
    EqEquinoxes -= 0.00000087 * T * sin(Omega);
    EqEquinoxes -= 0.00000063 * sin(l - Omega);
    EqEquinoxes -= 0.00000063 * sin(l + Omega);
    EqEquinoxes -= 0.00000126 * sin(l_prime - Omega);
    EqEquinoxes -= 0.00000141 * sin(l_prime + Omega);
    EqEquinoxes -= 0.00000172 * sin(3.0 * Omega);
    EqEquinoxes += 0.00000198 * sin(2.0 * f_arg + Omega);
    EqEquinoxes += 0.00000202 * sin(2.0 * f_arg + 3.0 * Omega);
    EqEquinoxes -= 0.00000455 * sin(2.0 * f_arg - 2.0 * D + 2.0 * Omega);
    EqEquinoxes += 0.00001121 * sin(2.0 * f_arg - 2.0 * D + Omega);
    EqEquinoxes += 0.00001175 * sin(2.0 * f_arg - 2.0 * D + 3.0 * Omega);
    EqEquinoxes += 0.00006352 * sin(2.0 * Omega);
    EqEquinoxes += 0.00264096 * sin(Omega);
    EqEquinoxes *= (M_PI / 43200.0) / 15.0; // Convert to radians
    EqEquinoxes += delta_psi * cos(eps_A);

    // Normalize
    return fmod(GMST + EqEquinoxes, 2 * M_PI);
}

// Converts calendar date and time to Modified Julian Date (MJD)
__host__ __device__ double CalendarToMJD(int year, int mon, int day, int h, int min, double sec)
{
    // Based on O. Montenbruck and E. Gill, Satellite Orbits: Models, Methods and Applications
    if (mon <= 2) {
        --year; // eq. A.3
        mon += 12; // eq. A.4
    }

    // eq. A.5
    int B;
    if ((10000L * year + 100L * mon + day) <= 15821004L)
        B = -2 + ((year + 4716) / 4) - 1179;
    else if ((10000L * year + 100L * mon + day)
        >= 15821015L) // the book mentions 1582 10 10, but no such date existed?
        B = (year / 400) - (year / 100) + (year / 4);
    else
        return -1.0; // No such date

    // eq. A.6
    long mjd = 365L * year - 679004L + B + int(30.6001 * (mon + 1)) + day;
    double frac = (h + min / 60.0 + sec / 3600.0) / 24.0;
    return mjd + frac;
}

// Converts MJD in UTC to MJD in TT
__host__ __device__ double MJD_UTCtoTT(double mjd_UTC, LeapSecondDataset lsDataset)
{
    return mjd_UTC + (lsDataset.getLeapSeconds(mjd_UTC) + CFMConst::TT_TAI) / 86400.0;
}

// Converts MJD in UTC to MJD in UT1
__host__ __device__ double MJD_UTCtoUT1(double mjd_UTC, EOPRecord eop)
{
    return mjd_UTC + eop.ut1_utc / 86400.0;
}

// Produces the F table, as per eq. 5.43 and 5.44.
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void ComputeFTable(double t, double* F)
{
    // Lunisolar nutation (eq. 5.43)
    const double l_coeffs[]
        = { 134.96340251 * 3600.0, 1717915923.2178, 31.8792, 0.051635, -0.00024470 };
    const double l_prime_coeffs[]
        = { 357.52910918 * 3600.0, 129596581.0481, -0.5532, 0.000136, -0.00001149 };
    const double f_arg_coeffs[]
        = { 93.27209062 * 3600.0, 1739527262.8478, -12.7512, -0.001037, 0.00000417 };
    const double d_coeffs[]
        = { 297.85019547 * 3600.0, 1602961601.2090, -6.3706, 0.006593, -0.00003169 };
    const double omega_coeffs[]
        = { 125.04455501 * 3600.0, -6962890.5431, 7.4722, 0.007702, -0.00005939 };

    F[0] = l_coeffs[4];
    F[1] = l_prime_coeffs[4];
    F[2] = f_arg_coeffs[4];
    F[3] = d_coeffs[4];
    F[4] = omega_coeffs[4];

    for (int i = 3; i >= 0; --i) {
        F[0] = F[0] * t + l_coeffs[i];
        F[1] = F[1] * t + l_prime_coeffs[i];
        F[2] = F[2] * t + f_arg_coeffs[i];
        F[3] = F[3] * t + d_coeffs[i];
        F[4] = F[4] * t + omega_coeffs[i];
    }

    for (int i = 0; i < 5; i++) {
        F[i] /= CFMConst::ArcsPerRad;
    }

    // Planetary nutation (eq. 5.44)
    const double L_Me_coeffs[] = { 4.402608842, 2608.7903141574, 0.0 };
    const double L_Ve_coeffs[] = { 3.176146697, 1021.3285546211, 0.0 };
    const double L_E_coeffs[] = { 1.753470314, 628.3075849991, 0.0 };
    const double L_Ma_coeffs[] = { 6.203480913, 334.0612426700, 0.0 };
    const double L_J_coeffs[] = { 0.599546497, 52.9690962641, 0.0 };
    const double L_Sa_coeffs[] = { 0.874016757, 21.3299104960, 0.0 };
    const double L_U_coeffs[] = { 5.481293872, 7.4781598567, 0.0 };
    const double L_Ne_coeffs[] = { 5.311886287, 3.8133035638, 0.0 };
    const double p_A_coeffs[] = { 0.0, 0.02438175, 0.00000538691 };

    F[5] = L_Me_coeffs[2];
    F[6] = L_Ve_coeffs[2];
    F[7] = L_E_coeffs[2];
    F[8] = L_Ma_coeffs[2];
    F[9] = L_J_coeffs[2];
    F[10] = L_Sa_coeffs[2];
    F[11] = L_U_coeffs[2];
    F[12] = L_Ne_coeffs[2];
    F[13] = p_A_coeffs[2];

    for (int i = 1; i >= 0; --i) {
        F[5] = F[5] * t + L_Me_coeffs[i];
        F[6] = F[6] * t + L_Ve_coeffs[i];
        F[7] = F[7] * t + L_E_coeffs[i];
        F[8] = F[8] * t + L_Ma_coeffs[i];
        F[9] = F[9] * t + L_J_coeffs[i];
        F[10] = F[10] * t + L_Sa_coeffs[i];
        F[11] = F[11] * t + L_U_coeffs[i];
        F[12] = F[12] * t + L_Ne_coeffs[i];
        F[13] = F[13] * t + p_A_coeffs[i];
    }

    // Normalize
    for (int i = 0; i < 14; i++) {
        F[i] = fmod(F[i], 2.0 * M_PI);
    }
}

// Calculates the psi/epsilon delta, as per eq. 5.35.
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ double CalculateDelta(double* F, int table_size, NutationTerm* terms)
{
    double result = 0;
    // We add from the smallest values to preserve numerical precision
    for (int i = table_size - 1; i >= 0; i--) {
        double arg = 0;
        for (int k = 13; k >= 0; k--) {
            arg += F[k] * terms[i].args[k];
        }
        result += terms[i].sine_coeff_uas / (CFMConst::ArcsPerRad * 1e6) * sin(arg);
        result += terms[i].cosine_coeff_uas / (CFMConst::ArcsPerRad * 1e6) * cos(arg);
    }
    return result;
}

// Calculates the precession quantities - eps_A (mean obliquity), gamma_bar, phi_bar, psi_bar
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void CalculatePrecessionQuantities(
    double t, double& eps_A, double& gamma_bar, double& phi_bar, double& psi_bar)
{
    // eq. 5.40
    const double eps_A_coeffs[]
        = { 84381.406, -46.836769, -0.0001831, 0.00200340, -0.000000576, -0.0000000434 };
    const double gamma_coeffs[]
        = { -0.052928, 10.556378, 0.4932044, -0.00031238, -0.000002788, 0.0000000260 };
    const double phi_coeffs[]
        = { 84381.412819, -46.811016, 0.0511268, 0.00053289, -0.000000440, -0.0000000176 };
    const double psi_coeffs[]
        = { -0.041775, 5038.481484, 1.5584175, -0.00018522, -0.000026452, -0.0000000148 };

    eps_A = eps_A_coeffs[5];
    gamma_bar = gamma_coeffs[5];
    phi_bar = phi_coeffs[5];
    psi_bar = psi_coeffs[5];

    for (int i = 4; i >= 0; --i) {
        eps_A = eps_A * t + eps_A_coeffs[i];
        gamma_bar = gamma_bar * t + gamma_coeffs[i];
        phi_bar = phi_bar * t + phi_coeffs[i];
        psi_bar = psi_bar * t + psi_coeffs[i];
    }

    eps_A /= CFMConst::ArcsPerRad;
    gamma_bar /= CFMConst::ArcsPerRad;
    phi_bar /= CFMConst::ArcsPerRad;
    psi_bar /= CFMConst::ArcsPerRad;
}

// Creates four rotation matrices, as per the IAU 2006/2000A precession-nutation model, which
// multiplied produce the Q(t) matrix, accounting for bias, precession and nutation. It uses the
// Fukushima-Williams method, as described in ch. 5.4.5.
//
// F, delta_psi_out and eps_A_out are outputted for later use.
//
// Note: In The Astronomical Almanach 2010 the Q(t) matrix is referred to as M.
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void CreateQMatrixComponents(double* R1, double* R2, double* R3, double* R4,
    double mjd_TT, TimePointDatasets* datasets, double* F, double* delta_psi_out, double* eps_A_out)
{
    double t = (mjd_TT - CFMConst::MJD_J2000) / 36525.0; // eq. 5.2

    // Calculate delta_psi and delta_eps using eq. 5.35
    double delta_psi = 0, delta_eps = 0;
    ComputeFTable(t, F);

    NutationTables nut_tables = datasets->nutDataset.lon;
    delta_psi += CalculateDelta(F, nut_tables.j0_size, nut_tables.j0);
    delta_psi += CalculateDelta(F, nut_tables.j1_size, nut_tables.j1) * t;

    nut_tables = datasets->nutDataset.obl;
    delta_eps += CalculateDelta(F, nut_tables.j0_size, nut_tables.j0);
    delta_eps += CalculateDelta(F, nut_tables.j1_size, nut_tables.j1) * t;

    double eps_A, gamma_bar, phi_bar, psi_bar;
    CalculatePrecessionQuantities(t, eps_A, gamma_bar, phi_bar, psi_bar);

    // Equations explicitly provided in The Astronomical Almanach 2010, p. B56
    double psi = psi_bar + delta_psi;
    double eps = eps_A + delta_eps;

    // ch. 5.4.5, but flipped for the other direction
    CreateRotZ(-gamma_bar, R1);
    CreateRotX(-phi_bar, R2);
    CreateRotZ(psi, R3);
    CreateRotX(eps, R4);

    *delta_psi_out = delta_psi;
    *eps_A_out = eps_A;
}

// Creates three rotation matrices, which multiplied produce the W(t) matrix, accounting for polar
// motion. (ch. 5.4.1)
//
// Note: This implementation ommits ocean tide and libration variations described in ch. 5.5.1
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void CreateWMatrixComponents(
    double* R1, double* R2, double* R3, double mjd_TT, EOPRecord eop_record)
{
    double t = (mjd_TT - CFMConst::MJD_J2000) / 36525.0; // eq. 5.2
    double s_prim = -47e-6 * t / CFMConst::ArcsPerRad; // eq. 5.13

    // eq. 5.3
    CreateRotZ(-s_prim, R1);
    CreateRotY(eop_record.xp, R2);
    CreateRotX(eop_record.yp, R3);
}

// Creates the R(t) matrix, accounting for the Earth's rotation. (ch. 5.4.3)
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void CreateRMatrix(
    double* R, double mjd_UT1, double mjd_TT, double delta_psi, double eps_A, double* F)
{
    CreateRotZ(-GAST(mjd_UT1, mjd_TT, delta_psi, eps_A, F), R);
}

// Converts an NTP timestamp (seconds since Jan 1, 1900) to MJD
__host__ __device__ double NTPToMJD(long long ntp_seconds)
{
    return CFMConst::MJD_EPOCH_1900 + (static_cast<double>(ntp_seconds) / 86400.0);
}
}