#pragma once
#include "force_model/types.cuh"

// Utility functions for reference system transformations
namespace RefSysTrans {

__host__ __device__ void CreateRotX(double angle, double* ptr);

__host__ __device__ void CreateRotY(double angle, double* ptr);

__host__ __device__ void CreateRotZ(double angle, double* ptr);

// Computes the Greenwich Apparent Sidereal Time (GAST) using the Equation of Equinoxes method.
// (p. B9)
//
// Based on:
// The Astronomical Almanac 2010
// https://archive.org/details/astronomicalalma0000unse_n8r6
__host__ __device__ double GAST(
    double mjd_UT1, double mjd_TT, double delta_psi, double eps_A, double* F);

// Converts calendar date and time to Modified Julian Date (MJD)
__host__ __device__ double CalendarToMJD(int year, int mon, int day, int h, int min, double sec);

// Converts MJD in UTC to MJD in TT
__host__ __device__ double MJD_UTCtoTT(double mjd_UTC, LeapSecondDataset lsDataset);

// Converts MJD in UTC to MJD in UT1
__host__ __device__ double MJD_UTCtoUT1(double mjd_UTC, EOPRecord eop);

// __host__ __device__ double MeanObliquity(double mjd_TT);

// Produces the F table, as per eq. 5.43 and 5.44.
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void ComputeFTable(double t, double* F);

// Calculates the psi/epsilon delta, as per eq. 5.35.
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ double CalculateDelta(double* F, int table_size, NutationTerm* terms);

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
    double mjd_TT, TimePointDatasets* datasets, double* F, double* delta_psi_out,
    double* eps_A_out);

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
    double* R1, double* R2, double* R3, double mjd_TT, EOPRecord eop_record);

// Creates the R(t) matrix, accounting for the Earth's rotation. (ch. 5.4.3)
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void CreateRMatrix(
    double* R, double mjd_UT1, double mjd_TT, double delta_psi, double eps_A, double* F);

// Converts an NTP timestamp (seconds since Jan 1, 1900) to MJD
__host__ __device__ double NTPToMJD(long long ntp_seconds);

// Calculates the precession quantities - eps_A (mean obliquity), gamma_bar, phi_bar, psi_bar
//
// Based on:
// IERS Conventions 2010 Chapter 5 - Transformation between the International Terrestrial Reference
// System and the Geocentric Celestial Reference System (10 August 2012)
// https://iers-conventions.obspm.fr/chapter5.php
__host__ __device__ void CalculatePrecessionQuantities(
    double t, double& eps_A, double& gamma_bar, double& phi_bar, double& psi_bar);

}