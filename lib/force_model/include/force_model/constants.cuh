#pragma once

// Cuda Force Model constants
namespace CFMConst {

constexpr double PI = 3.14159265358979324;
constexpr double ArcsPerRad = 3600.0 * 180.0 / PI; // Arcseconds per radian
constexpr double RadPerDeg = PI / 180.0; // Radians per degree

constexpr double MJD_J2000 = 51544.5; // Modified Julian Date of J2000.0 (TT)
constexpr double MJD_EPOCH_1900 = 15020.0; // Modified Julian Date of 1900-01-01 00:00:00 (UTC)
constexpr double TT_TAI = 32.184; // TT TAI offset in seconds

// Fractions of step size for calculating k-values in RK8 initial point generation
constexpr int rk8StepCount = 10;
__device__ constexpr double rk8Midpoints[rk8StepCount] { 0.0, 4.0 / 27.0, 2.0 / 9.0, 1.0 / 3.0,
    1.0 / 2.0, 2.0 / 3.0, 1.0 / 6.0, 1.0, 5.0 / 6.0, 1.0 };

}