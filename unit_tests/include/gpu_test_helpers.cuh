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

struct NoAcceleration {
    __device__ double* operator()(double pos[3], double vel[3], double t)
    {
        auto res = new double[3];
        res[0] = 0;
        res[1] = 0;
        res[2] = 0;
        return res;
    }
};

struct ZeroPosition {
    double operator()(double t) { return 0.0; }
};

struct AccelerationFunction {
    __device__ double* operator()(double pos[3], double vel[3], double t)
    {
        auto res = new double[3];
        res[0] = 1;
        res[1] = 2;
        res[2] = 3;
        return res;
    }
};

struct Exact7Position {
    double operator()(double t) { return pow(t, 7) - pow(t, 3); }
};

struct Poly7Accel {
    __device__ double* operator()(double pos[3], double vel[3], double t)
    {
        auto res = new double[3];
        double val = -3 * 2 * t + 7 * 6 * t * t * t * t * t; // 42*t^5 - 6t
        res[0] = val;
        res[1] = val;
        res[2] = val;
        return res;
    }
};

struct Exact8Position {
    double operator()(double t)
    {
        return pow(t, 8) - pow(t, 7) + 2 * pow(t, 6) - 5 * pow(t, 5) - 10 * pow(t, 4) + pow(t, 3)
            - pow(t, 2);
    }
};

struct Poly8Accel {
    __device__ double* operator()(double pos[3], double vel[3], double t)
    {
        auto res = new double[3];
        double val = 56 * pow(t, 6) - 42 * pow(t, 5) + 60 * pow(t, 4) - 100 * pow(t, 3)
            - 120 * pow(t, 2) + 6 * t - 2;

        res[0] = val;
        res[1] = val;
        res[2] = val;
        return res;
    }
};

struct Exact9Position {
    double operator()(double t)
    {
        return pow(t, 9) + pow(t, 8) - pow(t, 7) + 2 * pow(t, 6) - 5 * pow(t, 5) - 10 * pow(t, 4)
            + pow(t, 3) - pow(t, 2);
    }
};

struct Poly9Accel {
    __device__ double* operator()(double pos[3], double vel[3], double t)
    {
        auto res = new double[3];
        double val = 72 * pow(t, 7) + 56 * pow(t, 6) - 42 * pow(t, 5) + 60 * pow(t, 4)
            - 100 * pow(t, 3) - 120 * pow(t, 2) + 6 * t - 2;

        res[0] = val;
        res[1] = val;
        res[2] = val;
        return res;
    }
};

inline double l2_error(const double* actual, const double* expected, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = actual[i] - expected[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

struct ExactExponential {
    double operator()(double t) { return exp(-t) + t - 1; }
};

struct ExponentialAccel {
    __device__ double* operator()(double pos[3], double vel[3], double t)
    {
        auto res = new double[3];
        double val = exp(-t);

        res[0] = val;
        res[1] = val;
        res[2] = val;
        return res;
    }
};

struct DampedOscillatorAccel {
    __device__ double* operator()(double pos[3], double vel[3], double t)
    {
        auto res = new double[3];
        for (int i = 0; i < 3; ++i) {
            res[i] = -2 * pos[i] - 2 * vel[i];
        }
        return res;
    }
};

struct DampedOscillatorExact {
    double operator()(double t) { return exp(-t) * (sin(t) + cos(t)); }
};
