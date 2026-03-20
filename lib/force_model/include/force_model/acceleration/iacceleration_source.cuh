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

// A common interface for all acceleration sources
struct IAccelerationSource {
    virtual ~IAccelerationSource() = default;

    virtual void Initialize() = 0;

    // Computes the acceleration coming from a source and returns the vectors to the out array,
    // placed every outStride * 3 elements.
    //
    // For outStride = 1: out = { ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3, ... }
    // For outStride = 2: out = { ax1, ay1, az1, ___, ___, ___, ax2, ay2, az2, ___, ___, ___, ... }
    // where ___ are left as before
    virtual void ComputeAcceleration(
        double* devPosition, double* devVelocity, int stepNum, double* out, int outStride)
        = 0;
};