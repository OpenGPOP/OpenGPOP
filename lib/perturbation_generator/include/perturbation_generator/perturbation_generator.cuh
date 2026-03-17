#ifndef PERTURBATION_GENERATOR_CUH
#define PERTURBATION_GENERATOR_CUH

#include "cublas_v2.h"
#include "gpu_utils/gpu_utils.cuh"
#include <cuda_runtime.h>
#include <tuple>

class PerturbationGenerator {
    double epochPositionHost[3];
    double epochVelocityHost[3];

public:
    PerturbationGenerator(double epochPosition[3], double epochVelocity[3])
    {
        for (int i = 0; i < 3; ++i) {
            this->epochPositionHost[i] = epochPosition[i];
            this->epochVelocityHost[i] = epochVelocity[i];
        }
    }

    ~PerturbationGenerator() { }

    // In the object pointers are not null, they will be added to the end of the perturbed
    // satellites. If they are null, the perturbed satellites will be generated without them.
    std::tuple<double*, double*> generatePerturbedSatellites(int numSatellites,
        double perturbationStrength, double* objectPosition = nullptr,
        double* objectVelocity = nullptr);
};

#endif // PERTURBATION_GENERATOR_CUH
