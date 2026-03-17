#pragma once
#include "force_model/types.cuh"
#include <string>

namespace DatasetLoaders {
__host__ LeapSecondDataset LoadLeapSeconds(const std::string& filepath);
__host__ NutationTables LoadNutationData(const std::string& filepath);
__host__ EOPDataset LoadEOPData(const std::string& filepath);
__host__ GravityDataset LoadGravityModel(const std::string& filepath);
}