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

#include "force_model/constants.cuh"
#include "force_model/utility/dataset_loaders.cuh"
#include "force_model/utility/ref_sys_trans.cuh"
#include "gpu_utils/gpu_utils.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace DatasetLoaders {

// A single function to load either the Longitude or Obliquity nutation tables
NutationTables LoadNutationData(const std::string& filepath)
{
    NutationTables tables { };
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << "\n";
        return tables;
    }

    std::string line;
    int j = 0;

    std::vector<NutationTerm> j0;
    std::vector<NutationTerm> j1;

    while (std::getline(file, line)) {
        if (line.find("j = 0") != std::string::npos) {
            j = 0;
            continue;
        } else if (line.find("j = 1") != std::string::npos) {
            j = 1;
            continue;
        }

        std::istringstream iss(line);
        int i;

        if (iss >> i) {
            NutationTerm term;

            if (iss >> term.sine_coeff_uas >> term.cosine_coeff_uas) {
                bool validRow = true;

                for (int k = 0; k < 14; ++k) {
                    if (!(iss >> term.args[k])) {
                        validRow = false;
                        break;
                    }
                }

                if (validRow) {
                    if (j == 0) {
                        j0.push_back(term);
                    } else if (j == 1) {
                        j1.push_back(term);
                    }
                } else {
                    std::cerr << "Warning: Malformed line in " << filepath << ": " << line << "\n";
                }
            }
        }
    }

    tables.j0 = new NutationTerm[j0.size()];
    tables.j1 = new NutationTerm[j1.size()];
    std::memcpy(tables.j0, j0.data(), j0.size() * sizeof(NutationTerm));
    std::memcpy(tables.j1, j1.data(), j1.size() * sizeof(NutationTerm));
    tables.j0_size = j0.size();
    tables.j1_size = j1.size();
    return tables;
}

EOPDataset LoadEOPData(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::string line;
    size_t valid_line_count = 0;

    // First pass: count valid data lines so we know exactly how large to make our array
    const char* whitespace = " \t\n\r\f\v";
    while (std::getline(file, line)) {
        size_t endpos = line.find_last_not_of(whitespace);
        if (endpos == std::string::npos)
            line.clear();
        else
            line.erase(endpos + 1);

        // Skip empty lines, comments, and lines too short to contain UT1-UTC (column 68)
        if (line.empty() || line[0] == '#' || line.length() < 68) {
            continue;
        }
        valid_line_count++;
    }

    auto records = new EOPRecord[valid_line_count];

    // Second pass: extract the data using fixed-width substring parsing
    file.clear();
    file.seekg(0);

    size_t i = 0;
    while (std::getline(file, line)) {
        size_t endpos = line.find_last_not_of(whitespace);
        if (endpos == std::string::npos)
            line.clear();
        else
            line.erase(endpos + 1);

        if (line.empty() || line[0] == '#' || line.length() < 68) {
            continue;
        }

        try {
            // Extract MJD_UTC (Cols 8-15 -> 0-indexed 7, length 8)
            records[i].mjd_utc = std::stod(line.substr(7, 8));

            // Extract Bull. A PM-x (Cols 19-27 -> 0-indexed 18, length 9)
            records[i].xp = std::stod(line.substr(18, 9)) / CFMConst::ArcsPerRad;

            // Extract Bull. A PM-y (Cols 38-46 -> 0-indexed 37, length 9)
            records[i].yp = std::stod(line.substr(37, 9)) / CFMConst::ArcsPerRad;

            // Extract Bull. A UT1-UTC (Cols 59-68 -> 0-indexed 58, length 10)
            records[i].ut1_utc = std::stod(line.substr(58, 10));

            i++;
        } catch (const std::exception& e) {
            // If a line throws (e.g., missing data causing std::stod to fail), safely ignore it
            continue;
        }
    }

    return { .records = records, .size = i };
}

__host__ LeapSecondDataset LoadLeapSeconds(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::string line;
    size_t valid_line_count = 0;

    // First pass: Count valid data lines to allocate exact array sizes
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty())
            continue;

        // Find the first non-whitespace character
        size_t first_char = line.find_first_not_of(" \t");

        // If the line is just whitespace or starts with a comment, skip it
        if (first_char == std::string::npos || line[first_char] == '#') {
            continue;
        }

        valid_line_count++;
    }

    // Allocate arrays in the struct
    LeapSecondDataset dataset;
    dataset.size = valid_line_count;
    dataset.mjd_utc = new double[valid_line_count];
    dataset.leap_seconds = new double[valid_line_count];

    // Second pass: Extract the data
    file.clear();
    file.seekg(0);

    size_t i = 0;
    while (std::getline(file, line) && i < valid_line_count) {
        size_t first_char = line.find_first_not_of(" \t");
        if (first_char == std::string::npos || line[first_char] == '#') {
            continue;
        }

        std::stringstream ss(line);
        long long ntp_time;
        double dtai;

        // Extract the two numbers; stringstream automatically handles whitespace
        if (ss >> ntp_time >> dtai) {
            dataset.mjd_utc[i] = RefSysTrans::NTPToMJD(ntp_time);
            dataset.leap_seconds[i] = dtai;
            i++;
        }
    }

    return dataset;
}

// Helper to convert ICGEM date formats ("YYYYMMDD.xxxx" or "YYYYMMDD") to decimal years
static double ParseIcgemDate(const std::string& str)
{
    if (str.length() < 8)
        return 0.0;
    double year = std::stod(str.substr(0, 4));
    int month = std::stoi(str.substr(4, 2));
    double day = std::stod(str.substr(6, 2));
    size_t dot_pos = str.find('.');
    if (dot_pos != std::string::npos && dot_pos + 1 < str.length()) {
        day += std::stod("0." + str.substr(dot_pos + 1));
    }
    static const int days_before_month[]
        = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };
    int m_idx = (month >= 1 && month <= 12) ? (month - 1) : 0;
    double day_of_year = days_before_month[m_idx] + (day - 1.0);
    return year + (day_of_year / 365.25);
}

__host__ GravityDataset LoadGravityModel(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open gravity model file: " + filepath);
    }

    GravityDataset dataset;
    std::string line;
    bool endOfHeadFound = false;

    // 1. Parse Header
    while (std::getline(file, line)) {
        if (line.find("end_of_head") != std::string::npos) {
            endOfHeadFound = true;
            break;
        }

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "earth_gravity_constant") {
            iss >> dataset.GM;
        } else if (token == "radius") {
            iss >> dataset.R;
        } else if (token == "max_degree") {
            unsigned max_degree;
            iss >> max_degree;
            dataset.N = max_degree + 1; // N is now the total number of rows
        }
    }

    if (!endOfHeadFound) {
        throw std::runtime_error("Invalid ICGEM format: 'end_of_head' not found.");
    }

    // 2. Allocate Arrays
    size_t totalElements = TRI_SIZE(dataset.N);

    dataset.Cnm = new double[totalElements] { 0.0 };
    dataset.Snm = new double[totalElements] { 0.0 };
    dataset.is_gfct = new bool[totalElements] { false };
    dataset.trnd_C = new double[totalElements] { 0.0 };
    dataset.acos1_C = new double[totalElements] { 0.0 };
    dataset.asin1_C = new double[totalElements] { 0.0 };
    dataset.acos05_C = new double[totalElements] { 0.0 };
    dataset.asin05_C = new double[totalElements] { 0.0 };
    dataset.trnd_S = new double[totalElements] { 0.0 };
    dataset.acos1_S = new double[totalElements] { 0.0 };
    dataset.asin1_S = new double[totalElements] { 0.0 };
    dataset.acos05_S = new double[totalElements] { 0.0 };
    dataset.asin05_S = new double[totalElements] { 0.0 };
    dataset.t0 = new double[totalElements] { 0.0 };

    // 3. Parse Data Lines
    bool firstDataLine = true;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::istringstream iss(line);
        std::string key;
        if (!(iss >> key))
            continue;

        // Check the first data line rule
        if (firstDataLine) {
            if (key != "gfc" && key != "gfct") {
                throw std::runtime_error(
                    "Error: Only static models (gfc) are accepted. Found: " + key);
            }
            firstDataLine = false;
        }

        // Parse coefficients
        if (key != "gfc" && key != "gfct" && key != "trnd" && key != "acos" && key != "asin")
            continue;

        unsigned L, M;
        double C_val, S_val;

        // Extract L, M, C, S
        if (!(iss >> L >> M >> C_val >> S_val))
            continue;

        // Bounds check: L must be less than N (since N is max_degree + 1)
        if (L >= dataset.N || M > L)
            continue;

        size_t idx = TRI_C2IDX(L, M);

        if (key == "gfc") {
            dataset.Cnm[idx] = C_val;
            dataset.Snm[idx] = S_val;
            dataset.is_gfct[idx] = false;

        } else if (key == "gfct") {
            dataset.Cnm[idx] = C_val;
            dataset.Snm[idx] = S_val;
            dataset.is_gfct[idx] = true;
            dataset.trnd_C[idx] = 0.0;
            dataset.acos1_C[idx] = 0.0;
            dataset.asin1_C[idx] = 0.0;
            dataset.acos05_C[idx] = 0.0;
            dataset.asin05_C[idx] = 0.0;
            dataset.trnd_S[idx] = 0.0;
            dataset.acos1_S[idx] = 0.0;
            dataset.asin1_S[idx] = 0.0;
            dataset.acos05_S[idx] = 0.0;
            dataset.asin05_S[idx] = 0.0;

            double sigC, sigS;
            std::string t0_str;
            if (iss >> sigC >> sigS >> t0_str) {
                dataset.t0[idx] = ParseIcgemDate(t0_str);
            }

        } else if (key == "trnd") {
            dataset.trnd_C[idx] = C_val;
            dataset.trnd_S[idx] = S_val;

        } else if (key == "acos") {
            if (line.find(" 1.0") != std::string::npos) {
                dataset.acos1_C[idx] = C_val;
                dataset.acos1_S[idx] = S_val;
            } else if (line.find(" 0.5") != std::string::npos) {
                dataset.acos05_C[idx] = C_val;
                dataset.acos05_S[idx] = S_val;
            }

        } else if (key == "asin") {
            if (line.find(" 1.0") != std::string::npos) {
                dataset.asin1_C[idx] = C_val;
                dataset.asin1_S[idx] = S_val;
            } else if (line.find(" 0.5") != std::string::npos) {
                dataset.asin05_C[idx] = C_val;
                dataset.asin05_S[idx] = S_val;
            }
        }
    }

    return dataset;
}
}