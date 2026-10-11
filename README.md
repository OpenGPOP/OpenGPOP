# Introduction
OpenGPOP - Open **G**PU-**P**arallelised **O**rbit **P**ropagator is a CUDA C++ terminal application that uses GPUs to propagate orbits of  thousands of objects in parallel. The app has two main features - orbit propagation up to a given time and finding the closest approach of objects in a given period.

The propagator uses the 8th-order Gauss-Jackson method as integrator and a force model based on the Earth's gravitational field modelled with spherical harmonics. The [whitepaper](doc/whitepaper.pdf) provides a detailed description of the implemented numerical methods, force model, conjunction checks, and parallelisation techniques.

# Setup
As of now, OpenGPOP only works on Linux machines with NVIDIA GPUs.

## Prerequisites

Before you start the setup, make sure to run

```
sudo apt -y update
sudo apt -y upgrade
```

and `sudo reboot` if changes were made.

### CMake, Ninja, GTest

```
sudo apt install cmake ninja-build libgtest-dev
```

### CUDA Toolkit

Follow instructions on https://developer.nvidia.com/cuda-downloads and `sudo reboot`.

After the reboot, check if `nvidia-smi` and `nvcc --version` agree on the CUDA version.

If `nvcc --version` failed, **DO NOT INSTALL ~~`nvidia-cuda-toolkit`~~**, but enter the following:

```
echo -e "export PATH=\"/usr/local/cuda-YOUR_VERSION/bin:$PATH\"\nexport LD_LIBRARY_PATH=\"/usr/local/cuda-YOUR_VERSION/lib64:$LD_LIBRARY_PATH\"" >> /home/$USER/.bashrc
```

replacing `YOUR_VERSION` with your CUDA version, eg `12.9`

Then, restart the terminal and try again.

### Nsight debugger (optional)

In vscode, use `Ctrl+P` and enter `ext install NVIDIA.nsight-vscode-edition`

### In case vscode struggles to recognise symbols (optional)

Instead of using the standard Microsoft C/C++ Intelisense try using clangd. Install it using `sudo apt install clangd`
and add a vscode extension with the same name. Then, in `.vscode/settings.json` add the following lines:

```
    "clangd.path": "clangd",
    "clangd.arguments": [
        "--query-driver=/usr/bin/g*,/usr/local/cuda-13.1/bin/nvcc",
        "--compile-commands-dir=build-debug"
    ],
    "C_Cpp.intelliSenseEngine": "disabled",
    "C_Cpp.autocomplete": "disabled",
    "C_Cpp.errorSquiggles": "disabled"
```
Make sure to first build the project using the debug preset for this to work.

## Building

To build the project, use the provided CMake presets. You can choose between a debug and a release build.

### Debug Build

```bash
cmake --preset debug
cmake --build build-debug
```

### Release Build

```bash
cmake --preset release
cmake --build build-release
```

## Usage

The application provides a command-line interface with two main modes of operation: orbit propagation and closest approach calculation.

### Run Orbit Propagation

To propagate orbits for a set of objects:

```bash
./build-release/exec/exec propagate --input <input_file.json> [--output <output_file.json>]
```

**Options:**

- `-i, --input`: Path to the input JSON file (Required).
- `-o, --output`: Path to the output JSON file. Defaults to `propagation_out.json`.
- `--gravity-model`: Path to the gravity model file. Defaults to `../../data/EIGEN-6S4_(v2).gfc`.
- `--nutation-lon`: Path to IAU2000 nutation longitude file. Defaults to `../../data/IAU2000_R06_NutLon.txt`.
- `--nutation-obl`: Path to IAU2000 nutation obliquity file. Defaults to `../../data/IAU2000_R06_NutObl.txt`.
- `--eop`: Path to Earth Orientation Parameters (EOP) file. Defaults to `../../data/IAU2000A_EOP.txt`.
- `--leap-seconds`: Path to leap seconds file. Defaults to `../../data/leap-seconds.list`.

### Calculate Closest Approach

To calculate closest approaches between objects:

```bash
./build-release/exec/exec closest-approach --input <input_file.json> [--output <output_file.json>]
```

**Options:**

- `-i, --input`: Path to the input JSON file (Required).
- `-o, --output`: Path to the output JSON file. Defaults to `closest_approach_out.json`.
- `--gravity-model`: Path to the gravity model file. Defaults to `../../data/EIGEN-6S4_(v2).gfc`.
- `--nutation-lon`: Path to IAU2000 nutation longitude file. Defaults to `../../data/IAU2000_R06_NutLon.txt`.
- `--nutation-obl`: Path to IAU2000 nutation obliquity file. Defaults to `../../data/IAU2000_R06_NutObl.txt`.
- `--eop`: Path to Earth Orientation Parameters (EOP) file. Defaults to `../../data/IAU2000A_EOP.txt`.
- `--leap-seconds`: Path to leap seconds file. Defaults to `../../data/leap-seconds.list`.

### Input/Output Format

The application uses JSON for input and output. The data structures are defined using `nlohmann/json`.

**Common Structures:**

- `TimeData`: Year, Month, Day, Hour, Min, Sec.

**Propagation Input:**

- `propagationStart`: Start time (`TimeData`).
- `propagationSeconds`: Duration of propagation in seconds.
- `positions`: List of initial positions.
- `velocities`: List of initial velocities.

**Closest Approach Input:**

- `propagationStart`: Start time (`TimeData`).
- `propagationSeconds`: Duration of propagation in seconds.
- `positions`: List of initial positions.
- `velocities`: List of initial velocities.
- `setIndices`: Indices of the set of objects to check against each other.
