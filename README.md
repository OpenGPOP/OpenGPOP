# Setup

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

### Nsight debugger

In vscode, use `Ctrl+P` and enter `ext install NVIDIA.nsight-vscode-edition`

### In case vscode struggles to recognise symbols

Instead of using the standard Microsoft C/C++ Intelisense try using clangd. Install it using `sudo apt install clangd`
and add a vscode extension with the same name. Then, in `.vscode/settings.json` add the following lines:

```
    "clangd.path": "clangd",
    "C_Cpp.intelliSenseEngine": "disabled",
    "C_Cpp.autocomplete": "disabled",
    "C_Cpp.errorSquiggles": "disabled"
```

Next, do a clean CMake rebuild and restart VSCode.

## How to build & run

In the main project directory, run `cmake --preset build`.

You can use the vscode's *Run and Debug* menu to run targets in debug mode with breakpoint support. Alternatively, you
can run ninja targets directly.

## Ninja targets

In the `build/` directory, you can use `ninja <target>` to build the chosen target. There are several targets available.
Adding `.run` at the end of a target name automatically runs it after a successful build. To check for device memory
leaks, use a `.memcheck` target.

- `exec` / `exec.run`
- `unit_tests` / `unit_tests.run` / `unit_tests.memcheck`
- `accuracy_tests` / `accuracy_tests.run`

## Accuracy tests

accuracy_tests target tests gpu propagator using real world data. To perform the tests, test cases needs to be generated
using [generate_test_data.py](generate_test_data.py) script.
It fetches real-world, centimeter-level accuracy telemetry from ESA's Sentinel-1 mission and establishes a numerical
baseline using Orekit.

The script does the following:

1. Downloads historical Precise Orbit Ephemerides (`.EOF` files) from the Copernicus ecosystem.
2. Uses `astropy` to precisely convert the downloaded state vectors from Earth-Fixed (ITRS) to Inertial (GCRS).
3. Uses `orekit` (via JPype) with a 70x70 spherical harmonics gravity model and a fixed-step RK4 integrator to propagate
   the orbit and calculate a baseline numerical error.
4. Saves the results as a flat, space-separated text file (`data/state_vectors.txt`) optimized for fast `std::ifstream`
   parsing in C++.

### Requirements & Setup

Python dependencies (such as `orekit-jpype`, `astropy`, and `sentineleof`) are downloaded automatically to virtual
environment during project setup (`cmake --preset build`)

**Important Data Dependency:** Orekit requires physical models (gravity fields, JPL ephemerides, IERS Earth rotation
data) to run. You must download [orekit-data-main.zip](https://gitlab.orekit.org/orekit/orekit-data) and place it in the
same directory as the script. **Do not unzip this file**; JPype will mount it directly into the JVM.

### Output Format

Each line in the generated text file represents a single test case, formatted exactly as follows:
`x0 y0 z0 vx0 vy0 vz0 year month day hour minute second dt_seconds target_x target_y target_z baseline_error_meters`
