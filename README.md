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
Adding `.run` at the end of a target name automatically runs it after a successful build. To check for device memory leaks, use a `.memcheck` target.

- `exec` / `exec.run`
- `unit_tests` / `unit_tests.run` / `unit_tests.memcheck`
- `accuracy_tests` / `accuracy_tests.run`


# OLD ENTRIES - TO BE VERIFIED

### Accuracy tests

accuracy_tests target tests the cpu and gpu propagators using real world data. To perform the tests, test cases needs to
be generated using [generate_test_data.py](generate_test_data.py) script. The script fetches data from
[space-track.org](https://www.space-track.org/) and writes test cases to data/state_vectors.txt. **The script needs .env
file with credentials to space-track**. Required fields are as follows: **SPACETRACK_USERNAME** and
**SPACETRACK_PASSWORD**.

## Python visualisation

After running the `cpu` target, you can visualise the output using python. The `cmake --preset build` command
automatically creates and sets up the virtual environment inside the `.venv` directory.

To run the visualisation simply use:

```
.venv/bin/python3 visualisation.py
```

If the `requirements.txt` file changes, run `rm -rf .venv && cmake --preset build` to reinstall the virtual environment.
