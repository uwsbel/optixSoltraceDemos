
# optixSoltraceDemos

**optixSoltraceDemos** is a demo project that utilizes NVIDIA OptiX for ray tracing encountered in Concentrated Solar Power applications.

---

## Prerequisites

Before building `optixSoltraceDemos`, ensure you have the following:

1. **NVIDIA GPU and Drivers**
   - An NVIDIA GPU.
   - Minimum CUDA Toolkit version: 11.0.
   - NVIDIA drivers compatible with your GPU and CUDA version.

2. **NVIDIA OptiX SDK**
   - Download the NVIDIA OptiX SDK from [NVIDIA's website](https://developer.nvidia.com/designworks/optix/download).
   - Install the SDK to a directory. For example:
     ```
     C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/
     ```
   - The source code for the OptiX SDK is located in the `SDK` directory, for example, `C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/SDK`. Create a `build` folder under the `Optix SDK 8.1.0` directory:
     ```
     C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/build
     ```
   - Configure and build the SDK in the created `build` directory, in both `Release` and `Debug` modes
   - To confirm that the `Optix SDK` is built successfully, run the `optixHello` demo in `build/bin`.

3. **CMake**
   - Version 3.18 or higher.

4. **C++ Compiler**
   - A compiler that supports C++17 or later.

---

## Building `optixSoltraceDemos`

### Configure the Build

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:uwsbel/optixSoltraceDemos.git
   cd optixSoltraceDemos
   ```

2. **Set the `OptiX_INSTALL_DIR`**:
   - Provide the path to the OptiX SDK during CMake configuration.
   - For example, if the SDK is installed in `C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/`:
     ```bash
     cmake -DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/" .
     ```

3. **(Optional) Set the `OptiX_INSTALL_DIR` as an Environment Variable**:
   - Alternatively, you can set `OptiX_INSTALL_DIR` as an environment variable:
     - On **Windows**:
       ```powershell
       $env:OptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/"
       ```
     - On **Linux/macOS**:
       ```bash
       export OptiX_INSTALL_DIR="/path/to/OptiX SDK/"
       ```

4. **Generate Build Files**:  Run CMake to generate build files for your project

5. **Build the Project**:  Build the optixSoltraceDemos project
   
---

## Running the Demos

Once built, the executable demos can be found in the `bin` directory. Run them from the command line or within your IDE.
