# KMeans Clustering Implementation (CPU & GPU)

This project provides implementations of the K-Means clustering algorithm, supporting both CPU and GPU (CUDA) execution. It includes optimized versions leveraging CUDA Thrust for enhanced performance on compatible NVIDIA GPUs. The project also features a WinAPI-based visualization tool to display clustering results in 3D.

## Table of Contents

1.  [Project Title & Description](#project-title--description)
2.  [Table of Contents](#table-of-contents)
3.  [Installation Instructions](#installation-instructions)
4.  [Usage Guide](#usage-guide)
    *   [Command Line Arguments](#command-line-arguments)
    *   [Running the Examples](#running-the-examples)
    *   [Visualization](#visualization)
    *   [Data Formats](#data-formats)


## Installation Instructions

### Prerequisites

*   **Visual Studio:**  A compatible version of Visual Studio (2017 or later recommended, as indicated by the `.sln` file) is required for building the project on Windows. Ensure that you have the necessary C++ development components installed.
*   **CUDA Toolkit (Optional, for GPU support):** If you plan to use the GPU implementations, you *must* install the NVIDIA CUDA Toolkit.  Download it from [here](https://developer.nvidia.com/cuda-downloads). Version 11.x or later is likely required.
* **Git LFS:** Some of the data files are quite large and stored using Git LFS. Make sure that you have Git LFS installed. You can download it from [here](https://git-lfs.com/)

### Building the Project

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/F1r3d3v/KMeansCUDA.git
    cd KMeansCUDA
    git lfs install
    git lfs pull
    ```

2.  **Build the Project:**

    *   Open the `KMeans.sln` file in Visual Studio and build the solution (Debug or Release).
    This will create the `KMeans.exe` executable in either the `build/Release` or `build/Debug` directory.

### Dependencies
* **GLM (OpenGL Mathematics):** This project extensively uses GLM for vector and matrix operations.  The GLM library is included directly in the `include/glm` directory, so there's no need to install it separately. GLM is a header-only library.
* **CUDA Runtime (Optional):** The GPU-accelerated versions depend on the CUDA runtime library (`cudart`). This is part of the NVIDIA CUDA Toolkit.
* **Thrust (Optional):** The `gpu1` implementation uses the Thrust template library, which is part of the CUDA Toolkit. No separate installation is required if you have the CUDA Toolkit.

## Usage Guide

### Command Line Arguments

The `KMeans.exe` executable accepts the following command-line arguments:

```
KMeans.exe <file_format> <mode> <input_file> <output_file>
```

*   **`<file_format>`:**  Specifies the format of the input data file.  Must be one of:
    *   `txt`:  Plain text file (see "Data Formats" below).
    *   `bin`: Binary file (see "Data Formats" below).

*   **`<mode>`:** Specifies the execution mode. Must be one of:
    *   `cpu`:  Runs the standard K-Means algorithm on the CPU.
    *   `gpu1`: Runs the K-Means algorithm on the GPU using Thrust.
    *   `gpu2`: Runs a custom CUDA kernel implementation of K-Means on the GPU.

*   **`<input_file>`:** The path to the input data file. The repository includes sample data files in the `data` directory.

*   **`<output_file>`:** The path to the output file where the results (centroids and assignments) will be written.

### Running the Examples

The `scripts` directory contains batch files that demonstrate how to run the program with different configurations:

*   **`run_cpu_txt.bat`:** Runs the CPU version with a text input file.
*   **`run_cpu_bin.bat`:** Runs the CPU version with a binary input file.
*   **`run_gpu_txt.bat`:** Runs the GPU (`gpu2`) version with a text input file.
*   **`run_gpu_bin.bat`:** Runs the GPU (`gpu2`) version with a binary input file.
*   **`run_thrust_txt.bat`:** Runs the Thrust-based GPU (`gpu1`) version with a text input file.
*   **`run_thrust_bin.bat`:** Runs the Thrust-based GPU (`gpu1`) version with a binary input file.
* **`run_visualization.bat`:** Runs the visualizer using the GPU.

To run an example, navigate to the `scripts` directory in a command prompt and execute the desired batch file.  For example:

```bash
cd scripts
run_cpu_txt.bat
```
Make sure `KMeans.exe` is available in the same directory as the scripts, or adjust the paths in the scripts accordingly.

Before running, make sure you have built the project (see "Installation Instructions").

### Visualization

The `run_visualization.bat` script is specifically set up to run the visualization. It uses the GPU (`gpu2`) version and the `points_5mln_3d_5c.txt` data file. When you run this script, it should create a window displaying a 3D visualization of the clustered data points. The window is created using the WinAPI. The visualization shows points and cluster centroids.  The program's rendering-related functions are found in `src/renderer.cu` and `include/renderer.h`.  The visualization constants are found in `include/config.h`.

### Data Formats

The program supports two input data formats:

*   **Text Format (`.txt`):**
    *   The first line should contain three integers separated by spaces: `n` (number of points), `dim` (dimensionality of each point), and `k` (number of clusters).
    *   Each subsequent line represents a data point, with `dim` floating-point values separated by spaces.

    Example (`data/points_5mln_3d_5c.txt`, but with fewer points for brevity):

    ```
    4 3 2
    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0
    10.0 11.0 12.0
    ```

*   **Binary Format (`.dat`):**
    *   The file starts with three integers (`n`, `dim`, `k`) stored directly as binary data (4 bytes each, in little-endian format).  These are *not* text representations.
    *   Following the header, the data points are stored as a contiguous block of `n * dim` floating-point values (4 bytes each, in little-endian format) in binary format, *not* as text.

    The binary format is generally more efficient for large datasets, as it avoids the overhead of text parsing.

The output file (`.txt`) will contain:
* The final centroid coordinates.
* A list of cluster assignments for each point.
