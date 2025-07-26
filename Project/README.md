# 3D Shape Model Construction (3DSMC) Project

This project implements a 3D face reconstruction pipeline using the FLAME model for shape modeling and optimization. The project consists of multiple executables that work together to process depth data, perform nearest neighbor searches, and optimize 3D face shapes.


We use the docker env provided in the exercises.

## Project Structure

```
Project/
├── CMakeLists.txt          # Main build configuration
├── Eigen.h                 # Eigen library header
├── cnpy/                   # NumPy file I/O library
├── Data/                   # Input/output data directory
│   ├── betas/             # Shape parameters
│   ├── optimize_test/     # Test data for optimization
│   └── 3d/               # 3D mesh data
├── model/                 # FLAME model files
│   ├── FLAME2020/        # FLAME 2020 model
│   ├── FLAME2023/        # FLAME 2023 model
│   ├── mesh/             # Generated meshes
│   └── mediapipe_landmark_embedding/ # Landmark data
├── Lift_depth/           # Depth processing utilities
├── knn/                  # K-Nearest Neighbor search
├── optimizer/            # Shape optimization algorithms
├── RT/                   # Real-time processing (commented out)
├── RigidAlignment/       # Rigid alignment utilities
├── dataset/              # Dataset processing
├── build/                # Build output directory
└── out/                  # Output directory
```

## Dependencies

- **CMake** (3.10+)
- **Eigen3** - Linear algebra library
- **OpenCV** - Computer vision library
- **Ceres Solver** - Nonlinear optimization
- **OpenMP** - Parallel processing
- **ZLIB** - Compression library
- **CNPY** - NumPy file I/O (included)

## Building the Project

```bash
# Navigate to project directory
cd 3dsmc_project/Project

# Create and enter build directory
mkdir build && cd build

# Configure and build
cmake ..
make
```

## Executables

The following executables are generated in order of their typical usage in the pipeline:

### 1. `lift_depth`
**Location**: `Lift_depth/Lift_depth.cpp`
**Purpose**: Processes depth data and camera parameters to generate 3D point clouds
- Reads depth images and camera calibration files
- Converts depth data to 3D coordinates
- Exports results as COFF mesh files
- Handles camera intrinsics and extrinsics

### 2. `rt` (RT)
**Location**: `RigidAliment/rt.cpp`
**Purpose**: Real-time processing of depth and color data with landmark extraction
- change std::string frame to your desired framenumber for example std::string frame = "00001"
- Processes depth and color images simultaneously
- Extracts 3D landmarks from depth data using 2D MediaPipe landmarks
- Performs coordinate transformations between depth and color cameras
- Exports processed mesh and 3D landmarks
- **Note**: Currently commented out in CMakeLists.txt - uncomment to build

### 3. `optimize_plane`
**Location**: `optimizer/optimize_plane.cpp`
**Purpose**: Advanced optimization with point-to-plane constraints
- Implements both point-to-point and point-to-plane distances
- Calculates surface normals for plane constraints
- Uses weighted optimization for better convergence
- Maximum 7 iterations by default
- **Configuration**: Adjust `file_number` variable for specific data, it has to match with the std::string frame you set before

### 4. `read_flame`
**Location**: `optimizer/read_flame.cpp`
**Purpose**: Reads and visualizes FLAME model results
- Loads FLAME model from NPZ files
- Generates random or specific face shapes
- Exports optimized meshes as OBJ files
- Supports both FLAME2020 and FLAME2023 models
- **Configuration**: Adjust `file_number` variable for specific face data
- output path: project/model/mesh/ <frame>


## Usage Pipeline

1. **Data Preparation**: Place input depth data, color images, and camera parameters in appropriate directories
2. **Depth Processing**: Run `lift_depth` to convert depth to 3D point clouds
3. **Real-time Processing**: Run `rt`  to process depth/color data and extract 3D landmarks
4. **Shape Optimization**: Run `optimize_plane` to perform advanced shape optimization with plane constraints
5. **Result Visualization**: Run `read_flame` to generate and export final meshes



## Configuration

### File Number Configuration
Several executables require adjusting the `file_number` variable:
- In `optimize_plane.cpp`: Set `file_number` to match your input data
- In `read_flame.cpp`: Set `file_number` to load specific optimized results

### Model Paths
- FLAME model files are expected in `model/FLAME2023/` or `model/FLAME2020/`
- Input data should be placed in `Data/` subdirectories
- Output files are generated in the respective directories

## Notes

- **RT Directory**: Contains real-time processing utilities. The `main` executable is currently commented out in CMakeLists.txt - uncomment lines 28-30 to build it
- **Parallel Processing**: KNN and optimization algorithms use OpenMP for parallel execution
- **Memory Usage**: Large datasets may require significant memory for KNN operations
- **File Formats**: Supports OFF/COFF, OBJ, and NPZ file formats
- **Landmark Processing**: RT main executable extracts 3D landmarks from depth data using 2D MediaPipe landmarks

## Troubleshooting

- Ensure all dependencies are properly installed
- Check file paths in source code for your specific data structure
- Verify FLAME model files are present in the model directory
- Monitor memory usage for large point cloud processing 