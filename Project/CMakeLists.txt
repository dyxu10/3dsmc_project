cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (will use Eigen3_DIR if set above)
find_package(Eigen3 REQUIRED)

# Source and header files
set(SOURCES main.cpp)

# Optional: only list public headers if needed
# If "Eigen.h" is your own file, include it. But not needed for Eigen itself.
# Remove if you're mistakenly listing Eigen's internal headers.
set(HEADERS )

# Create executable
add_executable(main ${SOURCES} ${HEADERS})


# Include Eigen headers
target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIRS})

# Link Eigen (not strictly required for a header-only library, but fine)
target_link_libraries(main PRIVATE Eigen3::Eigen)

find_package(OpenCV REQUIRED)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

# Add Rigid_alignment_RT executable
add_executable(Rigid_alignment_RT Rigid_alignment_RT.cpp)
target_include_directories(Rigid_alignment_RT PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(extract_vertices extract_vertices_under4.3.cpp)
target_link_libraries(extract_vertices Eigen3::Eigen)

add_executable(transform_off transform_off.cpp)
target_link_libraries(transform_off PRIVATE Eigen3::Eigen)

add_executable(to_off to_off.cpp)
target_include_directories(to_off PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(to_off PRIVATE Eigen3::Eigen)

add_executable(transform_pip_to_rgb transform_pip_to_rgb.cpp)
target_link_libraries(transform_pip_to_rgb PRIVATE Eigen3::Eigen)

add_executable(scale scale.cpp)
target_include_directories(scale PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(scale PRIVATE Eigen3::Eigen ${OpenCV_LIBS})