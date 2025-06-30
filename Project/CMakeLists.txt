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

# Add test_alignment executable
add_executable(test_alignment test_alignment.cpp)
target_include_directories(test_alignment PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_alignment PRIVATE Eigen3::Eigen ${OpenCV_LIBS})