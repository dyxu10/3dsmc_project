cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Project_3DSMC)


# Set C++ flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (will use Eigen3_DIR if set above)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)

# Sources
set(SOURCES
    main.cpp
    read_flame.cpp
)

file(GLOB CNPY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cnpy/*.cpp)

# Create executable
add_executable(main main.cpp ${CNPY_SOURCES})

# Include directories
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cnpy        # correct path to cnpy.h
    ${CMAKE_CURRENT_SOURCE_DIR}             # for your own headers
    ${EIGEN3_INCLUDE_DIRS}                  # Eigen3
    ${OpenCV_INCLUDE_DIRS}
)


# Link libraries
target_link_libraries(main PRIVATE
    ZLIB::ZLIB
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

