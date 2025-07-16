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



file(GLOB CNPY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cnpy/*.cpp)



add_executable(lift_depth lift_depth.cpp)
target_include_directories(lift_depth PRIVATE
    ${EIGEN3_INCLUDE_DIRS}                 
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(lift_depth PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

add_executable(pixel_match pixel_match.cpp)
target_include_directories(pixel_match PRIVATE
    ${EIGEN3_INCLUDE_DIRS}                 
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(pixel_match PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
)
