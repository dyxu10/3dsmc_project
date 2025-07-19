cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Project_3DSMC)

find_package(OpenMP REQUIRED)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (will use Eigen3_DIR if set above)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)

# CNPY sources (optional, comment out if unused)
file(GLOB CNPY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cnpy/*.cpp)

add_library(cnpy STATIC ${CNPY_SOURCES})
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# RT executables
# add_executable(main RT/main.cpp)
# target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(main PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

# add_executable(Rigid_alignment_RT RT/Rigid_alignment_RT.cpp)
# target_include_directories(Rigid_alignment_RT PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(Rigid_alignment_RT PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

# add_executable(scale RT/scale.cpp)
# target_include_directories(scale PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(scale PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

# add_executable(transform_off RT/transform_off.cpp)
# target_link_libraries(transform_off PRIVATE Eigen3::Eigen)

# add_executable(to_off RT/to_off.cpp)
# target_include_directories(to_off PRIVATE ${EIGEN3_INCLUDE_DIRS})
# target_link_libraries(to_off PRIVATE Eigen3::Eigen)

# add_executable(transform_pip_to_rgb RT/transform_pip_to_rgb.cpp)
# target_link_libraries(transform_pip_to_rgb PRIVATE Eigen3::Eigen)

# Lift_depth executables
add_executable(lift_depth Lift_depth/Lift_depth.cpp)
target_include_directories(lift_depth PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lift_depth PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

add_executable(pixel_match Lift_depth/pixel_match.cpp)
target_include_directories(pixel_match PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pixel_match PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

# KNN executable
add_executable(knn knn/knn_merge.cpp)
target_include_directories(knn PRIVATE ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(knn PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX cnpy ZLIB::ZLIB)
