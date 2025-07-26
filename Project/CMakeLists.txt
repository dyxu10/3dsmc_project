cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Project_3DSMC)

find_package(OpenMP REQUIRED)

# Set C++ flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Find Eigen3 (will use Eigen3_DIR if set above)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# CNPY sources (optional, comment out if unused)
file(GLOB CNPY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cnpy/*.cpp)

add_library(cnpy STATIC ${CNPY_SOURCES})
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# RT executables
add_executable(rt RigidAlignment/rt.cpp)
target_include_directories(rt PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rt PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

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



# KNN executable
add_executable(knn1 knn/knn_1.cpp)
target_include_directories(knn1 PRIVATE ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(knn1 PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX cnpy ZLIB::ZLIB)

# KNN executable
add_executable(knn2 knn/knn_2.cpp)
target_include_directories(knn2 PRIVATE ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(knn2 PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX cnpy ZLIB::ZLIB)


add_executable(optimize optimizer/optimize.cpp)
target_include_directories(optimize PRIVATE ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(optimize PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX cnpy ZLIB::ZLIB ${CERES_LIBRARIES})


add_executable(read_flame optimizer/read_flame.cpp)
target_include_directories(read_flame PRIVATE ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(read_flame PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX cnpy ZLIB::ZLIB)


add_executable(optimize_face_only optimizer/optimize_face_only.cpp)
target_include_directories(optimize_face_only PRIVATE ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(optimize_face_only PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX cnpy ZLIB::ZLIB ${CERES_LIBRARIES})


add_executable(optimize_plane optimizer/optimize_plane.cpp)
target_include_directories(optimize_plane PRIVATE ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(optimize_plane PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX cnpy ZLIB::ZLIB ${CERES_LIBRARIES})

