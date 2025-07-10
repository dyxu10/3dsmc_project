cmake_minimum_required(VERSION 3.10)
<<<<<<< Updated upstream
project(MyProject)
=======
project(3DSMC-project)
>>>>>>> Stashed changes

# ========================
# Options
# ========================
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(OpenCV_DIR ${LIBRARY_DIR}/opencv/install/lib/cmake/opencv4 CACHE PATH "Path to OpenCV cmake config")
<<<<<<< Updated upstream
=======
set(CNPY_DIR ${LIBRARY_DIR}/cnpy CACHE PATH "Path to cnpy folder")
>>>>>>> Stashed changes

# ========================
# C++ flags
# ========================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================
# Dependencies
# ========================
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres   REQUIRED)

<<<<<<< Updated upstream
# ========================
# Sources
# ========================
set(MAIN_SOURCES main.cpp)
set(LANDMARK_SOURCES main_landmarks.cpp main.cpp)
=======
# 添加 cnpy 库（必须告诉 CMake 编译它）
add_subdirectory(${CNPY_DIR} cnpy_build)

# ========================
# Sources
# ========================
# set(LOSS_SOURCES   Loss.cpp)
set(READ_FLAME_SOURCES read_flame.cpp)
>>>>>>> Stashed changes

# ========================
# Targets
# ========================
<<<<<<< Updated upstream
add_executable(main main.cpp)  # 保持原样
add_executable(main_landmarks main_landmarks.cpp utils.cpp)


# ========== Shared include dirs ==========
target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_include_directories(main_landmarks PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# ========== Shared linked libs ==========
target_link_libraries(main PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(main_landmarks PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
=======
# 注释掉旧的运行文件
# add_executable(main            main.cpp)
# add_executable(main_landmarks  main_landmarks.cpp main.cpp)
# add_executable(loss ${LOSS_SOURCES})

add_executable(read_flame ${READ_FLAME_SOURCES})

# ========== Include 路径 ==========
# target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
# target_include_directories(main_landmarks PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
# target_include_directories(loss PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CNPY_DIR})
target_include_directories(read_flame PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CNPY_DIR})


# ========== 链接库 ==========
# target_link_libraries(main PRIVATE Eigen3::Eigen ${OpenCV_LIBS} ${CERES_LIBRARIES})
# target_link_libraries(main_landmarks PRIVATE Eigen3::Eigen ${OpenCV_LIBS} ${CERES_LIBRARIES})
# target_link_libraries(loss PRIVATE Eigen3::Eigen ${OpenCV_LIBS} ${CERES_LIBRARIES} cnpy)
target_link_libraries(read_flame PRIVATE Eigen3::Eigen ${OpenCV_LIBS} ${CERES_LIBRARIES} cnpy)
>>>>>>> Stashed changes
