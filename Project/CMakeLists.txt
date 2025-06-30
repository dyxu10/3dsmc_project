cmake_minimum_required(VERSION 3.10)
project(MyProject)

# ========================
# Options
# ========================
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(OpenCV_DIR ${LIBRARY_DIR}/opencv/install/lib/cmake/opencv4 CACHE PATH "Path to OpenCV cmake config")

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

# ========================
# Sources
# ========================
set(MAIN_SOURCES main.cpp)
set(LANDMARK_SOURCES main_landmarks.cpp main.cpp)

# ========================
# Targets
# ========================
add_executable(main main.cpp)  # 保持原样
add_executable(main_landmarks main_landmarks.cpp utils.cpp)


# ========== Shared include dirs ==========
target_include_directories(main PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_include_directories(main_landmarks PRIVATE ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# ========== Shared linked libs ==========
target_link_libraries(main PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(main_landmarks PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
