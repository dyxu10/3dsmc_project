cmake_minimum_required(VERSION 3.10)
project(3DSMC-project)

# ========================
# Options
# ========================
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(OpenCV_DIR ${LIBRARY_DIR}/opencv/install/lib/cmake/opencv4 CACHE PATH "Path to OpenCV cmake config")
set(CNPY_DIR ${LIBRARY_DIR}/cnpy CACHE PATH "Path to cnpy folder")

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

# 添加 cnpy 库
add_subdirectory(${CNPY_DIR} cnpy_build)

# ========================
# 设置想编译的目标名，只改这里就行
# 可选值: main, main_landmarks, loss, read_flame, lossTest
set(TARGET_NAME "read_flame")
# ========================

# 源码路径定义
set(SOURCE_main              main.cpp)
set(SOURCE_main_landmarks    main_landmarks.cpp main.cpp)
set(SOURCE_loss              Loss.cpp)
set(SOURCE_read_flame        read_flame.cpp)
set(SOURCE_lossTest          lossTest.cpp)

# 自动选择对应源文件
if(TARGET_NAME STREQUAL "main")
    set(SRC ${SOURCE_main})
elseif(TARGET_NAME STREQUAL "main_landmarks")
    set(SRC ${SOURCE_main_landmarks})
elseif(TARGET_NAME STREQUAL "loss")
    set(SRC ${SOURCE_loss})
elseif(TARGET_NAME STREQUAL "read_flame")
    set(SRC ${SOURCE_read_flame})
elseif(TARGET_NAME STREQUAL "lossTest")
    set(SRC ${SOURCE_lossTest})
else()
    message(FATAL_ERROR "Unknown TARGET_NAME: ${TARGET_NAME}")
endif()

# 添加可执行文件
add_executable(${TARGET_NAME} ${SRC})

# Include 路径
target_include_directories(${TARGET_NAME} PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${CNPY_DIR}
)

# 链接库
target_link_libraries(${TARGET_NAME} PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    cnpy
)