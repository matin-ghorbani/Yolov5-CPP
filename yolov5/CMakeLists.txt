cmake_minimum_required(VERSION 3.5)

project(Yolov8CPP VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR on)
set(CMAKE_CXX_STANDARD 17)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 11.8 REQUIRED)
set(CMAKE_CUDA_STANDARD 11.8)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


set(PROJECT_SOURCE
        main.cpp

        utils.h
        utils.cpp
)

add_executable(run ${PROJECT_SOURCE})
target_link_libraries(run ${OpenCV_LIBS})