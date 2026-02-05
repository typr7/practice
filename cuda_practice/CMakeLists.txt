cmake_minimum_required(VERSION 3.18)
project(rgbtogray LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)

set(BUILD_LIST core,imgproc,imgcodecs,highgui CACHE STRING "" FORCE)

FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.13.0
    OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(opencv)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs highgui)

add_executable(rgbtogray
    src/rgbtogray.cu
)

set_property(TARGET rgbtogray PROPERTY CUDA_ARCHITECTURES 86)

target_link_libraries(rgbtogray PRIVATE CUDA::cudart ${OpenCV_LIBS})