cmake_minimum_required(VERSION 3.18)
project(cuda_practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_BUILD_TYPE "")

include(cmake/dependencies.cmake)

add_subdirectory(src/01_vector_add)
add_subdirectory(src/02_rgb_to_gray)