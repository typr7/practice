cmake_minimum_required(VERSION 3.22)
project(cuda_practice LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

include(cmake/dependencies.cmake)

add_subdirectory(src/01_vector_add)
add_subdirectory(src/02_rgb_to_gray)
add_subdirectory(src/03_image_blur)
add_subdirectory(src/04_matrix_multiplication)