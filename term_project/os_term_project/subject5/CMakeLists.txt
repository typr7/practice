cmake_minimum_required(VERSION 3.0)
project(banker_algorithm)

set(CMAKE_C_STANDARD 23)

file(GLOB SRC
    ${PROJECT_SOURCE_DIR}/src/*.c
    ${PROJECT_SOURCE_DIR}/src/*.h
)

add_executable(${PROJECT_NAME} ${SRC})

