#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <glad/glad.h>

#include <CudaDeviceBuffer.h>
#include <CudaOutputBuffer.h>
#include <Exception.h>
#include <GlDisplay.h>
#include <TpUtil.h>
#include <Matrix.h>
#include <Model.h>
#include <VectorMath.h>
#include <Windows.h>
#include <Transform.h>
#include <cuda_runtime.h>
#include <glfw/glfw3.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <vector_functions.h>
#include <vector_types.h>

#define THROW_EXCEPTION(msg) throw std::exception(std::format("[{}, {}]: {}", __FILE__, __LINE__, msg).c_str())

#define LOG_INFO(msg) std::cerr << msg << "\n"
#define LOG_DEBUG(msg)                       \
    std::cerr << "\033[1;33m" << msg << "\n" \
              << "\033[0m"
#define LOG_ERROR(msg)                       \
    std::cerr << "\033[0;31m" << msg << "\n" \
              << "\033[0m"

#define TO_STRING(x) #x
#define SAMPLE_NAME_STRING(name) TO_STRING(name)
#define SAMPLE_NAME SAMPLE_NAME_STRING(OPTIX_SAMPLE_NAME_DEFINE)

inline void
ensureMinimumSize(int32_t& width, int32_t& height)
{
    if (width <= 0)
        width = 1;
    if (height <= 0)
        height = 1;
}