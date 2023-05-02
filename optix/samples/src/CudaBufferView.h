#pragma once

#include <cstdint>

#include <cuda.h>
#include <host_defines.h>

#include "ManetMacros.h"

template <typename ElementType>
struct CudaBufferView
{
    CUdeviceptr buffer_ptr        = 0;
    uint32_t    element_count     = 0;
    uint32_t    stride_byte_size  = 0;

    MANET_DECL bool isValid() const { return (buffer_ptr != 0); }
    MANET_DECL operator bool() const { return isValid(); }
    template <typename T = ElementType>
    __forceinline__ __device__ T& at(size_t index) const
    {
        return *reinterpret_cast<T*>(buffer_ptr + (stride_byte_size != 0 ? stride_byte_size : sizeof(T)) * index);
    }
};

struct TriangleIndexType {};

template <>
struct CudaBufferView<TriangleIndexType>
{
    enum TriangleIndexFormat
    {
        TRIANGLE_INDEX_FORMAT_USHORT3 = 6,
        TRIANGLE_INDEX_FORMAT_UINT3   = 12
    };

    CUdeviceptr         buffer_ptr       = 0;
    uint32_t            element_count    = 0;
    uint32_t            stride_byte_size = 0;
    TriangleIndexFormat index_format     = TRIANGLE_INDEX_FORMAT_UINT3;

    MANET_DECL bool isValid() const { return (buffer_ptr != 0); }
    MANET_DECL operator bool() const { return isValid(); }
    template <typename T>
    __forceinline__ __device__ T& at(size_t index) const
    {
        return *reinterpret_cast<T*>(buffer_ptr 
                                     + (stride_byte_size != 0 ? stride_byte_size : sizeof(T))
                                     * index);
    }
};

using CudaTriangleIndexBufferView = CudaBufferView<TriangleIndexType>;