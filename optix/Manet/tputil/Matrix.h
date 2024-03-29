#pragma once

#include <vector_types.h>

#include "Preprocessor.h"
#include "VectorMath.h"

namespace tputil
{

template <size_t dim> struct VectorDim { };
template <> struct VectorDim<1> { using VectorType = float;  };
template <> struct VectorDim<2> { using VectorType = float2; };
template <> struct VectorDim<3> { using VectorType = float3; };
template <> struct VectorDim<4> { using VectorType = float4; };

template <size_t rows, size_t cols>
class Matrix
{
public:
    using FloatRows   = typename VectorDim<rows>::VectorType;
    using FloatCols   = typename VectorDim<cols>::VectorType;
    using ElementType = float;

public:
    TPUTIL_HOSTDEVICE Matrix()           = default;
    TPUTIL_HOSTDEVICE ~Matrix() noexcept = default;

    TPUTIL_HOSTDEVICE Matrix(const Matrix& other);
    TPUTIL_HOSTDEVICE Matrix& operator=(const Matrix& other);

    TPUTIL_HOSTDEVICE Matrix(const float (&arr)[rows * cols]);

    TPUTIL_HOSTDEVICE float operator[](size_t index) const noexcept { return m_data[index]; }
    TPUTIL_HOSTDEVICE float& operator[](size_t index) noexcept { return m_data[index]; }
    TPUTIL_HOSTDEVICE float operator()(size_t row_index, size_t col_index) const noexcept { return m_data[row_index * cols + col_index]; }
    TPUTIL_HOSTDEVICE float& operator()(size_t row_index, size_t col_index) noexcept { return m_data[row_index * cols + col_index]; }

    TPUTIL_HOSTDEVICE static constexpr size_t size() noexcept { return s_size; }

private:
    static constexpr size_t s_size = rows * cols;

    float m_data[s_size];
};

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>::Matrix(const Matrix& other)
{
    for (size_t i = 0; i < s_size; i++) {
        m_data[i] = other.m_data[i];
    }
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>& Matrix<rows, cols>::operator=(const Matrix& other)
{
    for (size_t i = 0; i < s_size; i++) {
        m_data[i] = other.m_data[i];
    }

    return *this;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>::Matrix(const float (&arr)[rows * cols])
{
    for (size_t i = 0; i < s_size; i++) {
        m_data[i] = arr[i];
    }
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> operator+(const Matrix<rows, cols>& a, const Matrix<rows, cols>& b)
{
    Matrix<rows, cols> ret;
    for(size_t i = 0; i < ret.size(); i++) {
        ret[i] = a[i] + b[i];
    }

    return ret;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> operator+(const Matrix<rows, cols>& a, float f)
{
    Matrix<rows, cols> ret;
    for(size_t i = 0; i < ret.size(); i++) {
        ret[i] = a[i] + f;
    }

    return ret;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>& operator+=(Matrix<rows, cols>& a, const Matrix<rows, cols>& b)
{
    for(size_t i = 0; i < a.size(); i++) {
        a[i] += b[i];
    }

    return a;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>& operator+=(Matrix<rows, cols>& a, float f)
{
    for(size_t i = 0; i < a.size(); i++) {
        a[i] += f;
    }

    return a;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> operator-(const Matrix<rows, cols>& a)
{
    Matrix<rows, cols> ret;
    for(size_t i = 0; i < ret.size(); i++) {
        ret[i] = -a[i];
    }

    return ret;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> operator-(const Matrix<rows, cols>& a, const Matrix<rows, cols>& b)
{
    Matrix<rows, cols> ret;
    for(size_t i = 0; i < ret.size(); i++) {
        ret[i] = a[i] - b[i];
    }

    return ret;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> operator-(const Matrix<rows, cols>& a, float f)
{
    Matrix<rows, cols> ret;
    for(size_t i = 0; i < ret.size(); i++) {
        ret[i] = a[i] - f;
    }

    return ret;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>& operator-=(Matrix<rows, cols>& a, const Matrix<rows, cols>& b)
{
    for(size_t i = 0; i < a.size(); i++) {
        a[i] -= b[i];
    }

    return a;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>& operator-=(Matrix<rows, cols>& a, float f)
{
    for(size_t i = 0; i < a.size(); i++) {
        a[i] -= f;
    }

    return a;
}

template <size_t left_rows, size_t left_cols, size_t right_cols>
TPUTIL_DECL Matrix<left_rows, right_cols>
operator*(const Matrix<left_rows, left_cols>& a, const Matrix<left_cols, right_cols>& b)
{
    Matrix<left_rows, right_cols> ret;
    for(size_t i = 0; i < left_rows; i++) {
        for(size_t j = 0; j < right_cols; j++) {
            float tmp = 0.0f;
            for(size_t k = 0; k < left_cols; k++) {
                tmp += a(i, k) * b(k, j);
            }
            ret(i, j) = tmp;
        }
    }

    return ret;
}

TPUTIL_DECL float4 operator*(const Matrix<4, 4>& a, const float4& v)
{
    Matrix<4, 1> tmp = {{ v.x, v.y, v.z, v.w }};
    Matrix<4, 1> res = a * tmp;
    return make_float4(res[0], res[1], res[2], res[3]);
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> operator*(const Matrix<rows, cols>& a, float f)
{
    Matrix<rows, cols> ret;
    for(size_t i = 0; i < ret.size(); i++) {
        ret[i] = a[i] * f;
    }

    return ret;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> operator*(float f, const Matrix<rows, cols>& a)
{
    return a * f;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols>& operator*=(Matrix<rows, cols>& a, float f)
{
    for(size_t i = 0; i < a.size(); i++) {
        a[i] *= f;
    }
    
    return a;
}

template <size_t rows, size_t cols>
TPUTIL_DECL Matrix<rows, cols> elementWiseMultiply(const Matrix<rows, cols>& a, const Matrix<rows, cols>& b)
{
    Matrix<rows, cols> ret;
    for(size_t i = 0; i < ret.size(); i++) {
        ret[i] = a[i] * b[i];
    }

    return ret;
}

}  // namespace tputil