#pragma once

#include "backend.hpp"
#include "matrix.hpp"
#include "vector.hpp"

namespace hlcl {

// ---- 向量逐元素 (Hadamard) 积 ----
template<typename T, int Size, Backend B>
Vector<T, Size, B> hadamard_product(const Vector<T, Size, B>& a, const Vector<T, Size, B>& b) {
    assert(a.size() == b.size() && "hadamard_product requires equal sizes");
    Vector<T, Size, B> result;
    for (int i = 0; i < a.size(); ++i) result[i] = a[i] * b[i];
    return result;
}

// ---- 向量点积 (以 matTimesVec 命名, 供引擎标量投影使用) ----
template<typename T, typename U, int Size, Backend B>
T matTimesVec(const Vector<T, Size, B>& a, const Vector<U, Size, B>& b) {
    T result = T{0};
    for (int i = 0; i < a.size(); ++i) result += a[i] * static_cast<T>(b[i]);
    return result;
}

// ---- 矩阵乘向量: M (R x C) * v (C) -> (R) ----
template<typename T, int Rows, int Cols, Backend B, typename U, int VSize, Backend B2>
Vector<T, Rows, B> matTimesVec(const Matrix<T, Rows, Cols, B>& m,
                               const Vector<U, VSize, B2>& v) {
    static_assert(VSize == Cols, "matTimesVec: vector size must equal matrix columns");
    Vector<T, Rows, B> result;
    for (int i = 0; i < Rows; ++i) {
        T acc = T{0};
        for (int k = 0; k < Cols; ++k) acc += m(i, k) * static_cast<T>(v[k]);
        result[i] = acc;
    }
    return result;
}

// ---- 行向量乘矩阵: v^T (R) * M (R x C) -> (C)  (c_j = sum_k v_k M(k,j)) ----
template<typename U, int VSize, Backend B2, typename T, int Rows, int Cols, Backend B>
Vector<U, Cols, B2> vecTimesMat(const Vector<U, VSize, B2>& v,
                                const Matrix<T, Rows, Cols, B>& m) {
    static_assert(VSize == Rows, "vecTimesMat: vector size must equal matrix rows");
    Vector<U, Cols, B2> result;
    for (int j = 0; j < Cols; ++j) {
        U acc = U{0};
        for (int k = 0; k < Rows; ++k) acc += static_cast<U>(v[k] * static_cast<T>(m(k, j)));
        result[j] = acc;
    }
    return result;
}

// ---- 三维叉积 ----
template<typename T, Backend B>
Vector<T, 3, B> cross(const Vector<T, 3, B>& a, const Vector<T, 3, B>& b) {
    return Vector<T, 3, B>({a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0]});
}

} // namespace hlcl
