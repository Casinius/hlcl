#pragma once

#include "backend.hpp"
#include "vector.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

namespace hlcl {

// 固定大小矩阵 (行优先, 数据内联于 std::array, 零堆分配)。
// 仅支持 Rows, Cols >= 1 的编译期固定矩阵。
template<typename T, int Rows, int Cols, Backend B = Backend::CPU>
class Matrix {
public:
    using value_type = T;
    using size_type = int;
    static constexpr int kRows = Rows;
    static constexpr int kCols = Cols;
    static constexpr Backend kBackend = B;
    static_assert(Rows > 0 && Cols > 0, "Matrix requires Rows, Cols >= 1");

    Matrix() { store_.fill(T{0}); }

    // 大小构造 (固定大小矩阵忽略参数, 零填充)
    explicit Matrix(size_type /*n*/) { store_.fill(T{0}); }

    // 扁平初始化列表, 行优先填充
    Matrix(std::initializer_list<T> init) {
        store_.fill(T{0});
        assert(init.size() <= static_cast<std::size_t>(Rows * Cols) &&
               "too many initializer entries for Matrix");
        size_type i = 0;
        for (const T& v : init) store_[static_cast<std::size_t>(i++)] = v;
    }

    // 嵌套初始化列表: { {r0c0, ...}, {r1c0, ...}, ... }
    Matrix(std::initializer_list<std::initializer_list<T>> rows_init) {
        store_.fill(T{0});
        size_type r = 0;
        for (const auto& row : rows_init) {
            assert(r < Rows && "too many rows in Matrix initializer");
            size_type c = 0;
            for (const T& v : row) {
                assert(c < Cols && "too many columns in Matrix row");
                (*this)(r, c) = v;
                ++c;
            }
            ++r;
        }
    }

    // 从其它元素类型/后端转换拷贝
    template<typename U, Backend B2>
    Matrix(const Matrix<U, Rows, Cols, B2>& other) {
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) (*this)(i, j) = static_cast<T>(other(i, j));
    }

    // 从其它元素类型/后端转换赋值
    template<typename U, Backend B2>
    Matrix& operator=(const Matrix<U, Rows, Cols, B2>& other) {
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) (*this)(i, j) = static_cast<T>(other(i, j));
        return *this;
    }

    size_type rows() const { return Rows; }
    size_type cols() const { return Cols; }
    size_type size() const { return Rows * Cols; }

    T& at(size_type r, size_type c) {
        assert(r >= 0 && r < Rows && c >= 0 && c < Cols && "index out of range");
        return store_[static_cast<std::size_t>(r * Cols + c)];
    }
    const T& at(size_type r, size_type c) const {
        assert(r >= 0 && r < Rows && c >= 0 && c < Cols && "index out of range");
        return store_[static_cast<std::size_t>(r * Cols + c)];
    }

    T& operator()(size_type row, size_type col) { return at(row, col); }
    const T& operator()(size_type row, size_type col) const { return at(row, col); }

    // 连续行优先数据访问 (std::array 内联存储, 零拷贝)
    T* data() { return store_.data(); }
    const T* data() const { return store_.data(); }

    void setZero() { store_.fill(T{0}); }
    void setConstant(const T& value) { store_.fill(value); }

    void setIdentity() {
        setZero();
        const int n = std::min(Rows, Cols);
        for (int i = 0; i < n; ++i) (*this)(i, i) = T{1};
    }

    void setDiagonal(const std::initializer_list<T>& values) {
        const int n = std::min(std::min(Rows, Cols), static_cast<int>(values.size()));
        for (int i = 0; i < n; ++i) (*this)(i, i) = *(values.begin() + i);
    }

    // 用任意向量(或支持 [] / size 的类型)填充矩阵行/列
    template<typename VecLike>
    void setRow(size_type row, const VecLike& v) {
        for (size_type j = 0; j < Cols && j < static_cast<size_type>(v.size()); ++j) {
            (*this)(row, j) = static_cast<T>(v[j]);
        }
    }

    template<typename VecLike>
    void setCol(size_type col, const VecLike& v) {
        for (size_type i = 0; i < Rows && i < static_cast<size_type>(v.size()); ++i) {
            (*this)(i, col) = static_cast<T>(v[i]);
        }
    }

    void resize(size_type /*rows*/, size_type /*cols*/) { /* 固定大小: 无操作 */ }

    T trace() const {
        const int n = std::min(Rows, Cols);
        T result = T{0};
        for (int i = 0; i < n; ++i) result += (*this)(i, i);
        return result;
    }

    T squaredNorm() const {
        T result = T{0};
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) {
                const T v = (*this)(i, j);
                result += v * v;
            }
        return result;
    }

    T norm() const { return static_cast<T>(std::sqrt(squaredNorm())); }

    T sum() const {
        T result = T{0};
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result += (*this)(i, j);
        return result;
    }

    T maxCoeff() const {
        T result = (*this)(0, 0);
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result = std::max(result, (*this)(i, j));
        return result;
    }

    T minCoeff() const {
        T result = (*this)(0, 0);
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result = std::min(result, (*this)(i, j));
        return result;
    }

    // ---- 成员逐元素运算 (同构矩阵) ----
    Matrix operator+(const Matrix& other) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result(i, j) = (*this)(i, j) + other(i, j);
        return result;
    }
    Matrix operator-(const Matrix& other) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result(i, j) = (*this)(i, j) - other(i, j);
        return result;
    }

    // ---- 成员标量运算 ----
    Matrix operator+(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result(i, j) = (*this)(i, j) + scalar;
        return result;
    }
    Matrix operator-(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result(i, j) = (*this)(i, j) - scalar;
        return result;
    }
    Matrix operator*(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result(i, j) = (*this)(i, j) * scalar;
        return result;
    }
    Matrix operator/(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result(i, j) = (*this)(i, j) / scalar;
        return result;
    }

    // ---- 矩阵乘法: Rows x Cols * Cols x OtherCols -> Rows x OtherCols ----
    template<typename U, int OtherCols, Backend B2>
    Matrix<T, Rows, OtherCols, B> operator*(const Matrix<U, Cols, OtherCols, B2>& other) const {
        Matrix<T, Rows, OtherCols, B> result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < OtherCols; ++j) {
                T acc = T{0};
                for (int k = 0; k < Cols; ++k) acc += (*this)(i, k) * static_cast<T>(other(k, j));
                result(i, j) = acc;
            }
        }
        return result;
    }

private:
    std::array<T, static_cast<std::size_t>(Rows * Cols)> store_;
};

// ---- 矩阵标量前置运算 (后置形式为成员, 避免歧义) ----
template<typename T, int Rows, int Cols, Backend B>
Matrix<T, Rows, Cols, B> operator*(T scalar, const Matrix<T, Rows, Cols, B>& m) {
    Matrix<T, Rows, Cols, B> result(m);
    for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < Cols; ++j) result(i, j) = scalar * result(i, j);
    return result;
}

template<typename T, int Rows, int Cols, Backend B>
Matrix<T, Rows, Cols, B> operator+(T scalar, const Matrix<T, Rows, Cols, B>& m) {
    Matrix<T, Rows, Cols, B> result(m);
    for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < Cols; ++j) result(i, j) = scalar + result(i, j);
    return result;
}

template<typename T, int Rows, int Cols, Backend B>
Matrix<T, Rows, Cols, B> operator-(T scalar, const Matrix<T, Rows, Cols, B>& m) {
    Matrix<T, Rows, Cols, B> result(m);
    for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < Cols; ++j) result(i, j) = scalar - result(i, j);
    return result;
}

// 矩阵范数 (自由函数)
template<typename T, int Rows, int Cols, Backend B>
T norm(const Matrix<T, Rows, Cols, B>& m) {
    return m.norm();
}

// 单位矩阵
template<typename T, int N, Backend B = Backend::CPU>
Matrix<T, N, N, B> identity() {
    Matrix<T, N, N, B> result;
    result.setIdentity();
    return result;
}

// 转置
template<typename T, int Rows, int Cols, Backend B>
Matrix<T, Cols, Rows, B> transpose(const Matrix<T, Rows, Cols, B>& m) {
    Matrix<T, Cols, Rows, B> result;
    for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < Cols; ++j) result(j, i) = m(i, j);
    return result;
}

// 对角线向量 -> 对角矩阵
template<typename T, int N, Backend B>
Matrix<T, N, N, B> diagonal_matrix(const Vector<T, N, B>& d) {
    Matrix<T, N, N, B> result;
    for (int i = 0; i < N; ++i) result(i, i) = d[i];
    return result;
}

// 行列式: 高斯消元 (部分主元), 奇异(主元过小) 返回 0
template<typename T, int N, Backend B>
T determinant(const Matrix<T, N, N, B>& m) {
    static_assert(N > 0, "determinant requires N >= 1");
    Matrix<T, N, N, B> a(m);
    T scale = T{0};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) scale = std::max(scale, std::abs(m(i, j)));
    const T threshold = std::isfinite(scale)
                          ? scale * (std::numeric_limits<T>::epsilon() * T{1000})
                          : T{0};
    if (scale == T{0}) return T{0};

    T det = T{1};
    for (int col = 0; col < N; ++col) {
        // 选择主元行
        int pivot = col;
        T maxAbs = std::abs(a(col, col));
        for (int row = col + 1; row < N; ++row) {
            const T v = std::abs(a(row, col));
            if (v > maxAbs) { maxAbs = v; pivot = row; }
        }
        if (maxAbs <= threshold) return T{0};
        if (pivot != col) {
            for (int c = col; c < N; ++c) std::swap(a(col, c), a(pivot, c));
            det = -det;
        }
        const T pivotVal = a(col, col);
        det *= pivotVal;
        for (int row = col + 1; row < N; ++row) {
            const T factor = a(row, col) / pivotVal;
            if (factor == T{0}) continue;
            for (int c = col; c < N; ++c) a(row, c) -= factor * a(col, c);
        }
    }
    return det;
}

// 逆矩阵: 高斯-约当消元; 奇异时返回零矩阵 (不抛异常)
template<typename T, int N, Backend B>
Matrix<T, N, N, B> inverse(const Matrix<T, N, N, B>& m) {
    static_assert(N > 0, "inverse requires N >= 1");
    Matrix<T, N, N, B> a(m);
    Matrix<T, N, N, B> inv = identity<T, N, B>();
    T scale = T{0};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) scale = std::max(scale, std::abs(m(i, j)));
    const T threshold = std::isfinite(scale)
                          ? scale * (std::numeric_limits<T>::epsilon() * T{1000})
                          : T{0};
    if (scale == T{0}) {
        inv.setZero();
        return inv;
    }
    for (int col = 0; col < N; ++col) {
        int pivot = col;
        T maxAbs = std::abs(a(col, col));
        for (int row = col + 1; row < N; ++row) {
            const T v = std::abs(a(row, col));
            if (v > maxAbs) { maxAbs = v; pivot = row; }
        }
        if (maxAbs <= threshold) {
            // 奇异: 返回零矩阵
            inv.setZero();
            return inv;
        }
        if (pivot != col) {
            for (int c = 0; c < N; ++c) {
                std::swap(a(col, c), a(pivot, c));
                std::swap(inv(col, c), inv(pivot, c));
            }
        }
        const T pivotVal = a(col, col);
        for (int c = 0; c < N; ++c) {
            a(col, c) /= pivotVal;
            inv(col, c) /= pivotVal;
        }
        for (int row = 0; row < N; ++row) {
            if (row == col) continue;
            const T factor = a(row, col);
            if (factor == T{0}) continue;
            for (int c = 0; c < N; ++c) {
                a(row, c) -= factor * a(col, c);
                inv(row, c) -= factor * inv(col, c);
            }
        }
    }
    return inv;
}

// ---- 类型别名: 默认单精度浮点 ----
template<int R, int C, Backend B = Backend::CPU>
using Mat = Matrix<Float32, R, C, B>;

template<int R, int C, Backend B = Backend::CPU>
using Matd = Matrix<Float64, R, C, B>;

// 常用别名
using Mat3 = Mat<3, 3, Backend::CPU>;

} // namespace hlcl

// 主模板定义完成后才引入 GPU 偏特化 (偏特化要求主模板已声明)。
#ifdef HLCL_GPU_ENABLED
#include "matrix_gpu.hpp"
#endif // HLCL_GPU_ENABLED

// Kompute (Vulkan compute) 偏特化, 同样要求主模板已声明。
#ifdef HLCL_KOMPUTE_ENABLED
#include "matrix_kompute.hpp"
#endif // HLCL_KOMPUTE_ENABLED
