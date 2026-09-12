#pragma once

#include "backend_traits.hpp"
#include "vector.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace hlcl {

// 统一三后端固定矩阵模板 (行优先)。
// CPU: 内联 std::array (零堆分配); GPU: shared-USM; Kompute: eHost 张量。
// 平方同型矩阵乘经 BackendTraits 分派 (CPU 主机循环 / GPU SYCL 内核 /
// Kompute Vulkan 内核, double 回落宿主); 其余运算为主机循环——三种后端的
// 存储均为宿主可读, 与历史行为一致。
template<typename T, int Rows, int Cols, Backend B = Backend::CPU>
class Matrix {
    static_assert(Rows > 0 && Cols > 0, "Matrix requires Rows, Cols >= 1");
    using Traits = BackendTraits<B>;
    using Store = typename Traits::template Storage<T, Rows * Cols>;

public:
    using value_type = T;
    using size_type = int;
    static constexpr int kRows = Rows;
    static constexpr int kCols = Cols;
    static constexpr Backend kBackend = B;

    Matrix() = default;   // 零矩阵 (各后端 Storage 构造即清零)

    // 大小构造 (固定大小矩阵忽略参数, 零填充)
    explicit Matrix(size_type) {}

    // 从连续行优先数据深拷贝
    explicit Matrix(const T* p) {
        std::memcpy(st_.data(), p, sizeof(T) * static_cast<std::size_t>(Rows * Cols));
    }

    // 扁平初始化列表, 行优先填充
    Matrix(std::initializer_list<T> init) {
        assert(init.size() <= static_cast<std::size_t>(Rows * Cols) &&
               "too many initializer entries for Matrix");
        size_type i = 0;
        for (const T& v : init) st_.data()[i++] = v;
    }

    // 嵌套初始化列表: { {r0c0, ...}, {r1c0, ...}, ... }
    Matrix(std::initializer_list<std::initializer_list<T>> rows_init) {
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

    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

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
        return st_.data()[static_cast<std::size_t>(r * Cols + c)];
    }
    const T& at(size_type r, size_type c) const {
        assert(r >= 0 && r < Rows && c >= 0 && c < Cols && "index out of range");
        return st_.data()[static_cast<std::size_t>(r * Cols + c)];
    }

    T& operator()(size_type row, size_type col) { return at(row, col); }
    const T& operator()(size_type row, size_type col) const { return at(row, col); }

    // 连续行优先数据访问
    T* data() { return st_.data(); }
    const T* data() const { return st_.data(); }

    void setZero() {
        for (int i = 0; i < Rows * Cols; ++i) st_.data()[i] = T{0};
    }
    void setConstant(const T& value) {
        for (int i = 0; i < Rows * Cols; ++i) st_.data()[i] = value;
    }

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

    constexpr T trace() const {
        const int n = std::min(Rows, Cols);
        T result = T{0};
        for (int i = 0; i < n; ++i) result += (*this)(i, i);
        return result;
    }

    constexpr T squaredNorm() const {
        T result = T{0};
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) {
                const T v = (*this)(i, j);
                result += v * v;
            }
        return result;
    }

    T norm() const { return static_cast<T>(std::sqrt(squaredNorm())); }

    constexpr T sum() const {
        T result = T{0};
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result += (*this)(i, j);
        return result;
    }

    constexpr T maxCoeff() const {
        T result = (*this)(0, 0);
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result = std::max(result, (*this)(i, j));
        return result;
    }

    constexpr T minCoeff() const {
        T result = (*this)(0, 0);
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) result = std::min(result, (*this)(i, j));
        return result;
    }

    // ---- 成员逐元素运算 (同构矩阵, 主机循环) ----
    [[nodiscard]] Matrix operator+(const Matrix& other) const {
        Matrix result;
        for (int i = 0; i < Rows * Cols; ++i)
            result.st_.data()[i] = st_.data()[i] + other.st_.data()[i];
        return result;
    }
    [[nodiscard]] Matrix operator-(const Matrix& other) const {
        Matrix result;
        for (int i = 0; i < Rows * Cols; ++i)
            result.st_.data()[i] = st_.data()[i] - other.st_.data()[i];
        return result;
    }

    // ---- 成员标量运算 (主机循环) ----
    [[nodiscard]] Matrix operator+(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows * Cols; ++i) result.st_.data()[i] = st_.data()[i] + scalar;
        return result;
    }
    [[nodiscard]] Matrix operator-(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows * Cols; ++i) result.st_.data()[i] = st_.data()[i] - scalar;
        return result;
    }
    [[nodiscard]] Matrix operator*(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows * Cols; ++i) result.st_.data()[i] = st_.data()[i] * scalar;
        return result;
    }
    [[nodiscard]] Matrix operator/(T scalar) const {
        Matrix result;
        for (int i = 0; i < Rows * Cols; ++i) result.st_.data()[i] = st_.data()[i] / scalar;
        return result;
    }

    // ---- 矩阵乘法: Rows x Cols * Cols x OtherCols -> Rows x OtherCols ----
    // 平方同型走 BackendTraits 分派 (GPU/Kompute 内核), 其余主机循环。
    template<typename U, int OtherCols, Backend B2>
    [[nodiscard]] Matrix<T, Rows, OtherCols, B> operator*(
        const Matrix<U, Cols, OtherCols, B2>& other) const {
        Matrix<T, Rows, OtherCols, B> result;
        if constexpr (Rows == Cols && Cols == OtherCols && std::is_same_v<T, U>) {
            Traits::template mat_mul<T, Rows>(data(), other.data(), result.data());
        } else {
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < OtherCols; ++j) {
                    T acc = T{0};
                    for (int k = 0; k < Cols; ++k)
                        acc += (*this)(i, k) * static_cast<T>(other(k, j));
                    result(i, j) = acc;
                }
            }
        }
        return result;
    }

    // 与 CPU 主机的显式拷贝 (GPU/Kompute 用; 数据本就宿主可读)
    void copyFromHost(const Matrix<T, Rows, Cols, Backend::CPU>& h) {
        std::memcpy(st_.data(), h.data(), sizeof(T) * static_cast<std::size_t>(Rows * Cols));
    }
    void copyToHost(Matrix<T, Rows, Cols, Backend::CPU>& h) const {
        std::memcpy(h.data(), st_.data(), sizeof(T) * static_cast<std::size_t>(Rows * Cols));
    }

private:
    Store st_;
};

// ---- 矩阵标量前置运算 (后置形式为成员, 避免歧义) ----
template<typename T, int Rows, int Cols, Backend B>
[[nodiscard]] Matrix<T, Rows, Cols, B> operator*(T scalar, const Matrix<T, Rows, Cols, B>& m) {
    return m * scalar;
}

template<typename T, int Rows, int Cols, Backend B>
[[nodiscard]] Matrix<T, Rows, Cols, B> operator+(T scalar, const Matrix<T, Rows, Cols, B>& m) {
    return m + scalar;
}

template<typename T, int Rows, int Cols, Backend B>
[[nodiscard]] Matrix<T, Rows, Cols, B> operator-(T scalar, const Matrix<T, Rows, Cols, B>& m) {
    Matrix<T, Rows, Cols, B> result;
    for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < Cols; ++j) result(i, j) = scalar - m(i, j);
    return result;
}

// 矩阵范数 (自由函数)
template<typename T, int Rows, int Cols, Backend B>
[[nodiscard]] T norm(const Matrix<T, Rows, Cols, B>& m) {
    return m.norm();
}

// 单位矩阵
template<typename T, int N, Backend B = Backend::CPU>
[[nodiscard]] Matrix<T, N, N, B> identity() {
    Matrix<T, N, N, B> r;
    r.setIdentity();
    return r;
}

// 转置
template<typename T, int Rows, int Cols, Backend B>
[[nodiscard]] Matrix<T, Cols, Rows, B> transpose(const Matrix<T, Rows, Cols, B>& m) {
    Matrix<T, Cols, Rows, B> r;
    for (int i = 0; i < Rows; ++i)
        for (int j = 0; j < Cols; ++j) r(j, i) = m(i, j);
    return r;
}

// 对角线向量 -> 对角矩阵
template<typename T, int N, Backend B>
[[nodiscard]] Matrix<T, N, N, B> diagonal_matrix(const Vector<T, N, B>& d) {
    Matrix<T, N, N, B> r;
    for (int i = 0; i < N; ++i) r(i, i) = d[i];
    return r;
}

// 行列式: 高斯消元 (部分主元), 奇异(主元过小) 返回 0
template<typename T, int N, Backend B>
[[nodiscard]] T determinant(const Matrix<T, N, N, B>& m) {
    static_assert(N > 0, "determinant requires N >= 1");
    Matrix<T, N, N, B> a(m);
    T result = T{1};
    T scale = T{0};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) scale = std::max(scale, std::abs(a(i, j)));
    const T threshold = std::isfinite(scale)
                          ? scale * (std::numeric_limits<T>::epsilon() * T{1000})
                          : T{0};
    if (scale == T{0}) return T{0};
    for (int col = 0; col < N; ++col) {
        int pivot = col;
        T maxAbs = std::abs(a(col, col));
        for (int row = col + 1; row < N; ++row) {
            const T v = std::abs(a(row, col));
            if (v > maxAbs) { maxAbs = v; pivot = row; }
        }
        if (maxAbs <= threshold) return T{0};   // 奇异
        if (pivot != col) {
            for (int j = col; j < N; ++j)
                std::swap(a(col, j), a(pivot, j));
            result = -result;
        }
        const T pivotValue = a(col, col);
        for (int row = col + 1; row < N; ++row) {
            const T factor = a(row, col) / pivotValue;
            for (int j = col; j < N; ++j) a(row, j) -= factor * a(col, j);
        }
        result *= pivotValue;
    }
    return result;
}

// 逆矩阵: 高斯-约当消元; 奇异时返回零矩阵 (不抛异常)
template<typename T, int N, Backend B>
[[nodiscard]] Matrix<T, N, N, B> inverse(const Matrix<T, N, N, B>& m) {
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
        const T pivotValue = a(col, col);
        for (int c = 0; c < N; ++c) {
            a(col, c) /= pivotValue;
            inv(col, c) /= pivotValue;
        }
        for (int row = 0; row < N; ++row) {
            if (row == col) continue;
            const T factor = a(row, col);
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

#ifdef HLCL_GPU_ENABLED
// ---- GPU 内核入口 (保持历史自由函数名) ----

/// C = A * B via the SYCL kernel (N x N).
template<typename T, int N>
[[nodiscard]] Matrix<T, N, N, Backend::GPU> matrixMultiplyGPU(
    const Matrix<T, N, N, Backend::GPU>& A,
    const Matrix<T, N, N, Backend::GPU>& B) {
    Matrix<T, N, N, Backend::GPU> r;
    BackendTraits<Backend::GPU>::template mat_mul<T, N>(A.data(), B.data(), r.data());
    return r;
}

/// inv(m) via the SYCL Gauss-Jordan kernel. Singular -> zero matrix.
template<typename T, int N>
[[nodiscard]] Matrix<T, N, N, Backend::GPU> inverseGPU(const Matrix<T, N, N, Backend::GPU>& m) {
    Matrix<T, N, N, Backend::GPU> r;
    gpu::matrix_inverse<T, N>({m.data(), std::size_t(N) * N},
                              {r.data(), std::size_t(N) * N});
    return r;
}

/// det(m) via the SYCL elimination kernel.
template<typename T, int N>
[[nodiscard]] T determinantGPU(const Matrix<T, N, N, Backend::GPU>& m) {
    return gpu::matrix_determinant<T, N>({m.data(), std::size_t(N) * N});
}
#endif // HLCL_GPU_ENABLED

#ifdef HLCL_KOMPUTE_ENABLED
// ---- Kompute 内核入口 (保持历史自由函数名) ----

/// C = A * B via the Vulkan kernel (N x N, float; double 回落宿主).
template<typename T, int N>
[[nodiscard]] Matrix<T, N, N, Backend::Kompute> matrixMultiplyKompute(
    const Matrix<T, N, N, Backend::Kompute>& A,
    const Matrix<T, N, N, Backend::Kompute>& B) {
    Matrix<T, N, N, Backend::Kompute> r;
    kompute::matrix_multiply<T, N>(A.data(), B.data(), r.data());
    return r;
}

/// inv(m) via the Vulkan Gauss-Jordan kernel. Singular -> zero matrix.
template<typename T, int N>
[[nodiscard]] Matrix<T, N, N, Backend::Kompute> inverseKompute(
    const Matrix<T, N, N, Backend::Kompute>& m) {
    Matrix<T, N, N, Backend::Kompute> r;
    kompute::matrix_inverse<T, N>(m.data(), r.data());
    return r;
}

/// det(m) via the Vulkan elimination kernel.
template<typename T, int N>
[[nodiscard]] T determinantKompute(const Matrix<T, N, N, Backend::Kompute>& m) {
    return kompute::matrix_determinant<T, N>(m.data());
}
#endif // HLCL_KOMPUTE_ENABLED

} // namespace hlcl
