#pragma once

// Kompute (Backend::Kompute) Matrix partial specialization.
//
// Mirrors matrix_gpu.hpp 1:1 — same public API, same "host loops for small
// ops, device kernel for the square product" split — but storage is a
// kp::TensorT<T> in eHost mode (host-visible coherent memory, so data()
// remains a plain CPU pointer) and the square product dispatches to the
// Vulkan compute kernel in kompute_impl.hpp.
//
// Double is computed on the host (Kompute 0.8.0 cannot enable
// shaderFloat64); see kompute_impl.hpp. This header is included from
// matrix.hpp AFTER the primary template is defined.

#ifdef HLCL_KOMPUTE_ENABLED

#include "backend.hpp"
#include "kompute_impl.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <type_traits>

namespace hlcl {

template<typename T, int Rows, int Cols>
class Matrix<T, Rows, Cols, Backend::Kompute> {
public:
    using value_type = T;
    using size_type = int;
    static constexpr int kRows = Rows;
    static constexpr int kCols = Cols;
    static constexpr Backend kBackend = Backend::Kompute;

    Matrix() : tensor_(kompute::make_tensor<T>(Rows * Cols)) { fill(T{0}); }

    explicit Matrix(const T* p)
        : tensor_(kompute::make_tensor<T>(Rows * Cols)) {
        std::memcpy(tensor_->data(), p, sizeof(T) * Rows * Cols);
    }

    Matrix(std::initializer_list<std::initializer_list<T>> init)
        : tensor_(kompute::make_tensor<T>(Rows * Cols)) {
        fill(T{0});
        size_type r = 0;
        for (const auto& row : init) {
            size_type c = 0;
            for (const auto& v : row) {
                if (r < Rows && c < Cols) (*this)(r, c) = v;
                ++c;
            }
            ++r;
        }
    }

    // default dtor: the tensor dies with the last owner; device buffers are
    // released by Kompute. The leaked Manager singleton outlives everything.

    Matrix(const Matrix& o) : tensor_(kompute::make_tensor<T>(Rows * Cols)) {
        std::memcpy(tensor_->data(), o.tensor_->data(), sizeof(T) * Rows * Cols);
    }
    Matrix& operator=(const Matrix& o) {
        if (this != &o)
            std::memcpy(tensor_->data(), o.tensor_->data(),
                sizeof(T) * Rows * Cols);
        return *this;
    }
    Matrix(Matrix&& o) noexcept : tensor_(std::move(o.tensor_)) {}
    Matrix& operator=(Matrix&& o) noexcept {
        if (this != &o) tensor_ = std::move(o.tensor_);
        return *this;
    }

    // Converting ctor from any backend of the same element/shape.
    template<typename U, Backend B2>
    Matrix(const Matrix<U, Rows, Cols, B2>& o)
        : tensor_(kompute::make_tensor<T>(Rows * Cols)) {
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) (*this)(i, j) = static_cast<T>(o(i, j));
    }
    template<typename U, Backend B2>
    Matrix& operator=(const Matrix<U, Rows, Cols, B2>& o) {
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j) (*this)(i, j) = static_cast<T>(o(i, j));
        return *this;
    }

    size_type rows() const { return Rows; }
    size_type cols() const { return Cols; }
    size_type size() const { return Rows * Cols; }

    T& at(size_type r, size_type c) {
        assert(r >= 0 && r < Rows && c >= 0 && c < Cols && "index out of range");
        return tensor_->data()[r * Cols + c];
    }
    const T& at(size_type r, size_type c) const {
        assert(r >= 0 && r < Rows && c >= 0 && c < Cols && "index out of range");
        return tensor_->data()[r * Cols + c];
    }
    T& operator()(size_type r, size_type c) { return at(r, c); }
    const T& operator()(size_type r, size_type c) const { return at(r, c); }

    T* data() { return tensor_->data(); }
    const T* data() const { return tensor_->data(); }
    T* get_host_data() { return tensor_->data(); }
    const T* get_host_data() const { return tensor_->data(); }

    void setZero() { fill(T{0}); }
    void setConstant(const T& v) { fill(v); }
    void fill(T v) {
        for (int i = 0; i < Rows * Cols; ++i) tensor_->data()[i] = v;
    }
    void setIdentity() {
        setZero();
        const int n = std::min(Rows, Cols);
        for (int i = 0; i < n; ++i) (*this)(i, i) = T{1};
    }
    void setDiagonal(const std::initializer_list<T>& values) {
        setZero();
        const int n = std::min({Rows, Cols, static_cast<int>(values.size())});
        int i = 0;
        for (const T& v : values) { if (i >= n) break; (*this)(i, i) = v; ++i; }
    }

    T trace() const {
        const int n = std::min(Rows, Cols);
        T r = T{0};
        for (int i = 0; i < n; ++i) r += (*this)(i, i);
        return r;
    }
    T squaredNorm() const {
        T r = T{0};
        for (int i = 0; i < Rows * Cols; ++i) r += tensor_->data()[i] * tensor_->data()[i];
        return r;
    }
    T norm() const { return static_cast<T>(std::sqrt(squaredNorm())); }
    T sum() const {
        T r = T{0};
        for (int i = 0; i < Rows * Cols; ++i) r += tensor_->data()[i];
        return r;
    }
    T maxCoeff() const {
        assert(Rows * Cols > 0);
        T r = tensor_->data()[0];
        for (int i = 1; i < Rows * Cols; ++i) r = std::max(r, tensor_->data()[i]);
        return r;
    }
    T minCoeff() const {
        assert(Rows * Cols > 0);
        T r = tensor_->data()[0];
        for (int i = 1; i < Rows * Cols; ++i) r = std::min(r, tensor_->data()[i]);
        return r;
    }

    // ---- elementwise same-type ops (host) ----
    Matrix operator+(const Matrix& o) const {
        Matrix r;
        for (int i = 0; i < Rows * Cols; ++i)
            r.tensor_->data()[i] = tensor_->data()[i] + o.tensor_->data()[i];
        return r;
    }
    Matrix operator-(const Matrix& o) const {
        Matrix r;
        for (int i = 0; i < Rows * Cols; ++i)
            r.tensor_->data()[i] = tensor_->data()[i] - o.tensor_->data()[i];
        return r;
    }
    // ---- scalar ops ----
    Matrix operator+(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.tensor_->data()[i]=tensor_->data()[i]+s; return r; }
    Matrix operator-(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.tensor_->data()[i]=tensor_->data()[i]-s; return r; }
    Matrix operator*(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.tensor_->data()[i]=tensor_->data()[i]*s; return r; }
    Matrix operator/(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.tensor_->data()[i]=tensor_->data()[i]/s; return r; }

    // ---- matrix product: Rows x Cols * Cols x OtherCols -> Rows x OtherCols ----
    // Square Kompute x Kompute (float) uses the Vulkan kernel; everything
    // else (double or mixed shapes) falls back to the identical host
    // triple-loop (same summation order as CPU).
    template<typename U, int OtherCols, Backend B2>
    Matrix<T, Rows, OtherCols, Backend::Kompute> operator*(
        const Matrix<U, Cols, OtherCols, B2>& o) const {
        Matrix<T, Rows, OtherCols, Backend::Kompute> r;
        if constexpr (Rows == Cols && Cols == OtherCols && B2 == Backend::Kompute &&
                      std::is_same<T, U>::value && std::is_same<T, float>::value) {
            kompute::matrix_multiply<T, Rows>(tensor_, o.tensor_, r.tensor_);
        } else {
            for (int i = 0; i < Rows; ++i)
                for (int j = 0; j < OtherCols; ++j) {
                    T acc = T{0};
                    for (int k = 0; k < Cols; ++k)
                        acc += (*this)(i, k) * static_cast<T>(o(k, j));
                    r(i, j) = acc;
                }
        }
        return r;
    }

    void copyFromHost(const Matrix<T, Rows, Cols, Backend::CPU>& h) {
        std::memcpy(tensor_->data(), h.data(), sizeof(T) * Rows * Cols);
    }
    void copyToHost(Matrix<T, Rows, Cols, Backend::CPU>& h) const {
        std::memcpy(h.data(), tensor_->data(), sizeof(T) * Rows * Cols);
    }

private:
    kompute::TensorPtr<T> tensor_;
};

// ---- Kompute kernel-backed free operations ----

/// C = A * B via the Vulkan kernel (N x N, float).
template<typename T, int N>
Matrix<T, N, N, Backend::Kompute> matrixMultiplyKompute(
    const Matrix<T, N, N, Backend::Kompute>& A,
    const Matrix<T, N, N, Backend::Kompute>& B) {
    Matrix<T, N, N, Backend::Kompute> r;
    kompute::matrix_multiply<T, N>(A.data(), B.data(), r.data());
    return r;
}

/// inv(m) via the Vulkan Gauss-Jordan kernel. Singular -> zero matrix.
template<typename T, int N>
Matrix<T, N, N, Backend::Kompute> inverseKompute(
    const Matrix<T, N, N, Backend::Kompute>& m) {
    Matrix<T, N, N, Backend::Kompute> r;
    kompute::matrix_inverse<T, N>(m.data(), r.data());
    return r;
}

/// det(m) via the Vulkan elimination kernel.
template<typename T, int N>
T determinantKompute(const Matrix<T, N, N, Backend::Kompute>& m) {
    return kompute::matrix_determinant<T, N>(m.data());
}

} // namespace hlcl

#else // HLCL_KOMPUTE_ENABLED

// Kompute support disabled: Backend::Kompute falls back to the generic (host)
// Matrix primary template, so no specialization is declared here.

#endif // HLCL_KOMPUTE_ENABLED
