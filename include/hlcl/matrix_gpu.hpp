#pragma once

// GPU (Backend::GPU) Matrix partial specialization.
//
// The specialization mirrors the CPU Matrix public API on a contiguous host
// copy (so the generic free functions in matrix.hpp -- identity/transpose/
// inverse/determinant -- keep working over Backend::GPU), and dispatches the
// hot NxN matrix product to the verified SYCL kernel in gpu_impl.hpp.
//
// This header is included from matrix.hpp AFTER the primary template is
// defined (partial specialization requires the primary to be declared first).

#ifdef HLCL_GPU_ENABLED

#include "backend.hpp"
#include "gpu_impl.hpp"
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
class Matrix<T, Rows, Cols, Backend::GPU> {
public:
    using value_type = T;
    using size_type = int;
    static constexpr int kRows = Rows;
    static constexpr int kCols = Cols;
    static constexpr Backend kBackend = Backend::GPU;

    Matrix() : data_(cl::sycl::malloc_shared<T>(Rows * Cols, gpu::queue())) {
        fill(T{0});
    }

    explicit Matrix(const T* p)
        : data_(cl::sycl::malloc_shared<T>(Rows * Cols, gpu::queue())) {
        std::memcpy(data_, p, sizeof(T) * Rows * Cols);
    }

    Matrix(std::initializer_list<std::initializer_list<T>> init)
        : data_(cl::sycl::malloc_shared<T>(Rows * Cols, gpu::queue())) {
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

    ~Matrix() {
        if (data_) cl::sycl::free(data_, gpu::queue());
    }

    Matrix(const Matrix& o)
        : data_(cl::sycl::malloc_shared<T>(Rows * Cols, gpu::queue())) {
        std::memcpy(data_, o.data_, sizeof(T) * Rows * Cols);
    }
    Matrix& operator=(const Matrix& o) {
        if (this != &o) std::memcpy(data_, o.data_, sizeof(T) * Rows * Cols);
        return *this;
    }
    Matrix(Matrix&& o) noexcept : data_(o.data_) { o.data_ = nullptr; }
    Matrix& operator=(Matrix&& o) noexcept {
        if (this != &o) {
            if (data_) cl::sycl::free(data_, gpu::queue());
            data_ = o.data_;
            o.data_ = nullptr;
        }
        return *this;
    }

    // Converting ctor from any backend of the same element/shape.
    template<typename U, Backend B2>
    Matrix(const Matrix<U, Rows, Cols, B2>& o)
        : data_(cl::sycl::malloc_shared<T>(Rows * Cols, gpu::queue())) {
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
        return data_[r * Cols + c];
    }
    const T& at(size_type r, size_type c) const {
        assert(r >= 0 && r < Rows && c >= 0 && c < Cols && "index out of range");
        return data_[r * Cols + c];
    }
    T& operator()(size_type r, size_type c) { return at(r, c); }
    const T& operator()(size_type r, size_type c) const { return at(r, c); }

    T* data() { return data_; }
    const T* data() const { return data_; }
    T* get_host_data() { return data_; }
    const T* get_host_data() const { return data_; }

    void setZero() { fill(T{0}); }
    void setConstant(const T& v) { fill(v); }
    void fill(T v) {
        for (int i = 0; i < Rows * Cols; ++i) data_[i] = v;
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
        for (int i = 0; i < Rows * Cols; ++i) r += data_[i] * data_[i];
        return r;
    }
    T norm() const { return static_cast<T>(std::sqrt(squaredNorm())); }
    T sum() const {
        T r = T{0};
        for (int i = 0; i < Rows * Cols; ++i) r += data_[i];
        return r;
    }
    T maxCoeff() const {
        assert(Rows * Cols > 0);
        T r = data_[0];
        for (int i = 1; i < Rows * Cols; ++i) r = std::max(r, data_[i]);
        return r;
    }
    T minCoeff() const {
        assert(Rows * Cols > 0);
        T r = data_[0];
        for (int i = 1; i < Rows * Cols; ++i) r = std::min(r, data_[i]);
        return r;
    }

    // ---- elementwise same-type ops (host) ----
    Matrix operator+(const Matrix& o) const {
        Matrix r;
        for (int i = 0; i < Rows * Cols; ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    Matrix operator-(const Matrix& o) const {
        Matrix r;
        for (int i = 0; i < Rows * Cols; ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }
    // ---- scalar ops ----
    Matrix operator+(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.data_[i]=data_[i]+s; return r; }
    Matrix operator-(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.data_[i]=data_[i]-s; return r; }
    Matrix operator*(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.data_[i]=data_[i]*s; return r; }
    Matrix operator/(T s) const { Matrix r; for (int i = 0; i < Rows*Cols; ++i) r.data_[i]=data_[i]/s; return r; }

    // ---- matrix product: Rows x Cols * Cols x OtherCols -> Rows x OtherCols ----
    // Square GPU x GPU uses the verified SYCL kernel; everything else falls
    // back to the identical host triple-loop (same summation order as CPU).
    template<typename U, int OtherCols, Backend B2>
    Matrix<T, Rows, OtherCols, Backend::GPU> operator*(
        const Matrix<U, Cols, OtherCols, B2>& o) const {
        Matrix<T, Rows, OtherCols, Backend::GPU> r;
        if constexpr (Rows == Cols && Cols == OtherCols && B2 == Backend::GPU &&
                      std::is_same<T, U>::value) {
            gpu::matrix_multiply<T, Rows>(data(), o.data(), r.data());
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
        std::memcpy(data_, h.data(), sizeof(T) * Rows * Cols);
    }
    void copyToHost(Matrix<T, Rows, Cols, Backend::CPU>& h) const {
        std::memcpy(h.data(), data_, sizeof(T) * Rows * Cols);
    }

private:
    T* data_ = nullptr;
};

// ---- GPU kernel-backed free operations ----

/// C = A * B via the SYCL kernel (N x N).
template<typename T, int N>
Matrix<T, N, N, Backend::GPU> matrixMultiplyGPU(
    const Matrix<T, N, N, Backend::GPU>& A,
    const Matrix<T, N, N, Backend::GPU>& B) {
    Matrix<T, N, N, Backend::GPU> r;
    gpu::matrix_multiply<T, N>(A.data(), B.data(), r.data());
    return r;
}

/// inv(m) via the SYCL Gauss-Jordan kernel. Singular -> zero matrix.
template<typename T, int N>
Matrix<T, N, N, Backend::GPU> inverseGPU(const Matrix<T, N, N, Backend::GPU>& m) {
    Matrix<T, N, N, Backend::GPU> r;
    gpu::matrix_inverse<T, N>(m.data(), r.data());
    return r;
}

/// det(m) via the SYCL elimination kernel.
template<typename T, int N>
T determinantGPU(const Matrix<T, N, N, Backend::GPU>& m) {
    return gpu::matrix_determinant<T, N>(m.data());
}

} // namespace hlcl

#else // HLCL_GPU_ENABLED

// GPU-support disabled: Backend::GPU falls back to the generic (host) Matrix
// primary template, so no specialization is declared here.

#endif // HLCL_GPU_ENABLED
