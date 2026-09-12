#pragma once

// GPU (Backend::GPU) Vector partial specialization.
//
// Mirrors the CPU Vector public API on a contiguous host copy and dispatches
// elementwise operations (add / scale) to the verified SYCL kernels in
// gpu_impl.hpp. Norm / dot are exact host reductions, matching the CPU result
// to FP precision.
//
// Included from vector.hpp AFTER the primary template is defined.

#ifdef HLCL_GPU_ENABLED

#include "backend.hpp"
#include "gpu_impl.hpp"
#include <span>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>

namespace hlcl {

template<typename T, int Size>
class Vector<T, Size, Backend::GPU> {
public:
    using value_type = T;
    using size_type = int;
    static constexpr int kSize = Size;
    static constexpr Backend kBackend = Backend::GPU;
    static constexpr bool kDynamic = (Size < 0);

    Vector() : n_(Size > 0 ? Size : 0) { alloc_(); setZero(); }

    explicit Vector(size_type n) : n_(Size > 0 ? Size : n) { alloc_(); setZero(); }

    Vector(std::initializer_list<T> init) : n_(static_cast<size_type>(init.size())) {
        if (!kDynamic) {
            assert(init.size() == static_cast<std::size_t>(Size) &&
                   "initializer list size must match fixed Vector size");
            n_ = Size;
        }
        alloc_();
        size_type i = 0;
        for (const T& v : init) data_[i++] = v;
    }

    ~Vector() {
        if (data_) sycl::free(data_, gpu::queue());
    }

    Vector(const Vector& o) : data_(nullptr), n_(o.n_) { deepCopy_(o); }
    Vector& operator=(const Vector& o) {
        if (this != &o) {
            if (kDynamic) {
                if (data_) sycl::free(data_, gpu::queue());
                n_ = o.n_;
                data_ = nullptr;
            }
            deepCopy_(o);
        }
        return *this;
    }
    Vector(Vector&& o) noexcept : data_(o.data_), n_(o.n_) { o.n_ = 0; o.data_ = nullptr; }
    Vector& operator=(Vector&& o) noexcept {
        if (this != &o) {
            if (data_) sycl::free(data_, gpu::queue());
            n_ = o.n_;
            data_ = o.data_;
            o.n_ = 0;
            o.data_ = nullptr;
        }
        return *this;
    }

    template<typename U, int Size2, Backend B2>
    Vector(const Vector<U, Size2, B2>& o) : n_(Size > 0 ? Size : o.size()) {
        alloc_();
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) data_[i] = static_cast<T>(o[i]);
    }
    template<typename U, int Size2, Backend B2>
    Vector& operator=(const Vector<U, Size2, B2>& o) {
        if (kDynamic) {
            if (data_) sycl::free(data_, gpu::queue());
            n_ = o.size();
            alloc_();
        } else {
            setZero();
        }
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) data_[i] = static_cast<T>(o[i]);
        return *this;
    }

    size_type size() const { return n_; }

    T& at(size_type i) {
        assert(i >= 0 && i < size() && "index out of range");
        return data_[i];
    }
    const T& at(size_type i) const {
        assert(i >= 0 && i < size() && "index out of range");
        return data_[i];
    }
    T& operator()(size_type i) { return at(i); }
    const T& operator()(size_type i) const { return at(i); }
    T& operator[](size_type i) { return at(i); }
    const T& operator[](size_type i) const { return at(i); }
    T& coeffRef(size_type i) { return at(i); }
    T coeff(size_type i) const { return at(i); }

    T* data() { return data_; }
    const T* data() const { return data_; }
    T* get_host_data() { return data_; }
    const T* get_host_data() const { return data_; }

    T x() const { return at(0); }
    T y() const { return at(1); }
    T z() const { return at(2); }

    void setZero() {
        for (int i = 0; i < size(); ++i) data_[i] = T{0};
    }
    void setConstant(const T& v) {
        for (int i = 0; i < size(); ++i) data_[i] = v;
    }
    void resize(size_type n) {
        if (!kDynamic) return;
        if (n == n_) return;
        T* nd = (n > 0) ? sycl::malloc_shared<T>(n, gpu::queue()) : nullptr;
        if (nd) {
            const int m = (data_) ? std::min(n_, n) : 0;
            for (int i = 0; i < m; ++i) nd[i] = data_[i];
            for (int i = m; i < n; ++i) nd[i] = T{0};
        }
        if (data_) sycl::free(data_, gpu::queue());
        data_ = nd;
        n_ = n;
    }

    T sum() const {
        T r = T{0};
        for (int i = 0; i < size(); ++i) r += at(i);
        return r;
    }
    T maxCoeff() const {
        assert(size() > 0 && "maxCoeff on empty vector");
        T r = at(0);
        for (int i = 1; i < size(); ++i) r = std::max(r, at(i));
        return r;
    }
    T minCoeff() const {
        assert(size() > 0 && "minCoeff on empty vector");
        T r = at(0);
        for (int i = 1; i < size(); ++i) r = std::min(r, at(i));
        return r;
    }
    T squaredNorm() const {
        T r = T{0};
        for (int i = 0; i < size(); ++i) r += at(i) * at(i);
        return r;
    }
    T norm() const { return static_cast<T>(std::sqrt(squaredNorm())); }

    template<typename U, int Size2, Backend B2>
    T dot(const Vector<U, Size2, B2>& o) const {
        assert(o.size() == size() && "dot product requires equal sizes");
        T r = T{0};
        for (int i = 0; i < size(); ++i) r += at(i) * static_cast<T>(o[i]);
        return r;
    }

    // ---- elementwise / scalar ops: add, sub, scale via SYCL kernels ----
    Vector operator+(const Vector& o) const {
        assert(o.size() == size() && "vector sizes must match");
        Vector r;
        r.resize(size());
        if (size() > 0) gpu::vector_add<T>(span_(), o.span_(), r.span_());
        return r;
    }
    Vector operator-(const Vector& o) const {
        assert(o.size() == size() && "vector sizes must match");
        Vector r;
        r.resize(size());
        if (size() > 0) gpu::vector_sub<T>(span_(), o.span_(), r.span_());
        return r;
    }
    Vector& operator+=(const Vector& o) {
        assert(o.size() == size() && "vector sizes must match");
        if (size() > 0) gpu::vector_add<T>(span_(), o.span_(), span_());
        return *this;
    }
    Vector& operator-=(const Vector& o) {
        assert(o.size() == size() && "vector sizes must match");
        if (size() > 0) gpu::vector_sub<T>(span_(), o.span_(), span_());
        return *this;
    }
    Vector& operator*=(T s) {
        if (size() > 0) gpu::vector_scale<T>(s, span_(), span_());
        return *this;
    }
    Vector& operator/=(T s) {
        if (size() > 0) gpu::vector_scale<T>(static_cast<T>(T{1} / s), span_(), span_());
        return *this;
    }
    Vector operator-() const {
        Vector r(*this);
        if (size() > 0) gpu::vector_scale<T>(T{-1}, span_(), r.span_());
        return r;
    }
    Vector operator+(T s) const {
        Vector r(*this);
        for (int i = 0; i < size(); ++i) r[i] += s;
        return r;
    }
    Vector operator-(T s) const { return *this + (-s); }
    Vector operator*(T s) const {
        Vector r(*this);
        if (size() > 0) gpu::vector_scale<T>(s, span_(), r.span_());
        return r;
    }
    Vector operator/(T s) const {
        Vector r(*this);
        if (size() > 0) gpu::vector_scale<T>(static_cast<T>(T{1} / s), span_(), r.span_());
        return r;
    }

private:
    std::span<T> span_() { return {data_, static_cast<std::size_t>(n_)}; }
    std::span<const T> span_() const { return {data_, static_cast<std::size_t>(n_)}; }
    T* data_ = nullptr;
    size_type n_ = 0;

    void alloc_() {
        if (n_ > 0) data_ = sycl::malloc_shared<T>(n_, gpu::queue());
    }
    // (Re)allocate for n_ elements and deep-copy from o. Fixed-size copy
    // assignment reuses the existing allocation (same length); dynamic
    // frees and reallocates at o's length.
    void deepCopy_(const Vector& o) {
        const bool needAlloc = kDynamic || data_ == nullptr;
        if (kDynamic && data_) {
            sycl::free(data_, gpu::queue());
            data_ = nullptr;
        }
        n_ = kDynamic ? o.n_ : n_;
        if (needAlloc) alloc_();
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) data_[i] = o.data_[i];
        for (int i = m; i < size(); ++i) data_[i] = T{0};
    }
};

// ---- kernel-backed free operations (GPU) ----
template<typename T, int Size>
T vectorNormGPU(const Vector<T, Size, Backend::GPU>& v) { return v.norm(); }

template<typename T, int Size, Backend B1, Backend B2>
T vectorDotGPU(const Vector<T, Size, B1>& v1, const Vector<T, Size, B2>& v2) {
    return v1.dot(v2);
}

template<typename T, int Size>
Vector<T, Size, Backend::GPU> vectorAddGPU(const Vector<T, Size, Backend::GPU>& a,
                                           const Vector<T, Size, Backend::GPU>& b) {
    return a + b;
}
template<typename T, int Size>
Vector<T, Size, Backend::GPU> vectorSubGPU(const Vector<T, Size, Backend::GPU>& a,
                                           const Vector<T, Size, Backend::GPU>& b) {
    return a - b;
}
template<typename T, int Size>
Vector<T, Size, Backend::GPU> vectorScaleGPU(T s, const Vector<T, Size, Backend::GPU>& a) {
    return a * s;
}

} // namespace hlcl

#else // HLCL_GPU_ENABLED

// GPU-support disabled: Backend::GPU falls back to the generic (host) Vector
// primary template, so no specialization is declared here.

#endif // HLCL_GPU_ENABLED
