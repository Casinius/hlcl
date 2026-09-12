#pragma once

// Kompute (Backend::Kompute) Vector partial specialization.
//
// Mirrors vector_gpu.hpp 1:1 — same public API, host reductions for
// norm/dot/sum, Vulkan kernels (kompute_impl.hpp) for elementwise add / sub /
// scale. Storage is a kp::TensorT<T> in eHost mode, so data() is a plain CPU
// pointer. Double ops stay on the host (see kompute_impl.hpp).
//
// Included from vector.hpp AFTER the primary template is defined.

#ifdef HLCL_KOMPUTE_ENABLED

#include "backend.hpp"
#include "kompute_impl.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace hlcl {

template<typename T, int Size>
class Vector<T, Size, Backend::Kompute> {
public:
    using value_type = T;
    using size_type = int;
    static constexpr int kSize = Size;
    static constexpr Backend kBackend = Backend::Kompute;
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
        for (const T& v : init) data()[i++] = v;
    }

    Vector(const Vector& o) : n_(o.n_) { deepCopy_(o); }
    Vector& operator=(const Vector& o) {
        if (this != &o) {
            if (kDynamic) { tensor_ = nullptr; n_ = o.n_; }
            deepCopy_(o);
        }
        return *this;
    }
    Vector(Vector&& o) noexcept : tensor_(std::move(o.tensor_)), n_(o.n_) { o.n_ = 0; }
    Vector& operator=(Vector&& o) noexcept {
        if (this != &o) {
            tensor_ = std::move(o.tensor_);
            n_ = o.n_;
            o.n_ = 0;
        }
        return *this;
    }

    template<typename U, int Size2, Backend B2>
    Vector(const Vector<U, Size2, B2>& o) : n_(Size > 0 ? Size : o.size()) {
        alloc_();
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) data()[i] = static_cast<T>(o[i]);
    }
    template<typename U, int Size2, Backend B2>
    Vector& operator=(const Vector<U, Size2, B2>& o) {
        if (kDynamic) {
            tensor_ = nullptr;
            n_ = o.size();
            alloc_();
        } else {
            setZero();
        }
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) data()[i] = static_cast<T>(o[i]);
        return *this;
    }

    size_type size() const { return n_; }

    T& at(size_type i) {
        assert(i >= 0 && i < size() && "index out of range");
        return data()[i];
    }
    const T& at(size_type i) const {
        assert(i >= 0 && i < size() && "index out of range");
        return data()[i];
    }
    T& operator()(size_type i) { return at(i); }
    const T& operator()(size_type i) const { return at(i); }
    T& operator[](size_type i) { return at(i); }
    const T& operator[](size_type i) const { return at(i); }
    T& coeffRef(size_type i) { return at(i); }
    T coeff(size_type i) const { return at(i); }

    T* data() { return tensor_ ? tensor_->data() : nullptr; }
    const T* data() const { return tensor_ ? tensor_->data() : nullptr; }
    T* get_host_data() { return data(); }
    const T* get_host_data() const { return data(); }

    T x() const { return at(0); }
    T y() const { return at(1); }
    T z() const { return at(2); }

    void setZero() {
        for (int i = 0; i < size(); ++i) data()[i] = T{0};
    }
    void setConstant(const T& v) {
        for (int i = 0; i < size(); ++i) data()[i] = v;
    }
    void resize(size_type n) {
        if (!kDynamic) return;
        if (n == n_) return;
        tensor_ = (n > 0) ? kompute::make_tensor<T>(n) : nullptr;
        n_ = n;
        setZero();
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

    // ---- elementwise / scalar ops: add, sub, scale via Vulkan kernels
    // (float); double 分流宿主 (Kompute 0.8.0 无 shaderFloat64) ----
    Vector operator+(const Vector& o) const {
        assert(o.size() == size() && "vector sizes must match");
        Vector r; r.resize(size());
        if (size() > 0) {
            if constexpr (kompute::kDeviceSupported<T>)
                kompute::vector_add<T>(tensor_, o.tensor_, r.tensor_);
            else
                for (int i = 0; i < size(); ++i) r.data()[i] = data()[i] + o.data()[i];
        }
        return r;
    }
    Vector operator-(const Vector& o) const {
        assert(o.size() == size() && "vector sizes must match");
        Vector r; r.resize(size());
        if (size() > 0) {
            if constexpr (kompute::kDeviceSupported<T>)
                kompute::vector_sub<T>(tensor_, o.tensor_, r.tensor_);
            else
                for (int i = 0; i < size(); ++i) r.data()[i] = data()[i] - o.data()[i];
        }
        return r;
    }
    Vector& operator+=(const Vector& o) {
        assert(o.size() == size() && "vector sizes must match");
        if (size() > 0) {
            if constexpr (kompute::kDeviceSupported<T>)
                kompute::vector_add<T>(tensor_, o.tensor_, tensor_);
            else
                for (int i = 0; i < size(); ++i) data()[i] += o.data()[i];
        }
        return *this;
    }
    Vector& operator-=(const Vector& o) {
        assert(o.size() == size() && "vector sizes must match");
        if (size() > 0) {
            if constexpr (kompute::kDeviceSupported<T>)
                kompute::vector_sub<T>(tensor_, o.tensor_, tensor_);
            else
                for (int i = 0; i < size(); ++i) data()[i] -= o.data()[i];
        }
        return *this;
    }
    Vector& operator*=(T s) {
        if (size() > 0) {
            if constexpr (kompute::kDeviceSupported<T>)
                kompute::vector_scale<T>(s, tensor_, tensor_);
            else
                for (int i = 0; i < size(); ++i) data()[i] *= s;
        }
        return *this;
    }
    Vector& operator/=(T s) {
        return *this *= static_cast<T>(T{1} / s);
    }
    Vector operator-() const {
        Vector r(*this);
        if (size() > 0) {
            if constexpr (kompute::kDeviceSupported<T>)
                kompute::vector_scale<T>(T{-1}, tensor_, r.tensor_);
            else
                for (int i = 0; i < size(); ++i) r.data()[i] = -data()[i];
        }
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
        if (size() > 0) {
            if constexpr (kompute::kDeviceSupported<T>)
                kompute::vector_scale<T>(s, tensor_, r.tensor_);
            else
                for (int i = 0; i < size(); ++i) r.data()[i] = data()[i] * s;
        }
        return r;
    }
    Vector operator/(T s) const {
        return *this * static_cast<T>(T{1} / s);
    }

private:
    kompute::TensorPtr<T> tensor_;
    size_type n_ = 0;

    void alloc_() {
        if (n_ > 0) tensor_ = kompute::make_tensor<T>(n_);
    }
    // (Re)allocate for n_ elements and deep-copy from o. Fixed-size copy
    // assignment reuses the existing allocation (same length); dynamic
    // frees and reallocates at o's length.
    void deepCopy_(const Vector& o) {
        const bool needAlloc = kDynamic || !tensor_;
        if (kDynamic && tensor_) tensor_ = nullptr;
        n_ = kDynamic ? o.n_ : n_;
        if (needAlloc && n_ > 0) alloc_();
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) data()[i] = o.data()[i];
        for (int i = m; i < size(); ++i) data()[i] = T{0};
    }
};

// ---- kernel-backed free operations (Kompute) ----
template<typename T, int Size>
T vectorNormKompute(const Vector<T, Size, Backend::Kompute>& v) { return v.norm(); }

template<typename T, int Size, Backend B1, Backend B2>
T vectorDotKompute(const Vector<T, Size, B1>& v1, const Vector<T, Size, B2>& v2) {
    return v1.dot(v2);
}

template<typename T, int Size>
Vector<T, Size, Backend::Kompute> vectorAddKompute(
    const Vector<T, Size, Backend::Kompute>& a,
    const Vector<T, Size, Backend::Kompute>& b) {
    return a + b;
}
template<typename T, int Size>
Vector<T, Size, Backend::Kompute> vectorSubKompute(
    const Vector<T, Size, Backend::Kompute>& a,
    const Vector<T, Size, Backend::Kompute>& b) {
    return a - b;
}
template<typename T, int Size>
Vector<T, Size, Backend::Kompute> vectorScaleKompute(
    T s, const Vector<T, Size, Backend::Kompute>& a) {
    return a * s;
}

} // namespace hlcl

#else // HLCL_KOMPUTE_ENABLED

// Kompute support disabled: Backend::Kompute falls back to the generic (host)
// Vector primary template, so no specialization is declared here.

#endif // HLCL_KOMPUTE_ENABLED
