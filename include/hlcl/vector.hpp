#pragma once

#include "backend_traits.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace hlcl {

// 统一三后端向量模板 (CPU / GPU(SYCL) / Kompute(Vulkan))。
// 存储与算子分派经 BackendTraits<B>; 逐元素算子按后端分派 (GPU/Kompute 走
// 各自内核, 数值与 CPU 参考一致), 归约 (norm/dot/sum) 恒为主机循环——三者
// 存储均为宿主可读, 与历史行为一致。
template<typename T, int Size, Backend B = Backend::CPU>
class Vector {
    static_assert(Size != 0, "Vector Size must be positive (fixed) or negative (dynamic)");
    using Traits = BackendTraits<B>;
    using Store = std::conditional_t<Size < 0,
                    typename Traits::template Storage<T, 0>,
                    typename Traits::template Storage<T, Size>>;

public:
    using value_type = T;
    using size_type = int;
    static constexpr int kSize = Size;
    static constexpr Backend kBackend = B;
    static constexpr bool kDynamic = (Size < 0);

    Vector() = default;   // 固定: 零向量; 动态: 空

    // 大小构造 (固定大小忽略参数; 动态大小 resize)
    explicit Vector(size_type n) {
        if constexpr (kDynamic) resize(n);
    }

    // 初始化列表 (非 explicit: `Vec<3> v = {1,2,3};` 合法)
    Vector(std::initializer_list<T> init) {
        if constexpr (kDynamic) {
            resize(static_cast<size_type>(init.size()));
        } else {
            assert(init.size() == static_cast<std::size_t>(Size) &&
                   "initializer list size must match fixed Vector size");
        }
        size_type i = 0;
        for (const T& v : init) st_.data()[i++] = v;
    }

    Vector(const Vector&) = default;
    Vector& operator=(const Vector&) = default;
    Vector(Vector&&) noexcept = default;
    Vector& operator=(Vector&&) noexcept = default;

    // 从其它后端 / 其它元素类型转换拷贝 (与 Matrix 对称)
    template<typename U, int Size2, Backend B2>
    Vector(const Vector<U, Size2, B2>& o) {
        if constexpr (kDynamic) resize(o.size());
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) at(i) = static_cast<T>(o[i]);
    }
    template<typename U, int Size2, Backend B2>
    Vector& operator=(const Vector<U, Size2, B2>& o) {
        if constexpr (kDynamic) resize(o.size());
        else setZero();
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) at(i) = static_cast<T>(o[i]);
        return *this;
    }

    size_type size() const {
        return kDynamic ? static_cast<size_type>(st_.size()) : Size;
    }

    T& at(size_type i) {
        assert(i >= 0 && i < size() && "index out of range");
        return st_.data()[i];
    }
    const T& at(size_type i) const {
        assert(i >= 0 && i < size() && "index out of range");
        return st_.data()[i];
    }

    T& operator()(size_type i) { return at(i); }
    const T& operator()(size_type i) const { return at(i); }

    T& operator[](size_type i) { return at(i); }
    const T& operator[](size_type i) const { return at(i); }

    T& coeffRef(size_type i) { return at(i); }
    T coeff(size_type i) const { return at(i); }

    // 连续数据访问 (CPU 固定: std::array; CPU 动态: std::vector;
    // GPU: shared-USM; Kompute: eHost 张量)
    T* data() { return st_.data(); }
    const T* data() const { return st_.data(); }

    // 分量访问器 (x/y/z), 供引擎与测试使用
    T x() const { return at(0); }
    T y() const { return at(1); }
    T z() const { return at(2); }

    void setZero() {
        for (int i = 0; i < size(); ++i) st_.data()[i] = T{0};
    }
    void setConstant(const T& value) {
        for (int i = 0; i < size(); ++i) st_.data()[i] = value;
    }

    // 动态: 调整长度, 保留前缀, 新增尾部清零; 固定: 无操作
    void resize(size_type n) {
        if constexpr (kDynamic) st_.resize(static_cast<std::size_t>(n));
    }

    constexpr T sum() const {
        T result = T{0};
        for (size_type i = 0; i < size(); ++i) result += at(i);
        return result;
    }

    constexpr T maxCoeff() const {
        assert(size() > 0 && "maxCoeff on empty vector");
        T result = at(0);
        for (size_type i = 1; i < size(); ++i) result = std::max(result, at(i));
        return result;
    }

    constexpr T minCoeff() const {
        assert(size() > 0 && "minCoeff on empty vector");
        T result = at(0);
        for (size_type i = 1; i < size(); ++i) result = std::min(result, at(i));
        return result;
    }

    constexpr T squaredNorm() const {
        T result = T{0};
        for (size_type i = 0; i < size(); ++i) result += at(i) * at(i);
        return result;
    }

    T norm() const { return static_cast<T>(std::sqrt(squaredNorm())); }

    // 点积 (成员形式; 支持跨后端/跨元素类型)
    template<typename U, int Size2, Backend B2>
    constexpr T dot(const Vector<U, Size2, B2>& other) const {
        assert(other.size() == size() && "dot product requires equal sizes");
        T result = T{0};
        for (size_type i = 0; i < size(); ++i) result += at(i) * static_cast<T>(other[i]);
        return result;
    }

    // ---- 成员逐元素运算 (后端分派) ----
    [[nodiscard]] Vector operator+(const Vector& other) const {
        assert(other.size() == size() && "vector sizes must match");
        Vector r;
        if constexpr (kDynamic) r.resize(size());
        if (size() > 0) Traits::add(data(), other.data(), r.data(), size());
        return r;
    }
    [[nodiscard]] Vector operator-(const Vector& other) const {
        assert(other.size() == size() && "vector sizes must match");
        Vector r;
        if constexpr (kDynamic) r.resize(size());
        if (size() > 0) Traits::sub(data(), other.data(), r.data(), size());
        return r;
    }
    Vector& operator+=(const Vector& other) {
        assert(other.size() == size() && "vector sizes must match");
        if (size() > 0) Traits::add(data(), other.data(), data(), size());
        return *this;
    }
    Vector& operator-=(const Vector& other) {
        assert(other.size() == size() && "vector sizes must match");
        if (size() > 0) Traits::sub(data(), other.data(), data(), size());
        return *this;
    }
    Vector& operator*=(T scalar) {
        if (size() > 0) Traits::scale(scalar, data(), data(), size());
        return *this;
    }
    Vector& operator/=(T scalar) {
        if (size() > 0) Traits::div(scalar, data(), data(), size());
        return *this;
    }
    [[nodiscard]] Vector operator-() const {
        Vector r;
        if constexpr (kDynamic) r.resize(size());
        if (size() > 0) Traits::negate(data(), r.data(), size());
        return r;
    }
    [[nodiscard]] Vector operator+(T scalar) const {
        Vector r(*this);
        for (size_type i = 0; i < size(); ++i) r[i] += scalar;
        return r;
    }
    [[nodiscard]] Vector operator-(T scalar) const {
        Vector r(*this);
        for (size_type i = 0; i < size(); ++i) r[i] -= scalar;
        return r;
    }
    [[nodiscard]] Vector operator*(T scalar) const {
        Vector r;
        if constexpr (kDynamic) r.resize(size());
        if (size() > 0) Traits::scale(scalar, data(), r.data(), size());
        return r;
    }
    [[nodiscard]] Vector operator/(T scalar) const {
        Vector r;
        if constexpr (kDynamic) r.resize(size());
        if (size() > 0) Traits::div(scalar, data(), r.data(), size());
        return r;
    }

private:
    Store st_;
};

// ---- 自由函数: 点积 / 标量(前置)运算 / 范数 ----

// v1 * v2 点积 (Eigen 风格运算符重载)
template<typename T, typename U, int Size, Backend B>
[[nodiscard]] constexpr T operator*(const Vector<T, Size, B>& lhs, const Vector<U, Size, B>& rhs) {
    return lhs.dot(rhs);
}

// 标量前置运算 (标量后置形式由成员函数提供, 避免重载歧义)
template<typename T, int Size, Backend B>
[[nodiscard]] Vector<T, Size, B> operator+(T scalar, const Vector<T, Size, B>& v) {
    return v + scalar;
}

template<typename T, int Size, Backend B>
[[nodiscard]] Vector<T, Size, B> operator-(T scalar, const Vector<T, Size, B>& v) {
    Vector<T, Size, B> r;
    if constexpr (Size < 0) r.resize(v.size());
    for (int i = 0; i < v.size(); ++i) r[i] = scalar - v[i];
    return r;
}

template<typename T, int Size, Backend B>
[[nodiscard]] Vector<T, Size, B> operator*(T scalar, const Vector<T, Size, B>& v) {
    return v * scalar;
}

template<typename T, int Size, Backend B>
[[nodiscard]] T norm(const Vector<T, Size, B>& v) {
    return v.norm();
}

// ---- 类型别名: 默认使用单精度浮点 ----
template<int Size, Backend B = Backend::CPU>
using Vec = Vector<Float32, Size, B>;

template<int Size, Backend B = Backend::CPU>
using Vecd = Vector<Float64, Size, B>;

// 常用 2/3 维别名
using Vec2 = Vec<2, Backend::CPU>;
using Vec3 = Vec<3, Backend::CPU>;

} // namespace hlcl

// 后端 traits 特化 (GPU/Kompute) 在启用对应宏时引入。
#ifdef HLCL_GPU_ENABLED
#include "gpu_impl.hpp"
#endif // HLCL_GPU_ENABLED

#ifdef HLCL_KOMPUTE_ENABLED
#include "kompute_impl.hpp"
#endif // HLCL_KOMPUTE_ENABLED
