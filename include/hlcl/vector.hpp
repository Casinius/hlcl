#pragma once

#include "backend.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace hlcl {

// 固定大小动态存储向量 (Eigen 风格 API 的简化实现)。
// Size >= 0: 编译期固定大小, 数据内联于 std::array (零堆分配, 值语义快)。
// Size  < 0: 动态大小, 默认构造为空, 通过 resize(n) 调整 (std::vector)。
template<typename T, int Size, Backend B = Backend::CPU>
class Vector {
public:
    using value_type = T;
    using size_type = int;
    static constexpr int kSize = Size;
    static constexpr Backend kBackend = B;
    static constexpr bool kDynamic = (Size < 0);
    static constexpr std::size_t kCapacity =
        (Size > 0) ? static_cast<std::size_t>(Size) : static_cast<std::size_t>(1);

    // 默认构造: 固定大小 -> 零向量; 动态大小 -> 空
    Vector() : n_(Size > 0 ? Size : 0) {
        store_.fill(T{0});
    }

    // 大小构造 (固定大小 Vector 忽略参数, 零填充; 动态大小则 resize)
    explicit Vector(size_type n) : n_(Size > 0 ? Size : n) {
        if (kDynamic) dyn_.assign(static_cast<std::size_t>(n), T{0});
        else store_.fill(T{0});
    }

    // 初始化列表 (非 explicit: `Vec<3> v = {1,2,3};` 合法)
    Vector(std::initializer_list<T> init) : n_(static_cast<size_type>(init.size())) {
        if (kDynamic) {
            dyn_.assign(init.begin(), init.end());
        } else {
            assert(init.size() == static_cast<std::size_t>(Size) &&
                   "initializer list size must match fixed Vector size");
            n_ = Size;
            size_type i = 0;
            for (const T& v : init) store_[static_cast<std::size_t>(i++)] = v;
        }
    }

    // 从其它后端 / 其它元素类型转换拷贝 (与 Matrix 对称)
    template<typename U, int Size2, Backend B2>
    Vector(const Vector<U, Size2, B2>& o) : n_(Size > 0 ? Size : o.size()) {
        if (kDynamic) dyn_.assign(static_cast<std::size_t>(o.size()), T{0});
        else store_.fill(T{0});
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) at(i) = static_cast<T>(o[i]);
    }
    template<typename U, int Size2, Backend B2>
    Vector& operator=(const Vector<U, Size2, B2>& o) {
        if (kDynamic) {
            dyn_.assign(static_cast<std::size_t>(o.size()), T{0});
            n_ = o.size();
        } else {
            store_.fill(T{0});
        }
        const int m = std::min(size(), o.size());
        for (int i = 0; i < m; ++i) at(i) = static_cast<T>(o[i]);
        return *this;
    }

    size_type size() const { return kDynamic ? static_cast<size_type>(dyn_.size()) : n_; }

    T& at(size_type i) {
        assert(i >= 0 && i < size() && "index out of range");
        return kDynamic ? dyn_[static_cast<std::size_t>(i)]
                        : store_[static_cast<std::size_t>(i)];
    }
    const T& at(size_type i) const {
        assert(i >= 0 && i < size() && "index out of range");
        return kDynamic ? dyn_[static_cast<std::size_t>(i)]
                        : store_[static_cast<std::size_t>(i)];
    }

    T& operator()(size_type i) { return at(i); }
    const T& operator()(size_type i) const { return at(i); }

    T& operator[](size_type i) { return at(i); }
    const T& operator[](size_type i) const { return at(i); }

    T& coeffRef(size_type i) { return at(i); }
    T coeff(size_type i) const { return at(i); }

    // 连续数据访问 (固定大小 std::array / 动态 std::vector)
    T* data() { return kDynamic ? dyn_.data() : store_.data(); }
    const T* data() const { return kDynamic ? dyn_.data() : store_.data(); }

    // 分量访问器 (x/y/z), 供引擎与测试使用
    T x() const { return at(0); }
    T y() const { return at(1); }
    T z() const { return at(2); }

    void setZero() {
        if (kDynamic) std::fill(dyn_.begin(), dyn_.end(), T{0});
        else store_.fill(T{0});
    }
    void setConstant(const T& value) {
        if (kDynamic) std::fill(dyn_.begin(), dyn_.end(), value);
        else store_.fill(value);
    }

    void resize(size_type n) {
        if (kDynamic) {
            dyn_.assign(static_cast<std::size_t>(n), T{0});
            n_ = n;
        }
    }

    T sum() const {
        T result = T{0};
        for (size_type i = 0; i < size(); ++i) result += at(i);
        return result;
    }

    T maxCoeff() const {
        assert(size() > 0 && "maxCoeff on empty vector");
        T result = at(0);
        for (size_type i = 1; i < size(); ++i) result = std::max(result, at(i));
        return result;
    }

    T minCoeff() const {
        assert(size() > 0 && "minCoeff on empty vector");
        T result = at(0);
        for (size_type i = 1; i < size(); ++i) result = std::min(result, at(i));
        return result;
    }

    T squaredNorm() const {
        T result = T{0};
        for (size_type i = 0; i < size(); ++i) result += at(i) * at(i);
        return result;
    }

    T norm() const { return static_cast<T>(std::sqrt(squaredNorm())); }

    // 点积 (成员形式)
    template<typename U, int Size2, Backend B2>
    T dot(const Vector<U, Size2, B2>& other) const {
        assert(other.size() == size() && "dot product requires equal sizes");
        T result = T{0};
        for (size_type i = 0; i < size(); ++i) result += at(i) * static_cast<T>(other[i]);
        return result;
    }

    // ---- 成员逐元素运算 ----
    Vector operator+(const Vector& other) const {
        assert(other.size() == size() && "vector sizes must match");
        Vector result;
        for (size_type i = 0; i < size(); ++i) result[i] = at(i) + other[i];
        return result;
    }
    Vector operator-(const Vector& other) const {
        assert(other.size() == size() && "vector sizes must match");
        Vector result;
        for (size_type i = 0; i < size(); ++i) result[i] = at(i) - other[i];
        return result;
    }
    Vector& operator+=(const Vector& other) {
        assert(other.size() == size() && "vector sizes must match");
        for (size_type i = 0; i < size(); ++i) at(i) += other[i];
        return *this;
    }
    Vector& operator-=(const Vector& other) {
        assert(other.size() == size() && "vector sizes must match");
        for (size_type i = 0; i < size(); ++i) at(i) -= other[i];
        return *this;
    }
    Vector& operator*=(T scalar) {
        for (size_type i = 0; i < size(); ++i) at(i) *= scalar;
        return *this;
    }
    Vector& operator/=(T scalar) {
        for (size_type i = 0; i < size(); ++i) at(i) /= scalar;
        return *this;
    }
    Vector operator-() const {
        Vector result;
        for (size_type i = 0; i < size(); ++i) result[i] = -at(i);
        return result;
    }
    Vector operator+(T scalar) const {
        Vector result;
        for (size_type i = 0; i < size(); ++i) result[i] = at(i) + scalar;
        return result;
    }
    Vector operator-(T scalar) const {
        Vector result;
        for (size_type i = 0; i < size(); ++i) result[i] = at(i) - scalar;
        return result;
    }
    Vector operator*(T scalar) const {
        Vector result;
        for (size_type i = 0; i < size(); ++i) result[i] = at(i) * scalar;
        return result;
    }
    Vector operator/(T scalar) const {
        Vector result;
        for (size_type i = 0; i < size(); ++i) result[i] = at(i) / scalar;
        return result;
    }

private:
    std::array<T, kCapacity> store_;
    std::vector<T> dyn_; // 仅 Size < 0 时使用
    size_type n_;
};

// ---- 自由函数: 点积 / 标量(前置)运算 / 范数 ----

// v1 * v2 点积 (Eigen 风格运算符重载)
template<typename T, typename U, int Size, Backend B>
T operator*(const Vector<T, Size, B>& lhs, const Vector<U, Size, B>& rhs) {
    T result = T{0};
    for (int i = 0; i < lhs.size(); ++i) result += lhs[i] * static_cast<T>(rhs[i]);
    return result;
}

// 标量前置运算 (标量后置形式由成员函数提供, 避免重载歧义)
template<typename T, int Size, Backend B>
Vector<T, Size, B> operator+(T scalar, const Vector<T, Size, B>& v) {
    Vector<T, Size, B> result(v);
    for (int i = 0; i < result.size(); ++i) result[i] = scalar + result[i];
    return result;
}

template<typename T, int Size, Backend B>
Vector<T, Size, B> operator-(T scalar, const Vector<T, Size, B>& v) {
    Vector<T, Size, B> result(v);
    for (int i = 0; i < result.size(); ++i) result[i] = scalar - result[i];
    return result;
}

template<typename T, int Size, Backend B>
Vector<T, Size, B> operator*(T scalar, const Vector<T, Size, B>& v) {
    Vector<T, Size, B> result(v);
    for (int i = 0; i < result.size(); ++i) result[i] = scalar * result[i];
    return result;
}

template<typename T, int Size, Backend B>
T norm(const Vector<T, Size, B>& v) {
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

// 主模板定义完成后才引入 GPU 偏特化 (偏特化要求主模板已声明)。
#ifdef HLCL_GPU_ENABLED
#include "vector_gpu.hpp"
#endif // HLCL_GPU_ENABLED

// Kompute (Vulkan compute) 偏特化, 同样要求主模板已声明。
#ifdef HLCL_KOMPUTE_ENABLED
#include "vector_kompute.hpp"
#endif // HLCL_KOMPUTE_ENABLED
