#pragma once

// 后端统一入口: 存储 (Storage) 与纯主机指针算子 (ops)。
//
// Vector / Matrix 主模板只经 data() 指针与 BackendTraits<B> 调度, 不直接
// 依赖任何后端实现。三份显式特化:
//   * Backend::CPU     — 本文件 (常驻, 主机参考实现)
//   * Backend::GPU     — gpu_impl.hpp     (HLCL_GPU_ENABLED)
//   * Backend::Kompute — kompute_impl.hpp (HLCL_KOMPUTE_ENABLED)
//
// Storage<T, Cap>: Cap>0 为编译期固定容量 (零堆分配或单次分配), Cap==0 为
// 动态容量; 动态 resize 语义统一为 "保留前缀、新增尾部清零"。
// ops 全部面向原始指针, 供主模板以 data() 直接驱动; 各后端保证同一算子
// 数值结果与 CPU 参考一致 (尤其除法为真除法, 非乘倒数)。

#include "backend.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace hlcl {

template<Backend B>
struct BackendTraits;   // 仅声明; 特化见上

// ------------------------- CPU (宿主机) -------------------------
template<>
struct BackendTraits<Backend::CPU> {
    // 编译期固定容量: 内联 std::array, 零堆分配
    template<typename T, int Cap>
    class Storage {
        static_assert(Cap > 0, "fixed storage requires Cap > 0");
    public:
        std::size_t size() const { return static_cast<std::size_t>(Cap); }
        T* data() { return d_.data(); }
        const T* data() const { return d_.data(); }

    private:
        std::array<T, static_cast<std::size_t>(Cap)> d_{};   // 零初始化
    };

    // 动态容量: std::vector; resize 保留前缀, 新元素值初始化为 0
    template<typename T>
    class Storage<T, 0> {
    public:
        std::size_t size() const { return d_.size(); }
        T* data() { return d_.data(); }
        const T* data() const { return d_.data(); }
        void resize(std::size_t n) { d_.resize(n); }

    private:
        std::vector<T> d_;
    };

    // ---- 算子 (主机循环, 参考实现) ----
    template<typename T>
    static void add(const T* a, const T* b, T* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }
    template<typename T>
    static void sub(const T* a, const T* b, T* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
    }
    template<typename T>
    static void scale(T s, const T* v, T* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = s * v[i];
    }
    template<typename T>
    static void div(T s, const T* v, T* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = v[i] / s;
    }
    template<typename T>
    static void negate(const T* v, T* out, int n) {
        for (int i = 0; i < n; ++i) out[i] = -v[i];
    }
    template<typename T, int N>
    static void mat_mul(const T* a, const T* b, T* c) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                T s = T{0};
                for (int k = 0; k < N; ++k) s += a[i * N + k] * b[k * N + j];
                c[i * N + j] = s;
            }
    }
};

} // namespace hlcl
