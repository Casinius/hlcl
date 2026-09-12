#pragma once

// Kompute (Vulkan compute) kernel layer for AVBD.
//
// Design notes
// ------------
// * Header-only by intent (mirrors avbd_core): this header only calls the
//   Kompute API; the implementations live in the kompute package library
//   (v0.8.0 — the last release with kp::TensorT + Sequence::eval API).
// * Mirrors the SYCL gpu_impl.hpp surface 1:1: same kernel names/signatures,
//   a process-wide Manager singleton (intentionally leaked, same rationale as
//   the SYCL queue leak — Vulkan teardown at exit is fragile), and synchronous
//   kernels (Sequence::eval blocks on its fence).
// * Storage contract differs from SYCL USM: tensors are kp::TensorT<T> in
//   eHost mode (host-visible coherent memory), so data() is a plain CPU
//   pointer — the Matrix/Vector specializations keep tensors, and small ops
//   run on the host exactly like the GPU specialization does.
// * Float only on device: Kompute 0.8.0 does not enable the shaderFloat64
//   device feature, so double instantiations of the kernel wrappers fall
//   back to host computation over the same tensors (correct, just not
//   accelerated). GLSL kernels live in core/shaders/*.comp; the compiled
//   SPIR-V words are checked in as kompute_shaders.hpp and can be
//   regenerated with `xmake run kompute_shaders`.
// * Sizes travel through a tiny int params tensor bound as an extra SSBO —
//   avoids Kompute 0.8 push-constant plumbing entirely.

#ifdef HLCL_KOMPUTE_ENABLED

#include <kompute/Kompute.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kompute_shaders.hpp"

namespace hlcl {

namespace kompute {

/// Tensor handle: eHost tensor with host-accessible data().
template<typename T>
using TensorPtr = std::shared_ptr<kp::TensorT<T>>;

/// Process-wide Manager (lazily created, intentionally leaked — see comment
/// in gpu_impl.hpp for the SYCL analogue; Vulkan teardown races are the same
/// class of exit-time hazard). Physical device index comes from
/// HLCL_KOMPUTE_DEVICE (0-based), default 0.
inline std::shared_ptr<kp::Manager>& manager() {
    static std::shared_ptr<kp::Manager>* m = [] {
        uint32_t index = 0;
        if (const char* e = std::getenv("HLCL_KOMPUTE_DEVICE")) index =
            static_cast<uint32_t>(std::atoi(e));
        return new std::shared_ptr<kp::Manager>(new kp::Manager(index));
    }();
    return *m;
}

/// eHost tensor of n zeros (host-visible coherent memory).
template<typename T>
inline TensorPtr<T> make_tensor(std::size_t n) {
    return manager()->tensorT<T>(std::vector<T>(n, T{0}),
        kp::Tensor::TensorTypes::eHost);
}

/// Float kernel support flag: device dispatch is float-only (see header note).
template<typename T>
constexpr bool kDeviceSupported = std::is_same<T, float>::value;

/// Small int params tensor: carries the element/edge counts for kernels.
inline TensorPtr<int> params(int n) {
    auto t = make_tensor<int>(1);
    t->data()[0] = n;
    return t;
}

/// Run one SPIR-V algorithm over `tensors`, synchronously.
inline void dispatch(const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
    const uint32_t* spirv, std::size_t words, const std::array<uint32_t, 3>& groups) {
    auto algo = manager()->algorithm(tensors,
        std::vector<uint32_t>(spirv, spirv + words),
        kp::Workgroup{groups[0], groups[1], groups[2]});
    manager()->sequence()->eval<kp::OpAlgoDispatch>(algo);
}

inline std::vector<std::shared_ptr<kp::Tensor>> as_tensors() { return {}; }

/// Upcast tensor handles to the base kp::Tensor for Algorithm binding.
template<typename T, typename... Rest>
inline std::vector<std::shared_ptr<kp::Tensor>> as_tensors(
    const TensorPtr<T>& first, const Rest&... rest) {
    auto v = as_tensors(rest...);
    v.insert(v.begin(), std::static_pointer_cast<kp::Tensor>(first));
    return v;
}

// ---------------------------------------------------------------------------
// Host fallbacks for non-float types: run the same math over tensor data().
// (Kompute 0.8.0 cannot enable shaderFloat64; keep results correct.)
// ---------------------------------------------------------------------------
template<typename T, int N>
inline void matrix_multiply_host(const T* A, const T* B, T* C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            T s = T{0};
            for (int k = 0; k < N; ++k) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

template<typename T, int N>
inline void matrix_inverse_host(const T* A, T* C) {
    T aug[N][2 * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            aug[i][j] = A[i * N + j];
            aug[i][N + j] = (i == j) ? T{1} : T{0};
        }
    for (int k = 0; k < N; ++k) {
        int p = k;
        T best = std::abs(aug[k][k]);
        for (int r = k + 1; r < N; ++r)
            if (std::abs(aug[r][k]) > best) { best = std::abs(aug[r][k]); p = r; }
        if (best == T{0}) {
            for (int i = 0; i < N * N; ++i) C[i] = T{0};
            return;
        }
        if (p != k)
            for (int j = 0; j < 2 * N; ++j) std::swap(aug[k][j], aug[p][j]);
        T piv = aug[k][k];
        for (int j = 0; j < 2 * N; ++j) aug[k][j] /= piv;
        for (int r = 0; r < N; ++r) {
            if (r == k) continue;
            T f = aug[r][k];
            for (int j = 0; j < 2 * N; ++j) aug[r][j] -= f * aug[k][j];
        }
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) C[i * N + j] = aug[i][N + j];
}

template<typename T, int N>
inline T matrix_determinant_host(const T* A) {
    T m[N * N];
    for (int i = 0; i < N * N; ++i) m[i] = A[i];
    T det = T{1};
    for (int k = 0; k < N; ++k) {
        int p = k;
        T best = std::abs(m[k * N + k]);
        for (int r = k + 1; r < N; ++r)
            if (std::abs(m[r * N + k]) > best) { best = std::abs(m[r * N + k]); p = r; }
        if (best == T{0}) return T{0};
        if (p != k) {
            for (int j = 0; j < N; ++j) std::swap(m[k * N + j], m[p * N + j]);
            det = -det;
        }
        det *= m[k * N + k];
        for (int r = k + 1; r < N; ++r) {
            T f = m[r * N + k] / m[k * N + k];
            for (int j = k; j < N; ++j) m[r * N + j] -= f * m[k * N + j];
        }
    }
    return det;
}

// ---------------------------------------------------------------------------
// Matrix kernels: C = A * B   (all row-major, N x N, float only)
// ---------------------------------------------------------------------------
template<typename T, int N>
inline void matrix_multiply(const TensorPtr<T>& A, const TensorPtr<T>& B,
    const TensorPtr<T>& C) {
    if (!kDeviceSupported<T>) return; // caller falls back to host
    const int n = N;
    const auto ts = as_tensors(A, B, C, params(n));
    dispatch(ts, KP_SPV_MATMUL, sizeof(KP_SPV_MATMUL) / sizeof(uint32_t),
        std::array<uint32_t, 3>{uint32_t((N + 15) / 16), uint32_t((N + 15) / 16), 1});
}

template<typename T, int N>
inline void matrix_multiply(const T* A, const T* B, T* C) {
    if (!kDeviceSupported<T>) {
        matrix_multiply_host<T, N>(A, B, C);
        return;
    }
    auto tA = make_tensor<T>(std::size_t(N) * N);
    auto tB = make_tensor<T>(std::size_t(N) * N);
    auto tC = make_tensor<T>(std::size_t(N) * N);
    std::memcpy(tA->data(), A, sizeof(T) * N * N);
    std::memcpy(tB->data(), B, sizeof(T) * N * N);
    matrix_multiply<T, N>(tA, tB, tC);
    std::memcpy(C, tC->data(), sizeof(T) * N * N);
}

// ---------------------------------------------------------------------------
// Matrix kernels: inverse (Gauss-Jordan, partial pivoting) — float only.
// Singular input writes a zero matrix, mirroring the CPU/GPU contract.
// ---------------------------------------------------------------------------
template<typename T, int N>
inline void matrix_inverse(const TensorPtr<T>& A, const TensorPtr<T>& C) {
    if (!kDeviceSupported<T>) return;
    const int n = N;
    const auto ts = as_tensors(A, C, params(n));
    dispatch(ts, KP_SPV_INVERSE, sizeof(KP_SPV_INVERSE) / sizeof(uint32_t),
        std::array<uint32_t, 3>{1, 1, 1});
}

template<typename T, int N>
inline void matrix_inverse(const T* A, T* C) {
    if (!kDeviceSupported<T>) {
        matrix_inverse_host<T, N>(A, C);
        return;
    }
    auto tA = make_tensor<T>(std::size_t(N) * N);
    auto tC = make_tensor<T>(std::size_t(N) * N);
    std::memcpy(tA->data(), A, sizeof(T) * N * N);
    matrix_inverse<T, N>(tA, tC);
    std::memcpy(C, tC->data(), sizeof(T) * N * N);
}

// ---------------------------------------------------------------------------
// Matrix kernel: determinant (single invocation) — float only. Returns T.
// ---------------------------------------------------------------------------
template<typename T, int N>
inline T matrix_determinant(const TensorPtr<T>& A) {
    if (!kDeviceSupported<T>) return T{0};
    auto out = make_tensor<T>(1);
    const int n = N;
    const auto ts = as_tensors(A, out, params(n));
    dispatch(ts, KP_SPV_DETERMINANT, sizeof(KP_SPV_DETERMINANT) / sizeof(uint32_t),
        std::array<uint32_t, 3>{1, 1, 1});
    return out->data()[0];
}

template<typename T, int N>
inline T matrix_determinant(const T* A) {
    if (!kDeviceSupported<T>) return matrix_determinant_host<T, N>(A);
    auto tA = make_tensor<T>(std::size_t(N) * N);
    std::memcpy(tA->data(), A, sizeof(T) * N * N);
    return matrix_determinant<T, N>(tA);
}

// ---------------------------------------------------------------------------
// Vector kernels (n elements, float only)
// ---------------------------------------------------------------------------
template<typename T>
inline void vector_add(const TensorPtr<T>& v1, const TensorPtr<T>& v2,
    const TensorPtr<T>& out) {
    if (!kDeviceSupported<T>) return;
    const int n = static_cast<int>(v1->size());
    const auto ts = as_tensors(v1, v2, out, params(n));
    dispatch(ts, KP_SPV_VEC_ADD, sizeof(KP_SPV_VEC_ADD) / sizeof(uint32_t),
        std::array<uint32_t, 3>{uint32_t((n + 63) / 64), 1, 1});
}

template<typename T>
inline void vector_sub(const TensorPtr<T>& a, const TensorPtr<T>& b,
    const TensorPtr<T>& out) {
    if (!kDeviceSupported<T>) return;
    const int n = static_cast<int>(a->size());
    const auto ts = as_tensors(a, b, out, params(n));
    dispatch(ts, KP_SPV_VEC_SUB, sizeof(KP_SPV_VEC_SUB) / sizeof(uint32_t),
        std::array<uint32_t, 3>{uint32_t((n + 63) / 64), 1, 1});
}

template<typename T>
inline void vector_scale(T scalar, const TensorPtr<T>& v, const TensorPtr<T>& out) {
    if (!kDeviceSupported<T>) return;
    const int n = static_cast<int>(v->size());
    auto sn = make_tensor<int>(1);
    auto ss = make_tensor<float>(1);
    sn->data()[0] = n;
    ss->data()[0] = static_cast<float>(scalar);
    const auto ts = as_tensors(v, out, sn, ss);
    dispatch(ts, KP_SPV_VEC_SCALE, sizeof(KP_SPV_VEC_SCALE) / sizeof(uint32_t),
        std::array<uint32_t, 3>{uint32_t((n + 63) / 64), 1, 1});
}

/// out = v / ||v||; zero vector -> zeros (two dispatches, ordered by the
/// same command queue semantics as the SYCL in-order queue).
template<typename T>
inline void vector_normalize(const TensorPtr<T>& v, const TensorPtr<T>& out) {
    if (!kDeviceSupported<T>) return;
    const int n = static_cast<int>(v->size());
    auto norm = make_tensor<float>(1);
    {
        const auto ts = as_tensors(v, norm, params(n));
        dispatch(ts, KP_SPV_VEC_NORM, sizeof(KP_SPV_VEC_NORM) / sizeof(uint32_t),
            std::array<uint32_t, 3>{1, 1, 1});
    }
    {
        const auto ts = as_tensors(v, norm, out, params(n));
        dispatch(ts, KP_SPV_VEC_DIVIDE, sizeof(KP_SPV_VEC_DIVIDE) / sizeof(uint32_t),
            {uint32_t((n + 63) / 64), 1, 1});
    }
}

} // namespace kompute

/// Public surface, mirroring gpu_impl_init/cleanup/get_device_info.
inline void kompute_impl_init() { (void)kompute::manager(); }
inline void kompute_impl_cleanup() {}

struct KomputeDeviceInfo {
    std::string name;
    std::string vendor;
    size_t global_memory;
    size_t max_compute_units;
    bool float_device; // device kernel dispatch is float-only
};

inline KomputeDeviceInfo get_kompute_device_info() {
    KomputeDeviceInfo info{};
    try {
        const vk::PhysicalDeviceProperties props =
            kompute::manager()->getDeviceProperties();
        info.name = std::string(props.deviceName.data());
        info.vendor = std::to_string(props.vendorID);
        info.global_memory = 0; // 0.8.0 无内存堆查询接口
        info.max_compute_units = props.limits.maxComputeWorkGroupInvocations;
        info.float_device = true;
    } catch (const std::exception&) {
        info.float_device = false;
    }
    return info;
}

inline void print_kompute_device_info() {
    const auto info = get_kompute_device_info();
    std::fprintf(stderr, "Kompute device: %s (vendor %s)\n",
        info.name.c_str(), info.vendor.c_str());
    std::fprintf(stderr, "  compute invocations/workgroup: %zu, float-only kernels: %s\n",
        info.max_compute_units, info.float_device ? "yes" : "no");
}

} // namespace hlcl

#else // !HLCL_KOMPUTE_ENABLED

// CPU-only stubs so translation units may include this header unconditionally
// (mirrors gpu_impl.hpp's stub branch).
#include <string>

namespace hlcl {

inline void kompute_impl_init() {}
inline void kompute_impl_cleanup() {}

struct KomputeDeviceInfo {
    std::string name;
    std::string vendor;
    size_t global_memory;
    size_t max_compute_units;
    bool float_device;
};

inline KomputeDeviceInfo get_kompute_device_info() { return {}; }

inline void print_kompute_device_info() {
    std::fprintf(stderr, "Kompute backend disabled (built without HLCL_KOMPUTE_ENABLED)\n");
}

} // namespace hlcl

#endif // HLCL_KOMPUTE_ENABLED
