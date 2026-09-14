#pragma once

// GPU (SYCL 2020) kernel layer for hlcl.
//
// Design notes
// ------------
// * Header-only by intent (mirrors the CPU core). All kernels are inline so a
//   GPU-mode translation unit gets them without linking an orphaned .cpp.
// * SYCL 2020 interface: <sycl/sycl.hpp>, sycl::*_selector_v, sycl::reduction.
//   No deprecated SYCL 1.2.1 API (cl::sycl namespace, *_selector classes).
// * Backend selection is OpenCL-portable: it prefers an accelerator (GPU),
//   honours the HLCL_SYCL_DEVICE env var (gpu|cpu), and otherwise falls back
//   to whatever SYCL device the runtime exposes (AdaptiveCpp OpenMP host,
//   an OpenCL device, CUDA, ...). The SYCL source itself is backend agnostic.
// * Every kernel is deliberately tiny and self-contained. Mathematically they
//   replicate the CPU (Eigen-like) reference in core; correctness is asserted
//   by the GPU tests comparing kernel output against the CPU path.
// * Kernels take flat row-major std::span arguments whose storage MUST be
//   shared-USM allocations (sycl::malloc_shared; on the AdaptiveCpp OpenMP
//   host this is equivalent to plain malloc). Kernels read/write them in
//   place: no copies, no transient SYCL buffers, no accessors.

#ifdef HLCL_GPU_ENABLED

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hlcl {

namespace gpu {

/// Pick the SYCL device: honour HLCL_SYCL_DEVICE (gpu|cpu), else prefer a GPU
/// accelerator, else the runtime default (CPU/OpenMP host in CPU-only builds).
inline sycl::device select_device() {
    if (const char* e = std::getenv("HLCL_SYCL_DEVICE")) {
        const std::string_view sel = e;
        try {
            if (sel == "cpu") return sycl::device{sycl::cpu_selector_v};
            if (sel == "gpu") return sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::exception&) {
            // requested device class unavailable -> fall through to default
        }
    }
    try {
        const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) return devices.front();
    } catch (...) { /* fall through */ }
    return sycl::device{}; // runtime default selector semantics
}

/// Process-wide shared queue (lazily created).
///
/// Intentionally leaked (heap-allocated, never destroyed): AdaptiveCpp 25.10
/// can segfault at process exit when a process-wide SYCL queue is destroyed
/// (its runtime teardown races the OpenMP worker threads of prior kernels).
/// Results are fully correct; only the exit-time destructor path is unsafe.
inline sycl::queue& queue() {
    static sycl::queue* q = new sycl::queue(
        select_device(), sycl::property::queue::in_order());
    return *q;
}

// ---------------------------------------------------------------------------
// Matrix kernels: C = A * B   (all row-major, N x N)
// ---------------------------------------------------------------------------
template<typename T, Index N>
inline void matrix_multiply(std::span<const T> A, std::span<const T> B,
                            std::span<T> C) {
    const T* pA = A.data();
    const T* pB = B.data();
    T* pC = C.data();
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> idx) {
            const Index i = static_cast<Index>(idx[0]);
            const Index j = static_cast<Index>(idx[1]);
            T s = T{0};
            for (Index k = 0; k < N; ++k) s += pA[i * N + k] * pB[k * N + j];
            pC[i * N + j] = s;
        });
    });
    queue().wait();
}

// ---------------------------------------------------------------------------
// Matrix inverse via Gauss-Jordan elimination (single work item).
// Singular matrices produce a zero matrix (matches CPU inverse() contract).
// ---------------------------------------------------------------------------
template<typename T, Index N>
inline void matrix_inverse(std::span<const T> A_in, std::span<T> C_out) {
    const T* pA = A_in.data();
    T* pC = C_out.data();
    queue().submit([&](sycl::handler& h) {
        h.single_task([=]() {
            T m[N * N];
            for (Index i = 0; i < N * N; ++i) m[i] = pA[i];
            T inv[N * N];
            for (Index i = 0; i < N; ++i)
                for (Index j = 0; j < N; ++j) inv[i * N + j] = (i == j) ? T{1} : T{0};
            for (Index col = 0; col < N; ++col) {
                std::size_t piv = col;
                T mx = std::abs(m[col * N + col]);
                for (Index r = col + 1; r < N; ++r) {
                    const T v = std::abs(m[r * N + col]);
                    if (v > mx) { mx = v; piv = r; }
                }
                if (mx == T{0}) {
                    for (Index i = 0; i < N * N; ++i) pC[i] = T{0};
                    return; // singular
                }
                if (piv != col) {
                    for (Index j = 0; j < N; ++j) {
                        T t = m[col * N + j]; m[col * N + j] = m[piv * N + j]; m[piv * N + j] = t;
                        t = inv[col * N + j]; inv[col * N + j] = inv[piv * N + j]; inv[piv * N + j] = t;
                    }
                }
                const T d = m[col * N + col];
                for (Index j = 0; j < N; ++j) { m[col * N + j] /= d; inv[col * N + j] /= d; }
                for (Index r = 0; r < N; ++r) {
                    if (r == col) continue;
                    const T f = m[r * N + col];
                    for (Index j = 0; j < N; ++j) {
                        m[r * N + j] -= f * m[col * N + j];
                        inv[r * N + j] -= f * inv[col * N + j];
                    }
                }
            }
            for (Index i = 0; i < N * N; ++i) pC[i] = inv[i];
        });
    });
    queue().wait();
}

// ---------------------------------------------------------------------------
// Matrix determinant via Gaussian elimination with partial pivoting.
// ---------------------------------------------------------------------------
template<typename T, Index N>
[[nodiscard]] inline T matrix_determinant(std::span<const T> A_in) {
    const T* pA = A_in.data();
    T* d = sycl::malloc_shared<T>(1, queue());
    queue().submit([=](sycl::handler& h) {
        h.single_task([=]() {
            T m[N * N];
            for (Index i = 0; i < N * N; ++i) m[i] = pA[i];
            T result = T{1};
            for (Index k = 0; k < N; ++k) {
                std::size_t piv = k;
                T mx = std::abs(m[k * N + k]);
                for (Index r = k + 1; r < N; ++r) {
                    const T v = std::abs(m[r * N + k]);
                    if (v > mx) { mx = v; piv = r; }
                }
                if (mx == T{0}) { d[0] = T{0}; return; }
                if (piv != k) {
                    for (Index j = 0; j < N; ++j) {
                        const T t = m[k * N + j];
                        m[k * N + j] = m[piv * N + j];
                        m[piv * N + j] = t;
                    }
                    result = -result;
                }
                const T pivot = m[k * N + k];
                for (Index r = k + 1; r < N; ++r) {
                    const T f = m[r * N + k] / pivot;
                    for (Index j = k; j < N; ++j) m[r * N + j] -= f * m[k * N + j];
                }
                result *= pivot;
            }
            d[0] = result;
        });
    });
    queue().wait();
    const T det = d[0];
    sycl::free(d, queue());
    return det;
}

// ---------------------------------------------------------------------------
// Vector kernels (elementwise, length n).
// ---------------------------------------------------------------------------
template<typename T>
inline void vector_add(std::span<const T> v1, std::span<const T> v2,
                       std::span<T> out) {
    const T* p1 = v1.data();
    const T* p2 = v2.data();
    T* po = out.data();
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(out.size()),
                       [=](sycl::id<1> i) { po[i] = p1[i] + p2[i]; });
    });
    queue().wait();
}

template<typename T>
inline void vector_scale(T scalar, std::span<const T> v, std::span<T> out) {
    const T* pv = v.data();
    T* po = out.data();
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(out.size()),
                       [=](sycl::id<1> i) { po[i] = scalar * pv[i]; });
    });
    queue().wait();
}

/// out = a - b (elementwise). In-place aliasing (out == a or out == b) is safe.
template<typename T>
inline void vector_sub(std::span<const T> a, std::span<const T> b,
                       std::span<T> out) {
    const T* pa = a.data();
    const T* pb = b.data();
    T* po = out.data();
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(out.size()),
                       [=](sycl::id<1> i) { po[i] = pa[i] - pb[i]; });
    });
    queue().wait();
}

/// out = a / b (elementwise, componentwise division — NOT reciprocal
/// multiplication; keeps bit-for-bit parity with the CPU reference).
/// In-place aliasing (out == a) is safe.
template<typename T>
inline void vector_divide(std::span<const T> a, std::span<const T> b,
                          std::span<T> out) {
    const T* pa = a.data();
    const T* pb = b.data();
    T* po = out.data();
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(out.size()),
                       [=](sycl::id<1> i) { po[i] = pa[i] / pb[i]; });
    });
    queue().wait();
}

/// Normalize v to unit length. Writes zeros for a zero vector.
/// Pass 1 is a sycl::reduction over the squared components; pass 2 scales.
/// The in-order queue guarantees pass 2 observes the completed reduction.
template<typename T>
inline void vector_normalize(std::span<const T> v, std::span<T> out) {
    const T* pv = v.data();
    T* po = out.data();
    T* norm2 = sycl::malloc_shared<T>(1, queue());
    norm2[0] = T{0};
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(v.size()),
                       sycl::reduction(norm2, sycl::plus<T>()),
                       [=](sycl::id<1> i, auto& acc) { acc += pv[i] * pv[i]; });
    });
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(out.size()), [=](sycl::id<1> i) {
            const T norm = sycl::sqrt(norm2[0]);
            po[i] = (norm > T{0}) ? (pv[i] / norm) : T{0};
        });
    });
    queue().wait();
    sycl::free(norm2, queue());
}

/// out[i] = v[i] / s (true scalar division — NOT reciprocal multiplication;
/// keeps bit-for-bit parity with the CPU reference). In-place aliasing safe.
template<typename T>
inline void vector_divide_scalar(T s, std::span<const T> v, std::span<T> out) {
    const T* pv = v.data();
    T* po = out.data();
    queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(out.size()),
                       [=](sycl::id<1> i) { po[i] = pv[i] / s; });
    });
    queue().wait();
}

} // namespace gpu

// ---- top-level runtime helpers (unchanged public surface) ----
inline void gpu_impl_init() { (void)gpu::queue(); }

inline void gpu_impl_cleanup() {}

struct DeviceInfo {
    std::string name;
    std::string vendor;
    size_t global_memory = 0;
    size_t max_compute_units = 0;
};

inline DeviceInfo get_device_info() {
    const sycl::device& dev = gpu::queue().get_device();
    DeviceInfo info;
    info.name = dev.get_info<sycl::info::device::name>();
    info.vendor = dev.get_info<sycl::info::device::vendor>();
    info.global_memory = dev.get_info<sycl::info::device::global_mem_size>();
    info.max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    return info;
}

inline void print_device_info() {
    const DeviceInfo info = get_device_info();
    std::fprintf(stderr, "[avbd gpu] %s (%s), %.1f MB, %zu CUs\n",
                 info.name.c_str(), info.vendor.c_str(),
                 static_cast<double>(info.global_memory) / (1024.0 * 1024.0),
                 info.max_compute_units);
}

// ---------------------------------------------------------------------------
// BackendTraits<Backend::GPU>: storage = shared-USM; ops = SYCL kernels.
// ---------------------------------------------------------------------------
template<>
struct BackendTraits<Backend::GPU> {
    template<typename T, Index Cap>
    class Storage {
        static_assert(Cap > 0, "fixed storage requires Cap > 0");
    public:
        Storage() : d_(sycl::malloc_shared<T>(static_cast<std::size_t>(Cap), gpu::queue())) {
            for (Index i = 0; i < Cap; ++i) d_[i] = T{0};
        }
        ~Storage() { if (d_) sycl::free(d_, gpu::queue()); }

        Storage(const Storage& o) : Storage() {
            for (Index i = 0; i < Cap; ++i) d_[i] = o.d_[i];
        }
        Storage& operator=(const Storage& o) {
            if (this != &o) for (Index i = 0; i < Cap; ++i) d_[i] = o.d_[i];
            return *this;
        }
        Storage(Storage&& o) noexcept : d_(o.d_) { o.d_ = nullptr; }
        Storage& operator=(Storage&& o) noexcept {
            if (this != &o) {
                if (d_) sycl::free(d_, gpu::queue());
                d_ = o.d_;
                o.d_ = nullptr;
            }
            return *this;
        }

        constexpr std::size_t size() const { return static_cast<std::size_t>(Cap); }
        T* data() { return d_; }
        const T* data() const { return d_; }

    private:
        T* d_;
    };

    template<typename T>
    class Storage<T, 0> {
    public:
        Storage() = default;
        ~Storage() { if (d_) sycl::free(d_, gpu::queue()); }

        Storage(const Storage& o) : n_(o.n_) { copyAlloc_(o); }
        Storage& operator=(const Storage& o) {
            if (this != &o) {
                if (d_) sycl::free(d_, gpu::queue());
                n_ = o.n_;
                copyAlloc_(o);
            }
            return *this;
        }
        Storage(Storage&& o) noexcept : d_(o.d_), n_(o.n_) { o.d_ = nullptr; o.n_ = 0; }
        Storage& operator=(Storage&& o) noexcept {
            if (this != &o) {
                if (d_) sycl::free(d_, gpu::queue());
                d_ = o.d_;
                n_ = o.n_;
                o.d_ = nullptr;
                o.n_ = 0;
            }
            return *this;
        }

        std::size_t size() const { return n_; }
        T* data() { return d_; }
        const T* data() const { return d_; }
        void resize(std::size_t n) {
            if (n == n_) return;
            T* nd = (n > 0) ? sycl::malloc_shared<T>(n, gpu::queue()) : nullptr;
            if (nd) {
                const std::size_t m = std::min<std::size_t>(n, n_);
                for (Index i = 0; i < m; ++i) nd[i] = d_[i];
                for (Index i = m; i < n; ++i) nd[i] = T{0};
            }
            if (d_) sycl::free(d_, gpu::queue());
            d_ = nd;
            n_ = n;
        }

    private:
        void copyAlloc_(const Storage& o) {
            d_ = (n_ > 0) ? sycl::malloc_shared<T>(n_, gpu::queue()) : nullptr;
            for (Index i = 0; i < n_; ++i) d_[i] = o.d_[i];
        }
        T* d_ = nullptr;
        std::size_t n_ = 0;
    };

    // ---- 算子 (SYCL 内核) ----
    template<typename T>
    static void add(const T* a, const T* b, T* out, Index n) {
        gpu::vector_add<T>({a, static_cast<std::size_t>(n)},
                           {b, static_cast<std::size_t>(n)},
                           {out, static_cast<std::size_t>(n)});
    }
    template<typename T>
    static void sub(const T* a, const T* b, T* out, Index n) {
        gpu::vector_sub<T>({a, static_cast<std::size_t>(n)},
                           {b, static_cast<std::size_t>(n)},
                           {out, static_cast<std::size_t>(n)});
    }
    template<typename T>
    static void scale(T s, const T* v, T* out, Index n) {
        gpu::vector_scale<T>(s, {v, static_cast<std::size_t>(n)},
                             {out, static_cast<std::size_t>(n)});
    }
    template<typename T>
    static void div(T s, const T* v, T* out, Index n) {
        gpu::vector_divide_scalar<T>(s, {v, static_cast<std::size_t>(n)},
                                     {out, static_cast<std::size_t>(n)});
    }
    template<typename T>
    static void negate(const T* v, T* out, Index n) {
        gpu::vector_scale<T>(T{-1}, {v, static_cast<std::size_t>(n)},
                             {out, static_cast<std::size_t>(n)});
    }
    template<typename T, Index N>
    static void mat_mul(const T* a, const T* b, T* c) {
        gpu::matrix_multiply<T, N>({a, std::size_t(N) * N},
                                   {b, std::size_t(N) * N},
                                   {c, std::size_t(N) * N});
    }
};

} // namespace hlcl

#else // HLCL_GPU_ENABLED

// Stub declarations when GPU support is not enabled (keeps this header
// includable from CPU-only translation units).
#include <cstdio>
#include <cstddef>
#include <string>
namespace hlcl {

inline void gpu_impl_init() {}
inline void gpu_impl_cleanup() {}
struct DeviceInfo {
    std::string name;
    std::string vendor;
    size_t global_memory = 0;
    size_t max_compute_units = 0;
};
inline DeviceInfo get_device_info() { return DeviceInfo{}; }
inline void print_device_info() { std::fprintf(stderr, "[avbd gpu] disabled\n"); }

} // namespace hlcl

#endif // HLCL_GPU_ENABLED
