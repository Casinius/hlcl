#pragma once

// GPU (SYCL) kernel layer for AVBD.
//
// Design notes
// ------------
// * Header-only by intent (mirrors avbd_core). All kernels are inline so a
//   GPU-mode translation unit gets them without linking an orphaned .cpp.
// * Backend selection is OpenCL-portable: it prefers an accelerator (GPU),
//   honours the HLCL_SYCL_DEVICE env var (gpu|cpu), and otherwise falls back
//   to whatever SYCL device the runtime exposes (AdaptiveCpp OpenMP host,
//   an OpenCL device, CUDA, ...). The SYCL source itself is backend agnostic.
// * Every kernel is deliberately tiny and self-contained. Mathematically they
//   replicate the CPU (Eigen-like) reference in core; correctness is asserted
//   by the GPU tests comparing kernel output against the CPU path.
// * Kernels take flat row-major pointers that MUST be shared-USM allocations
//   (sycl::malloc_shared; on the AdaptiveCpp OpenMP host this is equivalent
//   to plain malloc). Kernels read/write them in place: no copies, no
//   transient SYCL buffers, no accessors.

#ifdef HLCL_GPU_ENABLED

#include <CL/sycl.hpp>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace hlcl {

namespace gpu {

/// Pick the SYCL device: honour HLCL_SYCL_DEVICE (gpu|cpu), else prefer a GPU
/// accelerator, else the runtime default (CPU/OpenMP host in CPU-only builds).
inline cl::sycl::device select_device() {
    cl::sycl::device dev;
    if (const char* e = std::getenv("HLCL_SYCL_DEVICE")) {
        std::string sel = e;
        try {
            if (sel == "cpu") dev = cl::sycl::device(cl::sycl::cpu_selector());
            else if (sel == "gpu") dev = cl::sycl::device(cl::sycl::gpu_selector());
            else dev = cl::sycl::device(cl::sycl::default_selector());
            return dev;
        } catch (const cl::sycl::exception&) {
            // requested device class unavailable -> fall through to default
        }
    }
    try {
        const auto devices = cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);
        if (!devices.empty()) return devices[0];
    } catch (...) { /* fall through */ }
    return cl::sycl::device(cl::sycl::default_selector());
}

/// Process-wide shared queue (lazily created).
///
/// Intentionally leaked (heap-allocated, never destroyed): AdaptiveCpp 25.10
/// can segfault at process exit when a process-wide SYCL queue is destroyed
/// (its runtime teardown races the OpenMP worker threads of prior kernels).
/// Results are fully correct; only the exit-time destructor path is unsafe.
inline cl::sycl::queue& queue() {
    static cl::sycl::queue* q = new cl::sycl::queue(
        select_device(), cl::sycl::property::queue::in_order());
    return *q;
}

// ---------------------------------------------------------------------------
// Matrix kernels: C = A * B   (all row-major, N x N)
// ---------------------------------------------------------------------------
template<typename T, int N>
inline void matrix_multiply(const T* A, const T* B, T* C) {
    queue().submit([&](cl::sycl::handler& h) {
        h.parallel_for(cl::sycl::range<2>(N, N), [=](cl::sycl::item<2> it) {
            const int i = static_cast<int>(it[0]);
            const int j = static_cast<int>(it[1]);
            T s = T{0};
            for (int k = 0; k < N; ++k) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        });
    });
    queue().wait();
}

// ---------------------------------------------------------------------------
// Matrix inverse via Gauss-Jordan elimination (single work item).
// Singular matrices produce a zero matrix (matches CPU inverse() contract).
// ---------------------------------------------------------------------------
template<typename T, int N>
inline void matrix_inverse(const T* A_in, T* C_out) {
    queue().submit([&](cl::sycl::handler& h) {
        h.single_task([=]() {
            T m[N * N];
            for (int i = 0; i < N * N; ++i) m[i] = A_in[i];
            T inv[N * N];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) inv[i * N + j] = (i == j) ? T{1} : T{0};
            for (int col = 0; col < N; ++col) {
                int piv = col;
                T mx = std::abs(m[col * N + col]);
                for (int r = col + 1; r < N; ++r) {
                    const T v = std::abs(m[r * N + col]);
                    if (v > mx) { mx = v; piv = r; }
                }
                if (mx == T{0}) {
                    for (int i = 0; i < N * N; ++i) C_out[i] = T{0};
                    return; // singular
                }
                if (piv != col) {
                    for (int j = 0; j < N; ++j) {
                        T t = m[col * N + j]; m[col * N + j] = m[piv * N + j]; m[piv * N + j] = t;
                        t = inv[col * N + j]; inv[col * N + j] = inv[piv * N + j]; inv[piv * N + j] = t;
                    }
                }
                const T d = m[col * N + col];
                for (int j = 0; j < N; ++j) { m[col * N + j] /= d; inv[col * N + j] /= d; }
                for (int r = 0; r < N; ++r) {
                    if (r == col) continue;
                    const T f = m[r * N + col];
                    for (int j = 0; j < N; ++j) {
                        m[r * N + j] -= f * m[col * N + j];
                        inv[r * N + j] -= f * inv[col * N + j];
                    }
                }
            }
            for (int i = 0; i < N * N; ++i) C_out[i] = inv[i];
        });
    });
    queue().wait();
}

// ---------------------------------------------------------------------------
// Matrix determinant via Gaussian elimination with partial pivoting.
// ---------------------------------------------------------------------------
template<typename T, int N>
inline T matrix_determinant(const T* A_in) {
    T* d = cl::sycl::malloc_shared<T>(1, queue());
    queue().submit([=](cl::sycl::handler& h) {
        h.single_task([=]() {
            T m[N * N];
            for (int i = 0; i < N * N; ++i) m[i] = A_in[i];
            T result = T{1};
            for (int k = 0; k < N; ++k) {
                int piv = k;
                T mx = std::abs(m[k * N + k]);
                for (int r = k + 1; r < N; ++r) {
                    const T v = std::abs(m[r * N + k]);
                    if (v > mx) { mx = v; piv = r; }
                }
                if (mx == T{0}) { d[0] = T{0}; return; }
                if (piv != k) {
                    for (int j = 0; j < N; ++j) {
                        const T t = m[k * N + j];
                        m[k * N + j] = m[piv * N + j];
                        m[piv * N + j] = t;
                    }
                    result = -result;
                }
                const T pivot = m[k * N + k];
                for (int r = k + 1; r < N; ++r) {
                    const T f = m[r * N + k] / pivot;
                    for (int j = k; j < N; ++j) m[r * N + j] -= f * m[k * N + j];
                }
                result *= pivot;
            }
            d[0] = result;
        });
    });
    queue().wait();
    const T det = d[0];
    cl::sycl::free(d, queue());
    return det;
}

// ---------------------------------------------------------------------------
// Vector kernels (elementwise, length n).
// ---------------------------------------------------------------------------
template<typename T>
inline void vector_add(const T* v1, const T* v2, T* out, int n) {
    queue().submit([&](cl::sycl::handler& h) {
        h.parallel_for(cl::sycl::range<1>(n), [=](cl::sycl::id<1> i) { out[i] = v1[i] + v2[i]; });
    });
    queue().wait();
}

template<typename T>
inline void vector_scale(T scalar, const T* v, T* out, int n) {
    queue().submit([&](cl::sycl::handler& h) {
        h.parallel_for(cl::sycl::range<1>(n), [=](cl::sycl::id<1> i) { out[i] = scalar * v[i]; });
    });
    queue().wait();
}

/// out = a - b (elementwise). In-place aliasing (out == a or out == b) is safe.
template<typename T>
inline void vector_sub(const T* a, const T* b, T* out, int n) {
    queue().submit([&](cl::sycl::handler& h) {
        h.parallel_for(cl::sycl::range<1>(n), [=](cl::sycl::id<1> i) { out[i] = a[i] - b[i]; });
    });
    queue().wait();
}

/// Normalize v to unit length. Writes zeros for a zero vector.
template<typename T>
inline void vector_normalize(const T* v, T* out, int n) {
    T* norm = cl::sycl::malloc_shared<T>(1, queue());
    norm[0] = T{0};
    // reduction (single work item over the whole vector); the second submit
    // is ordered after this one by the in-order queue.
    queue().submit([=](cl::sycl::handler& h) {
        h.single_task([=]() {
            T s = T{0};
            for (int i = 0; i < n; ++i) s += v[i] * v[i];
            norm[0] = static_cast<T>(std::sqrt(s));
        });
    });
    queue().submit([&](cl::sycl::handler& h) {
        h.parallel_for(cl::sycl::range<1>(n), [=](cl::sycl::id<1> i) {
            out[i] = (norm[0] > T{0}) ? (v[i] / norm[0]) : T{0};
        });
    });
    queue().wait();
    cl::sycl::free(norm, queue());
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
    const cl::sycl::device& dev = gpu::queue().get_device();
    DeviceInfo info;
    info.name = dev.get_info<cl::sycl::info::device::name>();
    info.vendor = dev.get_info<cl::sycl::info::device::vendor>();
    info.global_memory = dev.get_info<cl::sycl::info::device::global_mem_size>();
    info.max_compute_units = dev.get_info<cl::sycl::info::device::max_compute_units>();
    return info;
}

inline void print_device_info() {
    const DeviceInfo info = get_device_info();
    std::fprintf(stderr, "[avbd gpu] %s (%s), %.1f MB, %zu CUs\n",
                 info.name.c_str(), info.vendor.c_str(),
                 static_cast<double>(info.global_memory) / (1024.0 * 1024.0),
                 info.max_compute_units);
}

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
