// GPU matrix multiplication tests for AVBD.
//
// Two modes:
//   HLCL_GPU_ENABLED defined   -> Backend::GPU matrices run the SYCL kernels and
//                                 are cross-checked against the CPU (Eigen-like)
//                                 reference on the same matrix content.
//   HLCL_GPU_ENABLED undefined -> CPU-only self-consistency check (member
//                                 product operator* vs naive triple-loop).

#include "hlcl/core.hpp"
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>

#ifdef HLCL_GPU_ENABLED
#include "hlcl/gpu_impl.hpp"
#endif

using namespace hlcl;

namespace {

template<typename T> T tolerance();
template<> float  tolerance<float>()  { return 1e-5f; }
template<> double tolerance<double>() { return 1e-9; }

#ifdef HLCL_GPU_ENABLED
// GPU matrices multiply via SYCL kernel; compare against the CPU product.
template<typename T, int N>
bool checkAccuracy(const Matrix<T, N, N, Backend::GPU>& gA,
                   const Matrix<T, N, N, Backend::GPU>& gB,
                   const Matrix<T, N, N>& cpuA, const Matrix<T, N, N>& cpuB) {
    Matrix<T, N, N> want = cpuA * cpuB;                 // CPU reference
    Matrix<T, N, N, Backend::GPU> gr = gA * gB;         // SYCL kernel path
    Matrix<T, N, N> got;
    gr.copyToHost(got);

    T mx = T{0};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mx = std::max(mx, std::abs(got(i, j) - want(i, j)));
    bool ok = mx < tolerance<T>();
    std::cout << "    " << N << "x" << N << " max err " << std::scientific << mx
              << " (tol " << tolerance<T>() << ") -> " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

template<typename T, int N>
bool checkPerformance(const Matrix<T, N, N, Backend::GPU>& gA,
                      const Matrix<T, N, N, Backend::GPU>& gB) {
    constexpr int kIters = 200;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kIters; ++it) { Matrix<T, N, N, Backend::GPU> r = gA * gB; (void)r; }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    " << N << "x" << N << " GPU mul: "
              << std::fixed << std::setprecision(3) << (ms / kIters) << " ms/op\n";
    return true;
}

template<typename T, int N>
void runSuite(const char* type, bool& ok) {
    std::cout << "Suite " << N << "x" << N << " " << type << "\n";
    Matrix<T, N, N> cpuA, cpuB;
    cpuA.setConstant(T{2}); cpuB.setConstant(T{3});
    Matrix<T, N, N, Backend::GPU> gA, gB;
    gA.copyFromHost(cpuA);
    gB.copyFromHost(cpuB);
    ok &= checkAccuracy<T, N>(gA, gB, cpuA, cpuB);
    ok &= checkPerformance<T, N>(gA, gB);
}
#else
template<typename T, int N>
bool checkAccuracy(const Matrix<T, N, N>& A, const Matrix<T, N, N>& B) {
    Matrix<T, N, N> got = A * B;      // member product
    Matrix<T, N, N> want;             // naive triple-loop reference
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            T acc = T{0};
            for (int k = 0; k < N; ++k) acc += A(i, k) * B(k, j);
            want(i, j) = acc;
        }
    T mx = T{0};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) mx = std::max(mx, std::abs(got(i, j) - want(i, j)));
    bool ok = mx < tolerance<T>();
    std::cout << "    " << N << "x" << N << " max err " << std::scientific << mx
              << " (tol " << tolerance<T>() << ") -> " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}
template<typename T, int N>
bool checkPerformance(const Matrix<T, N, N>& A, const Matrix<T, N, N>& B) {
    constexpr int kIters = 200;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kIters; ++it) { Matrix<T, N, N> r = A * B; (void)r; }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    " << N << "x" << N << " CPU mul: "
              << std::fixed << std::setprecision(3) << (ms / kIters) << " ms/op\n";
    return true;
}
template<typename T, int N>
void runSuite(const char* type, bool& ok) {
    std::cout << "Suite " << N << "x" << N << " " << type << "\n";
    Matrix<T, N, N> A, B;
    A.setConstant(T{2}); B.setConstant(T{3});
    ok &= checkAccuracy<T, N>(A, B);
    ok &= checkPerformance<T, N>(A, B);
}
#endif

} // namespace

int main() {
    std::cout << "========================================\n";
#ifdef HLCL_GPU_ENABLED
    std::cout << "GPU matrix multiply (SYCL kernels)\n";
#else
    std::cout << "GPU matrix multiply (CPU-only build)\n";
#endif
    std::cout << "========================================\n";

    bool ok = true;
#ifdef HLCL_GPU_ENABLED
    gpu_impl_init();
    std::cout << "device: " << get_device_info().name << "\n\n";
#endif

    runSuite<float, 4>("float", ok);
    runSuite<float, 8>("float", ok);
    runSuite<float, 16>("float", ok);
    runSuite<double, 4>("double", ok);
    runSuite<double, 8>("double", ok);

    std::cout << "\n========================================\n";
    if (ok) { std::cout << "All matrix tests PASSED\n"; return 0; }
    std::cout << "Some matrix tests FAILED\n"; return 1;
}
