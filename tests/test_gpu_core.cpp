// GPU backend correctness test (SYCL kernels vs CPU Eigen-like reference).
//
// Compile ONLY with a SYCL compiler defining HLCL_GPU_ENABLED, e.g.:
//   acpp --acpp-targets=omp -O2 -std=c++17 -DHLCL_GPU_ENABLED
//        -Icore/include tests/test_gpu_core.cpp -o test_gpu_core
// Run: ./test_gpu_core
//
// Verifies, per type/size, that the Backend::GPU SYCL kernels match the CPU
// reference exactly (matrix multiply / inverse / determinant, vector ops).

#include "hlcl/core.hpp"
#include <cmath>
#include <cstdio>
#include <random>

#ifdef HLCL_GPU_ENABLED
#include "hlcl/gpu_impl.hpp"
#endif

using namespace hlcl;

static bool g_fail = false;
static int g_checks = 0;

template<typename T> static T tol();
template<> float  tol<float>()  { return 1e-5f; }
template<> double tol<double>() { return 1e-9;  }

#define CHECK(label, cond, extra)                                      \
    do { ++g_checks; if (cond) std::printf("  %-22s OK\n", label);     \
         else { std::printf("  %-22s FAIL %s\n", label, extra); g_fail = true; } } while (0)

#ifdef HLCL_GPU_ENABLED

template<typename T, int N>
void test_mat_mul() {
    std::mt19937 rng(1 + N); std::uniform_real_distribution<T> ud(-1, 1);
    Matrix<T, N, N> A, B;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) { A(i, j) = ud(rng); B(i, j) = ud(rng); }
    Matrix<T, N, N> cpu = A * B;
    Matrix<T, N, N, Backend::GPU> gA, gB;
    gA.copyFromHost(A); gB.copyFromHost(B);
    Matrix<T, N, N, Backend::GPU> gR = gA * gB;
    Matrix<T, N, N> out; gR.copyToHost(out);
    T mx = 0; for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) mx = std::max(mx, std::abs(out(i, j) - cpu(i, j)));
    char buf[64]; std::snprintf(buf, sizeof buf, "maxerr=%.3g", (double)mx);
    CHECK(("mul<T,N>"), mx < tol<T>(), buf);
}

template<typename T, int N>
void test_mat_inv_det() {
    std::mt19937 rng(7 + N); std::uniform_real_distribution<T> ud(-1, 1);
    Matrix<T, N, N> A;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) A(i, j) = ud(rng) + (i == j ? T{2} : T{0});
    Matrix<T, N, N> cpuInv = inverse(A);
    Matrix<T, N, N, Backend::GPU> gA; gA.copyFromHost(A);

    Matrix<T, N, N, Backend::GPU> gInv = inverseGPU<T, N>(gA);
    Matrix<T, N, N> outInv; gInv.copyToHost(outInv);
    T mx = 0; for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) mx = std::max(mx, std::abs(outInv(i, j) - cpuInv(i, j)));
    char b1[64]; std::snprintf(b1, sizeof b1, "maxerr=%.3g", (double)mx);
    CHECK(("invGPU<T,N>"), mx < tol<T>(), b1);

    Matrix<T, N, N, Backend::GPU> gInv2 = inverse(gA);   // generic host path over GPU backend
    Matrix<T, N, N> outInv2; gInv2.copyToHost(outInv2);
    T mx2 = 0; for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) mx2 = std::max(mx2, std::abs(outInv2(i, j) - cpuInv(i, j)));
    char b2[64]; std::snprintf(b2, sizeof b2, "maxerr=%.3g", (double)mx2);
    CHECK(("inv(generic)<T,N>"), mx2 < tol<T>(), b2);

    T cpuDet = determinant(A);
    T gpuDet = determinantGPU<T, N>(gA);
    char b3[64]; std::snprintf(b3, sizeof b3, "gpu=%.6g cpu=%.6g", (double)gpuDet, (double)cpuDet);
    CHECK(("detGPU<T,N>"), std::abs(cpuDet - gpuDet) < tol<T>(), b3);
}

template<typename T, int N>
void test_vec() {
    std::mt19937 rng(11 + N); std::uniform_real_distribution<T> ud(-2, 2);
    Vector<T, N> a, b;
    for (int i = 0; i < N; ++i) { a[i] = ud(rng); b[i] = ud(rng); }
    Vector<T, N> csum = a + b, csub = a - b, cscale = a * T{3};
    Vector<T, N, Backend::GPU> ga(a), gb(b);
    Vector<T, N, Backend::GPU> gsum = ga + gb, gsub = ga - gb, gscale = ga * T{3};
    Vector<T, N> sum2(gsum), sub2(gsub), scale2(gscale);
    T mx = 0;
    for (int i = 0; i < N; ++i) {
        mx = std::max(mx, std::abs(sum2[i] - csum[i]));
        mx = std::max(mx, std::abs(sub2[i] - csub[i]));
        mx = std::max(mx, std::abs(scale2[i] - cscale[i]));
    }
    char b1[64]; std::snprintf(b1, sizeof b1, "maxerr=%.3g", (double)mx);
    CHECK(("vec+,-,* <T,N>"), mx < tol<T>(), b1);

    Vector<T, N, Backend::GPU> gb2(gb);
    gb2 += ga; gb2 -= ga; gb2 *= T{2}; gb2 /= T{2};
    Vector<T, N> bfinal(gb2);
    bool ok = true; for (int i = 0; i < N; ++i) if (std::abs(bfinal[i] - b[i]) > tol<T>()) ok = false;
    CHECK(("vec inplace<T,N>"), ok, "roundtrip mismatch");

    CHECK(("vec norm<T,N>"), std::abs(ga.norm() - a.norm()) < tol<T>(), "norm mismatch");
    CHECK(("vec dot<T,N>"), std::abs(ga.dot(gb) - a.dot(b)) < tol<T>(), "dot mismatch");
}

#else
// Compiling without HLCL_GPU_ENABLED: report a build error context and skip.
#endif

int main() {
    std::printf("GPU core correctness (SYCL kernels vs CPU reference)\n");
#ifdef HLCL_GPU_ENABLED
    gpu_impl_init();
    std::printf("device: %s\n", get_device_info().name.c_str());
    test_mat_mul<float, 4>();   test_mat_mul<float, 8>();
    test_mat_mul<double, 4>();  test_mat_mul<double, 8>();
    test_mat_inv_det<float, 4>(); test_mat_inv_det<float, 8>();
    test_mat_inv_det<double, 4>();
    test_vec<float, 4>();       test_vec<float, 8>();
    test_vec<double, 4>();      test_vec<double, 8>();
    std::printf("%d checks: %s\n", g_checks, g_fail ? "SOME FAIL" : "ALL OK");
    return g_fail ? 1 : 0;
#else
    std::printf("ERROR: compiled without HLCL_GPU_ENABLED; use a SYCL compiler (acpp).\n");
    return 2;
#endif
}
