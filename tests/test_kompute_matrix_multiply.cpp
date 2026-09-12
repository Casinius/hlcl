// Kompute backend matrix-multiply accuracy test (Vulkan kernel vs CPU
// reference). Mirrors test_gpu_matrix_multiply.cpp; with --kompute=y the
// member operator* dispatches to the Vulkan compute kernel on the device
// chosen by HLCL_KOMPUTE_DEVICE; without it the same suites run as a CPU
// self-consistency check.

#include "hlcl/core.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#ifdef HLCL_KOMPUTE_ENABLED
#include "hlcl/kompute_impl.hpp"
#include "hlcl/matrix_kompute.hpp"
#endif

namespace {

using namespace hlcl;

template<typename T, int N>
void runSuite(const char* label, bool& ok) {
    std::printf("Suite %dx%d %s\n", N, N, label);
    std::mt19937 rng(N * 31 + (std::is_same<T, float>::value ? 7 : 13));
    std::uniform_real_distribution<T> ud(-1, 1);
    const T tol = std::is_same<T, float>::value ? T{1e-5} : T{1e-9};

    Matrix<T, N, N> A, B;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) { A(i, j) = ud(rng); B(i, j) = ud(rng); }
    Matrix<T, N, N> ref = A * B;

#ifdef HLCL_KOMPUTE_ENABLED
    Matrix<T, N, N, Backend::Kompute> kA, kB;
    kA.copyFromHost(A);
    kB.copyFromHost(B);
    Matrix<T, N, N, Backend::Kompute> kR = kA * kB;
    Matrix<T, N, N> out;
    kR.copyToHost(out);
#else
    Matrix<T, N, N> out = A * B; // CPU-only build: self-consistency
#endif

    T mx = T{0};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mx = std::max(mx, std::abs(out(i, j) - ref(i, j)));
    bool pass = mx < tol;
    ok = ok && pass;
    std::printf("    %dx%d max err %.*g (tol %g) -> %s\n", N, N,
        std::is_same<T, float>::value ? 6 : 12, static_cast<double>(mx),
        static_cast<double>(tol), pass ? "PASS" : "FAIL");
}

} // namespace

int main() {
    std::printf("========================================\n");
#ifdef HLCL_KOMPUTE_ENABLED
    std::printf("Kompute matrix multiply (Vulkan kernels)\n");
#else
    std::printf("Kompute matrix multiply (CPU-only build)\n");
#endif
    std::printf("========================================\n");

    bool ok = true;
#ifdef HLCL_KOMPUTE_ENABLED
    kompute_impl_init();
    std::printf("device: %s\n\n", get_kompute_device_info().name.c_str());
#endif

    runSuite<float, 4>("float", ok);
    runSuite<float, 8>("float", ok);
    runSuite<float, 16>("float", ok);
    runSuite<double, 4>("double", ok);
    runSuite<double, 8>("double", ok);

    std::printf("\n========================================\n");
    if (ok) { std::printf("All matrix tests PASSED\n"); return 0; }
    std::printf("Some matrix tests FAILED\n"); return 1;
}
