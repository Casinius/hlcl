// Kompute (Vulkan compute) backend correctness test.
//
// Build via xmake with --kompute=y (target test_kompute_core); the binary
// links the kompute package and defines HLCL_KOMPUTE_ENABLED. Run: the
// kernels execute on the Vulkan device (HLCL_KOMPUTE_DEVICE selects the
// physical device index), compared elementwise against the CPU reference.
//
// Verifies, per type/size: matrix multiply / inverse / determinant (incl.
// generic host path over the Kompute backend) and vector ops. Double runs
// the host fallback (Kompute 0.8.0 has no shaderFloat64) — still checked
// against CPU to pin the fallback contract.

#include "hlcl/core.hpp"
#include <cmath>
#include <cstdio>
#include <random>

#ifdef HLCL_KOMPUTE_ENABLED
#include "hlcl/kompute_impl.hpp"
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

#ifdef HLCL_KOMPUTE_ENABLED

template<typename T, int N>
void test_mat_mul() {
    std::mt19937 rng(1 + N); std::uniform_real_distribution<T> ud(-1, 1);
    Matrix<T, N, N> A, B;
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) { A(i, j) = ud(rng); B(i, j) = ud(rng); }
    Matrix<T, N, N> cpu = A * B;
    Matrix<T, N, N, Backend::Kompute> kA, kB;
    kA.copyFromHost(A); kB.copyFromHost(B);
    Matrix<T, N, N, Backend::Kompute> kR = kA * kB;
    Matrix<T, N, N> out; kR.copyToHost(out);
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
    Matrix<T, N, N, Backend::Kompute> kA; kA.copyFromHost(A);

    Matrix<T, N, N, Backend::Kompute> kInv = inverseKompute<T, N>(kA);
    Matrix<T, N, N> outInv; kInv.copyToHost(outInv);
    T mx = 0; for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) mx = std::max(mx, std::abs(outInv(i, j) - cpuInv(i, j)));
    char b1[64]; std::snprintf(b1, sizeof b1, "maxerr=%.3g", (double)mx);
    CHECK(("invK<T,N>"), mx < tol<T>(), b1);

    Matrix<T, N, N, Backend::Kompute> kInv2 = inverse(kA); // generic host path over Kompute backend
    Matrix<T, N, N> outInv2; kInv2.copyToHost(outInv2);
    T mx2 = 0; for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) mx2 = std::max(mx2, std::abs(outInv2(i, j) - cpuInv(i, j)));
    char b2[64]; std::snprintf(b2, sizeof b2, "maxerr=%.3g", (double)mx2);
    CHECK(("inv(generic)<T,N>"), mx2 < tol<T>(), b2);

    T cpuDet = determinant(A);
    T kDet = determinantKompute<T, N>(kA);
    char b3[64]; std::snprintf(b3, sizeof b3, "k=%.6g cpu=%.6g", (double)kDet, (double)cpuDet);
    T dtol = tol<T>() * std::max(T{1}, std::abs(cpuDet));
    CHECK(("detK<T,N>"), std::abs(cpuDet - kDet) < dtol, b3);
}

template<typename T, int N>
void test_vec() {
    std::mt19937 rng(11 + N); std::uniform_real_distribution<T> ud(-2, 2);
    Vector<T, N> a, b;
    for (int i = 0; i < N; ++i) { a[i] = ud(rng); b[i] = ud(rng); }
    Vector<T, N> csum = a + b, csub = a - b, cscale = a * T{3};
    Vector<T, N, Backend::Kompute> ka(a), kb(b);
    Vector<T, N, Backend::Kompute> ksum = ka + kb, ksub = ka - kb, kscale = ka * T{3};
    Vector<T, N> sum2(ksum), sub2(ksub), scale2(kscale);
    T mx = 0;
    for (int i = 0; i < N; ++i) {
        mx = std::max(mx, std::abs(sum2[i] - csum[i]));
        mx = std::max(mx, std::abs(sub2[i] - csub[i]));
        mx = std::max(mx, std::abs(scale2[i] - cscale[i]));
    }
    char b1[64]; std::snprintf(b1, sizeof b1, "maxerr=%.3g", (double)mx);
    CHECK(("vec+,-,* <T,N>"), mx < tol<T>(), b1);

    Vector<T, N, Backend::Kompute> kb2(kb);
    kb2 += ka; kb2 -= ka; kb2 *= T{2}; kb2 /= T{2};
    Vector<T, N> bfinal(kb2);
    bool ok = true; for (int i = 0; i < N; ++i) if (std::abs(bfinal[i] - b[i]) > tol<T>()) ok = false;
    CHECK(("vec inplace<T,N>"), ok, "roundtrip mismatch");

    CHECK(("vec norm<T,N>"), std::abs(ka.norm() - a.norm()) < tol<T>(), "norm mismatch");
    CHECK(("vec dot<T,N>"), std::abs(ka.dot(kb) - a.dot(b)) < tol<T>(), "dot mismatch");
}

#else
// Compiling without HLCL_KOMPUTE_ENABLED: report and skip.
#endif

int main() {
    std::printf("Kompute core correctness (Vulkan kernels vs CPU reference)\n");
#ifdef HLCL_KOMPUTE_ENABLED
    kompute_impl_init();
    std::printf("device: %s\n", get_kompute_device_info().name.c_str());
    test_mat_mul<float, 4>();   test_mat_mul<float, 8>();  test_mat_mul<float, 16>();
    test_mat_mul<double, 4>();  test_mat_mul<double, 8>();
    test_mat_inv_det<float, 4>(); test_mat_inv_det<float, 8>();
    test_mat_inv_det<float, 16>();
    test_mat_inv_det<double, 4>();
    test_vec<float, 4>();       test_vec<float, 8>();      test_vec<float, 16>();
    test_vec<float, 1024>();
    test_vec<double, 4>();      test_vec<double, 8>();
    std::printf("%d checks: %s\n", g_checks, g_fail ? "SOME FAIL" : "ALL OK");
    return g_fail ? 1 : 0;
#else
    std::printf("ERROR: compiled without HLCL_KOMPUTE_ENABLED; configure with --kompute=y.\n");
    return 2;
#endif
}
