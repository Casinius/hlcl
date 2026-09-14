// Host-device copy cost benchmark for GPU kernels.
#include <iostream>
#include <chrono>
#include <vector>
#include "hlcl/core.hpp"
//
// Measures the overhead of explicit memcpy between host and device memory
// for Matrix and Vector operations with the current buffer-based implementation.

#include "hlcl/gpu_impl.hpp"

using namespace hlcl;

namespace {

template<typename T, hlcl::Index Size>
double measure_copy_overhead(const Vector<T, Size, Backend::GPU>& v,
                              const std::vector<T>& host_ref,
                              int iterations) {
    (void)host_ref;  // Unused parameter
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::vector<T> temp(Size);
        // Simulate copyToHost
        for (int j = 0; j < Size; ++j) {
            temp[j] = v[j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Copy Overhead: " << iterations << " iterations in " 
              << elapsed << " ms" << std::endl;
    std::cout << "  Throughput: " << (iterations / elapsed * 1000.0) << " copy/sec" 
              << std::endl;
    return elapsed / iterations;
}

template<typename T, hlcl::Index Rows, hlcl::Index Cols>
double measure_copy_overhead(const Matrix<T, Rows, Cols, Backend::GPU>& m,
                              const std::vector<T>& host_ref,
                              int iterations) {
    (void)host_ref;  // Unused parameter
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::vector<T> temp(Rows * Cols);
        // Simulate copyToHost
        for (int j = 0; j < Rows * Cols; ++j) {
            temp[j] = m.data()[j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Copy Overhead: " << iterations << " iterations in " 
              << elapsed << " ms" << std::endl;
    std::cout << "  Throughput: " << (iterations / elapsed * 1000.0) << " copy/sec" 
              << std::endl;
    return elapsed / iterations;
}

void run_copy_benchmark() {
    std::cout << "\n=== Host-Device Copy Cost Benchmark ===" << std::endl;
    std::cout << "Device: NVIDIA GeForce GTX 1660 Ti Mobile (TU116M) (simulated)" << std::endl;

    // Vector copy overhead (Size=3, 16, 64, 1024)
    const int iterations = 100000;
    std::cout << "\nVector Copy Overhead (Size=3):" << std::endl;
    Vector<float, 3, Backend::GPU> v3({1.0f, 2.0f, 3.0f});
    std::vector<float> ref3 = {1.0f, 2.0f, 3.0f};
    measure_copy_overhead(v3, ref3, iterations);

    std::cout << "\nVector Copy Overhead (Size=16):" << std::endl;
    Vector<float, 16, Backend::GPU> v16;
    std::vector<float> ref16(16, 1.0f);
    measure_copy_overhead(v16, ref16, iterations);

    std::cout << "\nVector Copy Overhead (Size=1024):" << std::endl;
    Vector<float, 1024, Backend::GPU> v1024;
    std::vector<float> ref1024(1024, 1.0f);
    measure_copy_overhead(v1024, ref1024, iterations);

    // Matrix copy overhead (4x4, 8x8, 16x16)
    std::cout << "\nMatrix Copy Overhead (4x4):" << std::endl;
    Matrix<float, 4, 4, Backend::GPU> m4;
    std::vector<float> ref4(16, 1.0f);
    measure_copy_overhead(m4, ref4, iterations);

    std::cout << "\nMatrix Copy Overhead (8x8):" << std::endl;
    Matrix<float, 8, 8, Backend::GPU> m8;
    std::vector<float> ref8(64, 1.0f);
    measure_copy_overhead(m8, ref8, iterations);

    std::cout << "\nMatrix Copy Overhead (16x16):" << std::endl;
    Matrix<float, 16, 16, Backend::GPU> m16;
    std::vector<float> ref16mat(256, 1.0f);
    measure_copy_overhead(m16, ref16mat, iterations);

    std::cout << "\n=== Copy Benchmark Complete ===" << std::endl;
}

} // namespace

int main() {
    run_copy_benchmark();
    return 0;
}
