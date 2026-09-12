#include "test_framework.hpp"
#include "hlcl/core.hpp"
#include <limits>
#include <random>

namespace hlcl {
namespace test {

class VectorTestSuite : public TestSuite {
public:
    VectorTestSuite() : TestSuite("Vector Test Suite") {}

    void run() override {
        test_constructors();
        test_arithmetic();
        test_stats();
        test_setters();
        test_edge_cases();
        test_precision();
        test_random_values();
    }

private:
    void test_constructors() {
        std::cout << "\n--- Testing Constructors ---" << std::endl;

        // 默认构造
        Vec<3> v1;
        assert_true(v1.size() == 3, "Default constructor creates vector of size 3");
        assert_true(v1.sum() == 0.0f, "Default constructor zero vector");

        // 初始化列表
        Vec<3> v2({1.0f, 2.0f, 3.0f});
        assert_equal(v2[0], 1.0f, 1e-6f, "Initializer list works");
        assert_equal(v2[1], 2.0f, 1e-6f, "Initializer list works");
        assert_equal(v2[2], 3.0f, 1e-6f, "Initializer list works");

        // 零向量
        Vec<3> zero;
        zero.setZero();
        assert_equal(zero[0], 0.0f, 1e-6f, "setZero() works");
        assert_equal(zero[1], 0.0f, 1e-6f, "setZero() works");
        assert_equal(zero[2], 0.0f, 1e-6f, "setZero() works");

        // 常量向量
        Vec<3> constant;
        constant.setConstant(5.0f);
        assert_equal(constant[0], 5.0f, 1e-6f, "setConstant() works");
        assert_equal(constant[1], 5.0f, 1e-6f, "setConstant() works");
        assert_equal(constant[2], 5.0f, 1e-6f, "setConstant() works");
    }

    void test_arithmetic() {
        std::cout << "\n--- Testing Arithmetic ---" << std::endl;

        Vec<3> v1({1.0f, 2.0f, 3.0f});
        Vec<3> v2({4.0f, 5.0f, 6.0f});

        // 加法
        Vec<3> sum = v1 + v2;
        assert_equal(sum[0], 5.0f, 1e-6f, "Vector addition works");
        assert_equal(sum[1], 7.0f, 1e-6f, "Vector addition works");
        assert_equal(sum[2], 9.0f, 1e-6f, "Vector addition works");

        // 减法
        Vec<3> diff = v1 - v2;
        assert_equal(diff[0], -3.0f, 1e-6f, "Vector subtraction works");
        assert_equal(diff[1], -3.0f, 1e-6f, "Vector subtraction works");
        assert_equal(diff[2], -3.0f, 1e-6f, "Vector subtraction works");

        // 标量乘法
        Vec<3> scaled = 2.0f * v1;
        assert_equal(scaled[0], 2.0f, 1e-6f, "Scalar multiplication works");
        assert_equal(scaled[1], 4.0f, 1e-6f, "Scalar multiplication works");
        assert_equal(scaled[2], 6.0f, 1e-6f, "Scalar multiplication works");

        // 标量除法
        Vec<3> divided = v1 / 2.0f;
        assert_equal(divided[0], 0.5f, 1e-6f, "Scalar division works");
        assert_equal(divided[1], 1.0f, 1e-6f, "Scalar division works");
        assert_equal(divided[2], 1.5f, 1e-6f, "Scalar division works");

        // 点积
        float dot = v1 * v2;
        assert_equal(dot, 32.0f, 1e-6f, "Dot product works");
    }

    void test_stats() {
        std::cout << "\n--- Testing Statistics ---" << std::endl;

        Vec<3> v1({1.0f, 2.0f, 3.0f});

        // Sum
        assert_equal(v1.sum(), 6.0f, 1e-6f, "Sum works");

        // Max/Min
        assert_equal(v1.maxCoeff(), 3.0f, 1e-6f, "MaxCoeff works");
        assert_equal(v1.minCoeff(), 1.0f, 1e-6f, "MinCoeff works");

        // Norm
        float norm = v1.norm();
        assert_equal(norm, std::sqrt(14.0f), 1e-6f, "Norm works");

        // Squared Norm
        float sqNorm = v1.squaredNorm();
        assert_equal(sqNorm, 14.0f, 1e-6f, "SquaredNorm works");
    }

    void test_setters() {
        std::cout << "\n--- Testing Setters ---" << std::endl;

        Vec<3> v;
        v[0] = 1.0f;
        v[1] = 2.0f;
        v[2] = 3.0f;
        assert_equal(v[0], 1.0f, 1e-6f, "Direct assignment works");
        assert_equal(v[1], 2.0f, 1e-6f, "Direct assignment works");
        assert_equal(v[2], 3.0f, 1e-6f, "Direct assignment works");
    }

    void test_edge_cases() {
        std::cout << "\n--- Testing Edge Cases ---" << std::endl;

        // 零向量
        Vec<3> zero;
        zero.setZero();
        assert_equal(zero.norm(), 0.0f, 1e-6f, "Zero vector norm is zero");
        assert_equal(zero.sum(), 0.0f, 1e-6f, "Zero vector sum is zero");

        // 单位向量
        Vec<3> unit({1.0f, 0.0f, 0.0f});
        assert_equal(unit.norm(), 1.0f, 1e-6f, "Unit vector norm is 1");

        // 负向量
        Vec<3> neg({-1.0f, -2.0f, -3.0f});
        assert_equal(neg.norm(), std::sqrt(14.0f), 1e-6f, "Negative vector norm works");
        assert_equal(neg.maxCoeff(), -1.0f, 1e-6f, "MaxCoeff on negative vector");

        // 大值
        Vec<3> large({1e6f, 2e6f, 3e6f});
        assert_equal(large.norm(), std::sqrt(14e12f), 1e-6f, "Large values work");
    }

    void test_precision() {
        std::cout << "\n--- Testing Precision ---" << std::endl;

        Vec<3> v1({0.1f, 0.2f, 0.3f});
        Vec<3> v2({0.01f, 0.02f, 0.03f});

        Vec<3> sum = v1 + v2;
        assert_equal(sum[0], 0.11f, 1e-5f, "High precision addition");
        assert_equal(sum[1], 0.22f, 1e-5f, "High precision addition");
        assert_equal(sum[2], 0.33f, 1e-5f, "High precision addition");

        Vec<3> scaled = v1 * 1000.0f;
        assert_equal(scaled[0], 100.0f, 1e-3f, "Large scale multiplication");
        assert_equal(scaled[1], 200.0f, 1e-3f, "Large scale multiplication");
        assert_equal(scaled[2], 300.0f, 1e-3f, "Large scale multiplication");
    }

    void test_random_values() {
        std::cout << "\n--- Testing Random Values ---" << std::endl;

        std::mt19937 gen(42);  // 固定种子
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

        Vec<3> v;
        for (int i = 0; i < 1000; ++i) {
            v[i % 3] = dist(gen);
        }

        // 验证范围
        for (int i = 0; i < 3; ++i) {
            assert_true(v[i] >= -100.0f && v[i] <= 100.0f, "Random values in range");
        }

        // 验证非零范数 (大部分情况)
        float norm = v.norm();
        assert_true(norm > 0.0f, "Random vector has non-zero norm");

        std::cout << "  Tested 1000 random vectors" << std::endl;
    }
};

} // namespace test
} // namespace hlcl
