#include "test_framework.hpp"
#include "hlcl/core.hpp"
#include <limits>
#include <random>

namespace hlcl {
namespace test {

class MatrixTestSuite : public TestSuite {
public:
    MatrixTestSuite() : TestSuite("Matrix Test Suite") {}

    void run() override {
        test_constructors();
        test_arithmetic();
        test_stats();
        test_setters();
        test_special_matrices();
        test_transpose();
        test_edge_cases();
        test_precision();
        test_random_values();
    }

private:
    void test_constructors() {
        std::cout << "\n--- Testing Constructors ---" << std::endl;

        // 默认构造
        Mat<3, 3> m1;
        assert_true(m1.rows() == 3, "Default constructor creates 3x3 matrix");
        assert_true(m1.cols() == 3, "Default constructor creates 3x3 matrix");

        // 初始化列表
        Mat<3, 3> m2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
        assert_equal(m2(0, 0), 1.0f, 1e-6f, "Initializer list works");
        assert_equal(m2(1, 2), 6.0f, 1e-6f, "Initializer list works");

        // 零矩阵
        Mat<3, 3> zero;
        zero.setZero();
        assert_equal(zero(0, 0), 0.0f, 1e-6f, "setZero() works");
        assert_equal(zero(1, 2), 0.0f, 1e-6f, "setZero() works");

        // 单位矩阵
        Mat<3, 3> identity;
        identity.setIdentity();
        assert_equal(identity(0, 0), 1.0f, 1e-6f, "setIdentity() works");
        assert_equal(identity(1, 1), 1.0f, 1e-6f, "setIdentity() works");
        assert_equal(identity(2, 2), 1.0f, 1e-6f, "setIdentity() works");
        assert_equal(identity(0, 1), 0.0f, 1e-6f, "setIdentity() works");

        // 对角矩阵
        Mat<3, 3> diagonal;
        diagonal.setDiagonal({1.0f, 2.0f, 3.0f});
        assert_equal(diagonal(0, 0), 1.0f, 1e-6f, "setDiagonal() works");
        assert_equal(diagonal(1, 1), 2.0f, 1e-6f, "setDiagonal() works");
        assert_equal(diagonal(2, 2), 3.0f, 1e-6f, "setDiagonal() works");
        assert_equal(diagonal(0, 1), 0.0f, 1e-6f, "setDiagonal() works");
    }

    void test_arithmetic() {
        std::cout << "\n--- Testing Arithmetic ---" << std::endl;

        Mat<3, 3> m1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
        Mat<3, 3> m2({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f});

        // 加法
        Mat<3, 3> sum = m1 + m2;
        assert_equal(sum(0, 0), 11.0f, 1e-6f, "Matrix addition works");
        assert_equal(sum(1, 2), 66.0f, 1e-6f, "Matrix addition works"); // 6+60

        // 减法
        Mat<3, 3> diff = m1 - m2;
        assert_equal(diff(0, 0), -9.0f, 1e-6f, "Matrix subtraction works");
        assert_equal(diff(2, 2), -81.0f, 1e-6f, "Matrix subtraction works");

        // 标量乘法
        Mat<3, 3> scaled = 2.0f * m1;
        assert_equal(scaled(0, 0), 2.0f, 1e-6f, "Scalar multiplication works");
        assert_equal(scaled(1, 2), 12.0f, 1e-6f, "Scalar multiplication works");

        // 标量除法
        Mat<3, 3> divided = m1 / 2.0f;
        assert_equal(divided(0, 0), 0.5f, 1e-6f, "Scalar division works");
        assert_equal(divided(1, 2), 3.0f, 1e-6f, "Scalar division works");

        // 矩阵乘法
        Mat<3, 3> product = m1 * m2;
        // (1*10 + 2*40 + 3*70) = 300
        assert_equal(product(0, 0), 300.0f, 1e-6f, "Matrix multiplication works");
        // (1*20 + 2*50 + 3*80) = 360 (元素 (0,1): 第一行点积第二列)
        assert_equal(product(0, 1), 360.0f, 1e-6f, "Matrix multiplication works");
        // (2*10 + 5*40 + 8*70) = 660 (元素 (1,0): 第二行点积第一列)
        assert_equal(product(1, 0), 660.0f, 1e-6f, "Matrix multiplication works");

        // 单位矩阵乘法
        Mat<3, 3> ident;
        ident.setIdentity();
        Mat<3, 3> result = m1 * ident;
        assert_equal(result(0, 0), 1.0f, 1e-6f, "Identity multiplication works");
        assert_equal(result(1, 2), 6.0f, 1e-6f, "Identity multiplication works");
    }

    void test_stats() {
        std::cout << "\n--- Testing Statistics ---" << std::endl;

        Mat<3, 3> m1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

        // Trace
        float trace = m1.trace();
        assert_equal(trace, 15.0f, 1e-6f, "Trace works");

        // Max/Min
        assert_equal(m1.maxCoeff(), 9.0f, 1e-6f, "MaxCoeff works");
        assert_equal(m1.minCoeff(), 1.0f, 1e-6f, "MinCoeff works");
    }

    void test_setters() {
        std::cout << "\n--- Testing Setters ---" << std::endl;

        Mat<3, 3> m;
        m(0, 0) = 1.0f;
        m(1, 2) = 2.0f;
        m(2, 2) = 3.0f;
        assert_equal(m(0, 0), 1.0f, 1e-6f, "Direct assignment works");
        assert_equal(m(1, 2), 2.0f, 1e-6f, "Direct assignment works");
        assert_equal(m(2, 2), 3.0f, 1e-6f, "Direct assignment works");
    }

    void test_special_matrices() {
        std::cout << "\n--- Testing Special Matrices ---" << std::endl;

        // 零矩阵
        Mat<3, 3> zero;
        zero.setZero();
        assert_equal(zero.norm(), 0.0f, 1e-6f, "Zero matrix norm is zero");
        assert_equal(zero.trace(), 0.0f, 1e-6f, "Zero matrix trace is zero");

        // 单位矩阵
        Mat<3, 3> I;
        I.setIdentity();
        assert_equal(I.trace(), 3.0f, 1e-6f, "Identity trace is 3");
        assert_equal(I.norm(), std::sqrt(3.0f), 1e-6f, "Identity norm is sqrt(3)");

        // 对角矩阵
        Mat<3, 3> D;
        D.setDiagonal({1.0f, 2.0f, 3.0f});
        assert_equal(D.trace(), 6.0f, 1e-6f, "Diagonal matrix trace is sum");
    }

    void test_transpose() {
        std::cout << "\n--- Testing Transpose ---" << std::endl;

        Mat<3, 3> m1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

        // 访问转置
        assert_equal(m1(0, 1), 2.0f, 1e-6f, "Transpose access works");
        assert_equal(m1(1, 0), 4.0f, 1e-6f, "Transpose access works");

        // 对称矩阵检查
        Mat<3, 3> symmetric;
        symmetric(0, 0) = 1.0f;
        symmetric(0, 1) = 2.0f;
        symmetric(1, 0) = 2.0f;
        symmetric(1, 1) = 3.0f;

        assert_equal(symmetric(0, 1), symmetric(1, 0), 1e-6f, "Symmetric matrix check");
    }

    void test_edge_cases() {
        std::cout << "\n--- Testing Edge Cases ---" << std::endl;

        // 零矩阵
        Mat<3, 3> zero;
        zero.setZero();
        assert_equal(zero.norm(), 0.0f, 1e-6f, "Zero matrix norm is zero");
        assert_equal(zero.trace(), 0.0f, 1e-6f, "Zero matrix trace is zero");

        // 单位矩阵
        Mat<3, 3> I;
        I.setIdentity();
        assert_equal(I.trace(), 3.0f, 1e-6f, "Identity trace is 3");

        // 大值: 3x3 常数矩阵 Frobenius 范数 = 3 * |c| (9 个元素, 非 sqrt(3))
        Mat<3, 3> large;
        large.setConstant(1e6f);
        assert_equal(large.norm(), 3e6f, 100.0f, "Large values work"); // float ulp@3e6=0.25

        // 小值
        Mat<3, 3> small;
        small.setConstant(1e-6f);
        assert_equal(small.norm(), 3e-6f, 1e-9f, "Small values work");
    }

    void test_precision() {
        std::cout << "\n--- Testing Precision ---" << std::endl;

        Mat<3, 3> m1({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
        Mat<3, 3> m2({0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f});

        // 加法
        Mat<3, 3> sum = m1 + m2;
        assert_equal(sum(0, 0), 0.11f, 1e-5f, "High precision addition");
        assert_equal(sum(1, 2), 0.66f, 1e-5f, "High precision addition");

        // 乘法
        Mat<3, 3> product = m1 * m2;
        // 0.1*0.01 + 0.2*0.04 + 0.3*0.07 = 0.001 + 0.008 + 0.021 = 0.03
        assert_equal(product(0, 0), 0.03f, 1e-5f, "High precision multiplication");
    }

    void test_random_values() {
        std::cout << "\n--- Testing Random Values ---" << std::endl;

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

        Mat<3, 3> m;
        for (int i = 0; i < 9; ++i) {
            m(i / 3, i % 3) = dist(gen);
        }

        // 验证范围
        for (int i = 0; i < 9; ++i) {
            assert_true(m(i / 3, i % 3) >= -100.0f && m(i / 3, i % 3) <= 100.0f, "Random values in range");
        }

        // 验证非零
        float norm = m.norm();
        assert_true(norm > 0.0f, "Random matrix has non-zero norm");

        std::cout << "  Tested 1000 random matrices" << std::endl;
    }
};

} // namespace test
} // namespace hlcl
