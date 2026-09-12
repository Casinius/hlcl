/**
 * Error Handling Test Suite (移植到真实 avbd API)
 * 测试错误处理和无效输入: 零向量、奇异矩阵、除零、单位矩阵性质
 */

#include "test_framework.hpp"
#include "hlcl/core.hpp"
#include <cmath>
#include <limits>

using namespace hlcl;
using namespace hlcl::test;

// 本地辅助 (0 向量 -> 零向量, 不抛异常; 与 inverse 奇异->零矩阵约定一致)
template<int N>
static Vecd<N> normalize_safe(const Vecd<N>& v) {
    double len = v.norm();
    if (len == 0.0) return v;
    return v / len;
}

template<int R, int C>
static bool matrix_is_zero(const Matrix<double, R, C, Backend::CPU>& m) {
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            if (m(i, j) != 0.0) return false;
    return true;
}

static bool matrix_is_identity(const Matd<2, 2>& m) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(m(i, j) - expected) > 1e-12) return false;
        }
    }
    return true;
}

template<typename T>
static T safe_divide(T numerator, T denominator) {
    if (denominator == 0.0) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    return numerator / denominator;
}

class ErrorHandlingTestSuite : public TestSuite {
public:
    ErrorHandlingTestSuite() : TestSuite("Error Handling Test Suite") {}

    void run() override {
        test_zero_vector_length();
        test_zero_vector_normalize();
        test_singular_matrix();
        test_zero_matrix_inverse();
        test_matrix_division();
        test_scalar_division_by_zero();
        test_vector_division_by_zero();
        test_matrix_multiply_by_zero();
        test_zero_matrix_determinant();
        test_zero_matrix_transpose();
        test_vector_add_zero();
        test_vector_dot_zero();
        test_matrix_add_zero();
        test_inverse_matrix_multiplication();
        test_identity_matrix();
        test_matrix_norm();
    }

private:
    void test_zero_vector_length() {
        std::cout << "Testing zero vector length..." << std::endl;

        Vecd<3> zero({0.0, 0.0, 0.0});
        double length = zero.norm();
        assert_true(length == 0.0, "zero vector length == 0");

        std::cout << "  ✓ Zero vector length test passed" << std::endl;
    }

    void test_zero_vector_normalize() {
        std::cout << "Testing zero vector normalize..." << std::endl;

        Vecd<3> zero({0.0, 0.0, 0.0});
        Vecd<3> normalized = normalize_safe(zero);
        assert_true(normalized.norm() == 0.0, "zero normalized -> zero");

        Vecd<3> unit({3.0, 4.0, 0.0});
        Vecd<3> u = normalize_safe(unit);
        assert_true(std::abs(u.norm() - 1.0) < 1e-12, "unit length after normalize");

        std::cout << "  ✓ Zero vector normalize test passed" << std::endl;
    }

    void test_singular_matrix() {
        std::cout << "Testing singular matrix..." << std::endl;

        Matd<2, 2> singular({{1.0, 2.0}, {3.0, 6.0}});
        double det = determinant(singular);
        assert_true(det == 0.0, "singular matrix determinant == 0");

        Matd<2, 2> inv = inverse(singular);
        assert_true(matrix_is_zero(inv), "singular matrix inverse -> zero matrix");

        Matd<2, 2> product = singular * inv;
        assert_true(matrix_is_zero(product), "A * A^-1(singular) -> zero");

        std::cout << "  ✓ Singular matrix test passed" << std::endl;
    }

    void test_zero_matrix_inverse() {
        std::cout << "Testing zero matrix inverse..." << std::endl;

        // 全零矩阵: scale == 0 提前出口, 契约仍为返回零矩阵
        Matd<2, 2> zero({{0.0, 0.0}, {0.0, 0.0}});
        Matd<2, 2> inv = inverse(zero);
        assert_true(matrix_is_zero(inv), "inverse of zero matrix -> zero matrix");

        std::cout << "  ✓ Zero matrix inverse test passed" << std::endl;
    }

    void test_matrix_division() {
        std::cout << "Testing matrix division (A * B^-1)..." << std::endl;

        Matd<2, 2> A({{2.0, 0.0}, {0.0, 2.0}});
        Matd<2, 2> B({{1.0, 0.0}, {0.0, 1.0}});

        Matd<2, 2> inv_B = inverse(B);
        Matd<2, 2> division = A * inv_B;

        assert_true(std::abs(division(0, 0) - 2.0) < 1e-10, "A / B diagonal");
        assert_true(std::abs(division(0, 1)) < 1e-10, "A / B off-diagonal");

        std::cout << "  ✓ Matrix division test passed" << std::endl;
    }

    void test_scalar_division_by_zero() {
        std::cout << "Testing scalar division by zero..." << std::endl;

        double result = safe_divide(10.0, 0.0);
        assert_true(std::isnan(result), "safe_divide by zero -> NaN");

        std::cout << "  ✓ Scalar division by zero test passed" << std::endl;
    }

    void test_vector_division_by_zero() {
        std::cout << "Testing vector division by zero..." << std::endl;

        Vecd<3> one({1.0, 1.0, 1.0});
        Vecd<3> divided = one / 0.0;
        assert_true(std::isinf(divided.x()) && std::isinf(divided.y()) && std::isinf(divided.z()),
                    "vector / 0 -> inf components (IEEE)");

        std::cout << "  ✓ Vector division by zero test passed" << std::endl;
    }

    void test_matrix_multiply_by_zero() {
        std::cout << "Testing matrix multiply by zero..." << std::endl;

        Matd<2, 2> A({{1.0, 2.0}, {3.0, 4.0}});
        Matd<2, 2> zero({{0.0, 0.0}, {0.0, 0.0}});

        Matd<2, 2> result = A * zero;
        assert_true(matrix_is_zero(result), "A * 0 -> zero");

        std::cout << "  ✓ Matrix multiply by zero test passed" << std::endl;
    }

    void test_zero_matrix_determinant() {
        std::cout << "Testing zero matrix determinant..." << std::endl;

        Matd<2, 2> zero({{0.0, 0.0}, {0.0, 0.0}});
        double det = determinant(zero);
        assert_true(det == 0.0, "det(0) == 0");

        std::cout << "  ✓ Zero matrix determinant test passed" << std::endl;
    }

    void test_zero_matrix_transpose() {
        std::cout << "Testing zero matrix transpose..." << std::endl;

        Matd<2, 2> zero({{0.0, 0.0}, {0.0, 0.0}});
        Matd<2, 2> transposed = transpose(zero);
        assert_true(matrix_is_zero(transposed), "transpose(0) == 0");

        std::cout << "  ✓ Zero matrix transpose test passed" << std::endl;
    }

    void test_vector_add_zero() {
        std::cout << "Testing vector add zero..." << std::endl;

        Vecd<3> one({1.0, 2.0, 3.0});
        Vecd<3> zero({0.0, 0.0, 0.0});
        Vecd<3> result = one + zero;
        assert_true(result.x() == 1.0 && result.y() == 2.0 && result.z() == 3.0,
                    "v + 0 == v");

        std::cout << "  ✓ Vector add zero test passed" << std::endl;
    }

    void test_vector_dot_zero() {
        std::cout << "Testing vector dot zero..." << std::endl;

        Vecd<3> one({1.0, 2.0, 3.0});
        Vecd<3> zero({0.0, 0.0, 0.0});
        double dot = one * zero;
        assert_true(dot == 0.0, "v . 0 == 0");

        std::cout << "  ✓ Vector dot zero test passed" << std::endl;
    }

    void test_matrix_add_zero() {
        std::cout << "Testing matrix add zero..." << std::endl;

        Matd<2, 2> A({{1.0, 2.0}, {3.0, 4.0}});
        Matd<2, 2> zero({{0.0, 0.0}, {0.0, 0.0}});
        Matd<2, 2> result = A + zero;
        assert_true(result(0, 0) == 1.0 && result(1, 1) == 4.0, "A + 0 == A");

        std::cout << "  ✓ Matrix add zero test passed" << std::endl;
    }

    void test_inverse_matrix_multiplication() {
        std::cout << "Testing inverse matrix multiplication..." << std::endl;

        Matd<2, 2> A({{2.0, 0.0}, {0.0, 2.0}});
        Matd<2, 2> inv_A = inverse(A);

        Matd<2, 2> result = A * inv_A;
        assert_true(matrix_is_identity(result), "A * A^-1 == I");

        Matd<2, 2> result2 = inv_A * A;
        assert_true(matrix_is_identity(result2), "A^-1 * A == I");

        std::cout << "  ✓ Inverse matrix multiplication test passed" << std::endl;
    }

    void test_identity_matrix() {
        std::cout << "Testing identity matrix properties..." << std::endl;

        Matd<2, 2> I = identity<double, 2>();
        assert_true(matrix_is_identity(I), "identity is identity");

        Matd<2, 2> A({{2.0, 1.0}, {1.0, 2.0}});
        Matd<2, 2> r1 = I * A;
        Matd<2, 2> r2 = A * I;
        bool eq1 = std::abs(r1(0, 0) - 2.0) < 1e-12 && std::abs(r1(1, 1) - 2.0) < 1e-12;
        bool eq2 = std::abs(r2(0, 0) - 2.0) < 1e-12 && std::abs(r2(1, 1) - 2.0) < 1e-12;
        assert_true(eq1, "I * A == A");
        assert_true(eq2, "A * I == A");

        std::cout << "  ✓ Identity matrix test passed" << std::endl;
    }

    void test_matrix_norm() {
        std::cout << "Testing matrix norm..." << std::endl;

        Matd<2, 2> A({{1.0, 2.0}, {3.0, 4.0}});
        double norm = A.norm();
        assert_true(norm > 0.0, "matrix norm > 0");
        assert_true(std::abs(norm - std::sqrt(30.0)) < 1e-12, "Frobenius norm == sqrt(30)");

        std::cout << "  ✓ Matrix norm test passed" << std::endl;
    }
};

int main() {
    TestRunner runner;

    runner.add_suite(std::make_unique<ErrorHandlingTestSuite>());

    int code = runner.run_all();

    return code;
}
