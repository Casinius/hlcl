/**
 * Special Values Test Suite (移植到真实 avbd API)
 * 测试特殊数值: NaN、无穷大、负零、最小/最大值 (IEEE-754 语义)
 */

#include "test_framework.hpp"
#include "hlcl/core.hpp"
#include <cmath>
#include <limits>

using namespace hlcl;
using namespace hlcl::test;

// 辅助谓词
static inline bool is_nan(double v) { return std::isnan(v); }
static inline bool is_inf(double v) { return std::isinf(v); }
static inline bool is_neg_zero(double v) { return std::signbit(v) && v == 0.0; }

class SpecialValuesTestSuite : public TestSuite {
public:
    SpecialValuesTestSuite() : TestSuite("Special Values Test Suite") {}

    void run() override {
        test_nan();
        test_infinity();
        test_negative_zero();
        test_min_max_values();
        test_special_values_boundary();
        test_matrix_infinity();
    }

private:
    void test_nan() {
        std::cout << "Testing NaN values..." << std::endl;

        const double qnan = std::numeric_limits<double>::quiet_NaN();
        Vecd<3> nan_vec1({0.0, qnan, 0.0});
        Vecd<3> nan_vec2({0.0, qnan, 0.0});

        double dot_result = nan_vec1 * nan_vec2;
        assert_true(is_nan(dot_result), "NaN dot product");

        Vecd<3> add_result = nan_vec1 + nan_vec2;
        assert_true(is_nan(add_result.y()), "NaN vector addition (NaN component)");
        assert_true(add_result.x() == 0.0 && add_result.z() == 0.0,
                    "NaN vector addition (finite components stay finite)");

        double length = nan_vec1.norm();
        assert_true(is_nan(length), "NaN vector length");

        std::cout << "  ✓ NaN test passed" << std::endl;
    }

    void test_infinity() {
        std::cout << "Testing infinity values..." << std::endl;

        const double pos_inf = std::numeric_limits<double>::infinity();
        const double neg_inf = -std::numeric_limits<double>::infinity();

        Vecd<3> inf_vec({pos_inf, pos_inf, pos_inf});
        double length = inf_vec.norm();
        assert_true(is_inf(length) && length > 0.0, "infinity length");

        // 每分量 inf + (-inf) = NaN (标量加法作用于各分量)
        Vecd<3> inf_vec2({pos_inf, pos_inf, pos_inf});
        Vecd<3> inf_add = inf_vec2 + neg_inf;
        assert_true(is_nan(inf_add.x()), "inf + (-inf) = NaN");

        Vecd<3> vec_inf1({pos_inf, 0.0, 0.0});
        Vecd<3> vec_inf2({pos_inf, 0.0, 0.0});
        double dot_result = vec_inf1 * vec_inf2;
        assert_true(is_inf(dot_result), "inf dot inf = inf");

        std::cout << "  ✓ Infinity test passed" << std::endl;
    }

    void test_negative_zero() {
        std::cout << "Testing negative zero..." << std::endl;

        const double neg_zero = -0.0;
        Vecd<3> neg_zero_vec({neg_zero, neg_zero, neg_zero});

        // -0.0 * 2 = -0.0 (IEEE 754)
        Vecd<3> scaled = neg_zero_vec * 2.0;
        assert_true(is_neg_zero(scaled.x()), "-0.0 * 2 keeps signbit");

        // -0.0 + 1 = 1
        Vecd<3> one_vec({1.0, 1.0, 1.0});
        Vecd<3> plus_one = neg_zero_vec + one_vec;
        assert_true(plus_one.x() == 1.0 && !std::signbit(plus_one.x()),
                    "-0.0 + 1 = 1");

        // -0.0 + +0.0 = +0.0
        Vecd<3> zero_vec({0.0, 0.0, 0.0});
        Vecd<3> plus_zero = neg_zero_vec + zero_vec;
        assert_true(plus_zero.x() == 0.0 && !std::signbit(plus_zero.x()),
                    "-0.0 + 0.0 = +0.0");

        // -0.0 + -0.0 = -0.0
        Vecd<3> neg_plus = neg_zero_vec + neg_zero_vec;
        assert_true(is_neg_zero(neg_plus.x()), "-0.0 + -0.0 = -0.0");

        std::cout << "  ✓ Negative zero test passed" << std::endl;
    }

    void test_min_max_values() {
        std::cout << "Testing min/max values..." << std::endl;

        const double min = std::numeric_limits<double>::min();
        const double max = std::numeric_limits<double>::max();

        // 原始测试对 min 分量向量取欧氏长度会因平方下溢为 0;
        // 用 sqrt(min) 幅度保持 "极小正长度 > 0" 的语义 (平方 = min 可表示)
        const double tiny = std::sqrt(min);
        Vecd<3> tiny_vec({tiny, tiny, tiny});
        double tiny_length = tiny_vec.norm();
        assert_true(tiny_length > 0.0, "min-magnitude vector length > 0");

        Vecd<3> max_vec({max, max, max});
        double max_length = max_vec.norm();
        assert_true(is_inf(max_length), "max vector length overflows to inf");

        // min + min 可表示且 > 0 (无平方, 不发生下溢)
        Vecd<3> min_vec({min, min, min});
        Vecd<3> min_plus_min = min_vec + min_vec;
        assert_true(min_plus_min.x() > 0.0, "min + min > 0");

        std::cout << "  ✓ Min/Max values test passed" << std::endl;
    }

    void test_special_values_boundary() {
        std::cout << "Testing special values boundary..." << std::endl;

        const double qnan = std::numeric_limits<double>::quiet_NaN();
        const double pos_inf = std::numeric_limits<double>::infinity();
        Vecd<3> nan_vec({0.0, qnan, 0.0});
        Vecd<3> zero_vec({0.0, 0.0, 0.0});

        Vecd<3> nan_plus_zero = nan_vec + zero_vec;
        assert_true(is_nan(nan_plus_zero.y()), "NaN + 0 = NaN");

        Vecd<3> nan_times_zero = nan_vec * 0.0;
        assert_true(is_nan(nan_times_zero.y()), "NaN * 0 = NaN");

        Vecd<3> nan_times_inf = nan_vec * pos_inf;
        assert_true(is_nan(nan_times_inf.y()), "NaN * inf = NaN");

        double nan_div_zero = nan_vec.y() / 0.0;
        assert_true(is_nan(nan_div_zero), "NaN / 0 = NaN");

        std::cout << "  ✓ Special values boundary test passed" << std::endl;
    }

    void test_matrix_infinity() {
        std::cout << "Testing matrix infinity..." << std::endl;

        const double pos_inf = std::numeric_limits<double>::infinity();

        Matd<2, 2> inf_matrix({{pos_inf, 0.0}, {0.0, pos_inf}});

        double det = determinant(inf_matrix);
        assert_true(is_inf(det), "determinant of inf diagonal is inf");

        // 无穷矩阵乘有限向量 -> 无穷分量 (0 * inf 会得到 NaN, 故用有限向量)
        Vecd<2> finite_vec({1.0, 2.0});
        Vecd<2> result = matTimesVec(inf_matrix, finite_vec);
        assert_true(is_inf(result.x()) && is_inf(result.y()),
                    "inf matrix * finite vector = inf");

        // 非有限矩阵的逆未定义: 若求得非有限/零矩阵则跳过 (不要求逆)
        Matd<2, 2> inv = inverse(inf_matrix);
        bool finite_inv = std::isfinite(inv(0, 0)) && std::isfinite(inv(1, 1)) &&
                          std::isfinite(inv(0, 1)) && std::isfinite(inv(1, 0));
        assert_true(!finite_inv || inv(0, 0) == 0.0,
                    "inverse of inf matrix is not a finite regular matrix");

        std::cout << "  ✓ Matrix infinity test passed" << std::endl;
    }
};

int main(int argc, char** argv) {
    TestRunner runner;

    runner.add_suite(std::make_unique<SpecialValuesTestSuite>());

    int code = runner.run_all();

    return code;
}
