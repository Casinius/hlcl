#include "test_framework.hpp"
#include "hlcl/core.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>

namespace hlcl {
namespace test {

// Fuzz 测试基类
class FuzzTestSuite : public TestSuite {
public:
    FuzzTestSuite() : TestSuite("Fuzz Test Suite") {}

    void run() override {
        test_vector_fuzz();
        test_matrix_fuzz();
        test_arithmetic_fuzz();
    }

private:
    // 向量 Fuzz 测试
    void test_vector_fuzz() {
        std::cout << "\n--- Vector Fuzz Tests ---" << std::endl;

        const int iterations = 1000;
        for (int i = 0; i < iterations; ++i) {
            // 随机生成向量
            Vec<3> v1, v2;
            for (int j = 0; j < 3; ++j) {
                v1[j] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
                v2[j] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
            }

            // 确保不会除以零
            float inv1 = (v1.norm() > 1e-10f) ? 1.0f / v1.norm() : 1.0f;
            float inv2 = (v2.norm() > 1e-10f) ? 1.0f / v2.norm() : 1.0f;

            // 所有运算都应该不会崩溃
            Vec<3> sum = v1 + v2;
            Vec<3> diff = v1 - v2;
            Vec<3> scaled = v1 * 2.0f;
            Vec<3> divided = v2 / 3.0f;

            // 验证结果在合理范围内
            assert_true(sum[0] >= -2.0f && sum[0] <= 2.0f, "Fuzz sum in range");
            assert_true(diff[0] >= -2.0f && diff[0] <= 2.0f, "Fuzz diff in range");
            assert_true(scaled[0] >= -2.0f && scaled[0] <= 2.0f, "Fuzz scaled in range");
            assert_true(divided[0] >= -1.0f && divided[0] <= 1.0f, "Fuzz divided in range");

            // 范数计算
            float norm1 = v1.norm();
            float norm2 = v2.norm();
            float normSum = sum.norm();
            assert_true(norm1 >= 0.0f && norm1 <= 3.0f, "Fuzz norm in range");
            assert_true(norm2 >= 0.0f && norm2 <= 3.0f, "Fuzz norm in range");

            // 点积
            float dot = v1 * v2;
            assert_true(dot >= -9.0f && dot <= 9.0f, "Fuzz dot product in range");
        }

        std::cout << "  Tested " << iterations << " random vector operations" << std::endl;
    }

    // 矩阵 Fuzz 测试
    void test_matrix_fuzz() {
        std::cout << "\n--- Matrix Fuzz Tests ---" << std::endl;

        const int iterations = 1000;
        for (int i = 0; i < iterations; ++i) {
            // 随机生成矩阵
            Mat<3, 3> m1, m2;
            for (int j = 0; j < 9; ++j) {
                m1(j / 3, j % 3) = rand() / (float)RAND_MAX * 2.0f - 1.0f;
                m2(j / 3, j % 3) = rand() / (float)RAND_MAX * 2.0f - 1.0f;
            }

            // 所有运算都应该不会崩溃
            Mat<3, 3> sum = m1 + m2;
            Mat<3, 3> diff = m1 - m2;
            Mat<3, 3> scaled = m1 * 2.0f;
            Mat<3, 3> multiplied = m1 * m2;

            // 验证结果在合理范围内
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    assert_true(sum(r, c) >= -2.0f && sum(r, c) <= 2.0f, "Fuzz matrix sum in range");
                    assert_true(diff(r, c) >= -2.0f && diff(r, c) <= 2.0f, "Fuzz matrix diff in range");
                    assert_true(scaled(r, c) >= -2.0f && scaled(r, c) <= 2.0f, "Fuzz matrix scaled in range");
                }
            }

            // 矩阵范数
            float norm1 = m1.norm();
            float norm2 = m2.norm();
            float normMult = multiplied.norm();
            assert_true(norm1 >= 0.0f && norm1 <= 9.0f, "Fuzz matrix norm in range");
            assert_true(norm2 >= 0.0f && norm2 <= 9.0f, "Fuzz matrix norm in range");

            // Trace
            float trace1 = m1.trace();
            float trace2 = m2.trace();
            assert_true(trace1 >= -3.0f && trace1 <= 3.0f, "Fuzz matrix trace in range");
        }

        std::cout << "  Tested " << iterations << " random matrix operations" << std::endl;
    }

    // 算术运算 Fuzz 测试
    void test_arithmetic_fuzz() {
        std::cout << "\n--- Arithmetic Fuzz Tests ---" << std::endl;

        const int iterations = 1000;
        for (int i = 0; i < iterations; ++i) {
            // 随机生成向量
            Vec<3> v1, v2;
            for (int j = 0; j < 3; ++j) {
                v1[j] = rand() / (float)RAND_MAX * 100.0f;
                v2[j] = rand() / (float)RAND_MAX * 100.0f;
            }

            // 随机标量
            float s1 = rand() / (float)RAND_MAX * 10.0f;
            float s2 = rand() / (float)RAND_MAX * 10.0f;

            // 加法和减法
            Vec<3> sum = v1 + v2;
            Vec<3> diff = v1 - v2;

            // 标量运算
            Vec<3> sumScalar = v1 + s1;
            Vec<3> diffScalar = v1 - s2;
            Vec<3> scalarMult1 = s1 * v1;
            Vec<3> scalarMult2 = v2 * s2;
            Vec<3> scalarDiv1 = v1 / (s1 + 1e-10f);
            Vec<3> scalarDiv2 = v2 / (s2 + 1e-10f);

            // 验证结果
            assert_true(sum[0] >= -200.0f && sum[0] <= 200.0f, "Fuzz arithmetic sum in range");
            assert_true(diff[0] >= -200.0f && diff[0] <= 200.0f, "Fuzz arithmetic diff in range");
            assert_true(scalarMult1[0] >= 0.0f && scalarMult1[0] <= 1000.0f, "Fuzz scalar mult in range");
            assert_true(std::isfinite(scalarDiv1[0]) && scalarDiv1[0] >= 0.0f,
                        "Fuzz scalar div finite");
        }

        std::cout << "  Tested " << iterations << " random arithmetic operations" << std::endl;
    }
};

// Fuzz 测试辅助函数（用于 libFuzzer）
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 1) return 0;

    // 根据输入种子生成随机数
    int seed = data[0];
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    // 测试各种操作
    Vec<3> v1, v2, v3;
    for (int i = 0; i < 3; ++i) {
        v1[i] = dist(gen);
        v2[i] = dist(gen);
    }

    // 测试向量运算
    Vec<3> sum = v1 + v2;
    Vec<3> diff = v1 - v2;
    Vec<3> scaled = v1 * 2.0f;
    float dot = v1 * v2;

    // 测试矩阵运算
    Mat<3, 3> m1, m2;
    for (int i = 0; i < 9; ++i) {
        m1(i / 3, i % 3) = dist(gen);
        m2(i / 3, i % 3) = dist(gen);
    }

    Mat<3, 3> product = m1 * m2;

    // 测试矩阵-向量运算
    Vec<3> result = matTimesVec(m1, v1);

    // 所有操作都应该完成且不崩溃
    (void)sum;
    (void)diff;
    (void)scaled;
    (void)dot;
    (void)product;
    (void)result;

    return 0;
}

} // namespace test
} // namespace hlcl
