#include "hlcl/core.hpp"
#include <cassert>
#include <iostream>

using namespace hlcl;

void test_vector_constructors() {
    std::cout << "Testing vector constructors..." << std::endl;

    // 默认构造
    Vec<3, hlcl::Backend::CPU> v1;
    assert(v1.size() == 3);
    assert(v1.sum() == 0.0f);
    std::cout << "  ✓ Default constructor works" << std::endl;

    // 初始化列表
    Vec<3, hlcl::Backend::CPU> v2({1.0f, 2.0f, 3.0f});
    assert(v2[0] == 1.0f);
    assert(v2[1] == 2.0f);
    assert(v2[2] == 3.0f);
    std::cout << "  ✓ Initializer list constructor works" << std::endl;

    // 大小构造
    Vec<3, hlcl::Backend::CPU> v3(3);
    assert(v3.size() == 3);
    std::cout << "  ✓ Size constructor works" << std::endl;
}

void test_vector_arithmetic() {
    std::cout << "Testing vector arithmetic..." << std::endl;

    Vec<3, hlcl::Backend::CPU> v1({1.0f, 2.0f, 3.0f});
    Vec<3, hlcl::Backend::CPU> v2({4.0f, 5.0f, 6.0f});

    // 加法
    Vec<3, hlcl::Backend::CPU> v3 = v1 + v2;
    assert(v3[0] == 5.0f);
    assert(v3[1] == 7.0f);
    assert(v3[2] == 9.0f);
    std::cout << "  ✓ Addition works" << std::endl;

    // 减法
    Vec<3, hlcl::Backend::CPU> v4 = v1 - v2;
    assert(v4[0] == -3.0f);
    assert(v4[1] == -3.0f);
    assert(v4[2] == -3.0f);
    std::cout << "  ✓ Subtraction works" << std::endl;

    // 标量乘法
    Vec<3, hlcl::Backend::CPU> v5 = v1 * 2.0f;
    assert(v5[0] == 2.0f);
    assert(v5[1] == 4.0f);
    assert(v5[2] == 6.0f);
    std::cout << "  ✓ Scalar multiplication works" << std::endl;

    // 点积
    float dot = v1.dot(v2);
    assert(dot == 32.0f);
    std::cout << "  ✓ Dot product works" << std::endl;

    // 除法
    Vec<3, hlcl::Backend::CPU> v6 = v1 / 2.0f;
    assert(v6[0] == 0.5f);
    assert(v6[1] == 1.0f);
    assert(v6[2] == 1.5f);
    std::cout << "  ✓ Division works" << std::endl;
}

void test_vector_stats() {
    std::cout << "Testing vector statistics..." << std::endl;

    Vec<3, hlcl::Backend::CPU> v1({1.0f, 2.0f, 3.0f});
    Vec<3, hlcl::Backend::CPU> v2({-1.0f, -2.0f, -3.0f});

    // 求和
    assert(v1.sum() == 6.0f);
    assert(v2.sum() == -6.0f);
    std::cout << "  ✓ Sum works" << std::endl;

    // 最大/最小值
    assert(v1.maxCoeff() == 3.0f);
    assert(v1.minCoeff() == 1.0f);
    assert(v2.maxCoeff() == -1.0f);
    assert(v2.minCoeff() == -3.0f);
    std::cout << "  ✓ Max/Min works" << std::endl;

    // 范数
    float norm = v1.norm();
    assert(std::abs(norm - 3.741657f) < 1e-5f);
    std::cout << "  ✓ Norm works" << std::endl;

    // 平方范数
    float sqNorm = v1.squaredNorm();
    assert(std::abs(sqNorm - 14.0f) < 1e-5f);
    std::cout << "  ✓ Squared norm works" << std::endl;
}

void test_vector_setters() {
    std::cout << "Testing vector setters..." << std::endl;

    Vec<3, hlcl::Backend::CPU> v;
    v.setZero();
    assert(v[0] == 0.0f);
    assert(v[1] == 0.0f);
    assert(v[2] == 0.0f);
    std::cout << "  ✓ setZero works" << std::endl;

    v.setConstant(5.0f);
    assert(v[0] == 5.0f);
    assert(v[1] == 5.0f);
    assert(v[2] == 5.0f);
    std::cout << "  ✓ setConstant works" << std::endl;
}

int main() {
    std::cout << "=== AVBD Vector Tests ===" << std::endl << std::endl;

    test_vector_constructors();
    test_vector_arithmetic();
    test_vector_stats();
    test_vector_setters();

    std::cout << "\n✓ All vector tests passed!" << std::endl;

    return 0;
}
