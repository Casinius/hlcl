// release 模式构建 (xmake 定义 NDEBUG) 下 assert 会被编译剔除;
// 测试断言必须始终生效, 故先取消 NDEBUG。
#undef NDEBUG
#include "hlcl/core.hpp"
#include <cassert>
#include <iostream>

using namespace hlcl;

void test_matrix_constructors() {
    std::cout << "Testing matrix constructors..." << std::endl;

    // 默认构造
    Mat<2, 2> M1;
    assert(M1.rows() == 2);
    assert(M1.cols() == 2);
    assert(M1(0, 0) == 0.0f);
    std::cout << "  ✓ Default constructor works" << std::endl;

    // 初始化列表
    Mat<2, 3> M2({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
    assert(M2.rows() == 2);
    assert(M2.cols() == 3);
    assert(M2(0, 0) == 1.0f);
    assert(M2(1, 2) == 6.0f);
    std::cout << "  ✓ Initializer list constructor works" << std::endl;

    // 大小构造
    Mat<3, 3> M3(3);
    assert(M3.rows() == 3);
    assert(M3.cols() == 3);
    std::cout << "  ✓ Size constructor works" << std::endl;
}

void test_matrix_arithmetic() {
    std::cout << "Testing matrix arithmetic..." << std::endl;

    Mat<2, 2> M1;
    M1(0, 0) = 1.0f; M1(0, 1) = 2.0f;
    M1(1, 0) = 3.0f; M1(1, 1) = 4.0f;

    Mat<2, 2> M2;
    M2(0, 0) = 5.0f; M2(0, 1) = 6.0f;
    M2(1, 0) = 7.0f; M2(1, 1) = 8.0f;

    // 加法
    [[maybe_unused]] Mat<2, 2> M3 = M1 + M2;
    assert(M3(0, 0) == 6.0f);
    assert(M3(0, 1) == 8.0f);
    assert(M3(1, 0) == 10.0f);
    assert(M3(1, 1) == 12.0f);
    std::cout << "  ✓ Addition works" << std::endl;

    // 减法
    [[maybe_unused]] Mat<2, 2> M4 = M1 - M2;
    assert(M4(0, 0) == -4.0f);
    assert(M4(0, 1) == -4.0f);
    assert(M4(1, 0) == -4.0f);
    assert(M4(1, 1) == -4.0f);
    std::cout << "  ✓ Subtraction works" << std::endl;

    // 标量乘法
    [[maybe_unused]] Mat<2, 2> M5 = 2.0f * M1;
    assert(M5(0, 0) == 2.0f);
    assert(M5(0, 1) == 4.0f);
    assert(M5(1, 0) == 6.0f);
    assert(M5(1, 1) == 8.0f);
    std::cout << "  ✓ Scalar multiplication works" << std::endl;

    // 矩阵乘法
    [[maybe_unused]] Mat<2, 2> M6 = M1 * M2;
    assert(M6(0, 0) == 19.0f);
    assert(M6(0, 1) == 22.0f);
    assert(M6(1, 0) == 43.0f);
    assert(M6(1, 1) == 50.0f);
    std::cout << "  ✓ Matrix multiplication works" << std::endl;
}

void test_matrix_setters() {
    std::cout << "Testing matrix setters..." << std::endl;

    Mat<2, 2> M;
    M.setZero();
    assert(M(0, 0) == 0.0f);
    assert(M(0, 1) == 0.0f);
    assert(M(1, 0) == 0.0f);
    assert(M(1, 1) == 0.0f);
    std::cout << "  ✓ setZero works" << std::endl;

    M.setIdentity();
    assert(M(0, 0) == 1.0f);
    assert(M(0, 1) == 0.0f);
    assert(M(1, 0) == 0.0f);
    assert(M(1, 1) == 1.0f);
    std::cout << "  ✓ setIdentity works" << std::endl;

    M.setConstant(5.0f);
    assert(M(0, 0) == 5.0f);
    assert(M(0, 1) == 5.0f);
    assert(M(1, 0) == 5.0f);
    assert(M(1, 1) == 5.0f);
    std::cout << "  ✓ setConstant works" << std::endl;

    // 设置行
    Vec<3> row({1.0f, 2.0f, 3.0f});
    Mat<2, 3> M2;
    M2.setRow(0, row);
    assert(M2(0, 0) == 1.0f);
    assert(M2(0, 1) == 2.0f);
    assert(M2(0, 2) == 3.0f);
    std::cout << "  ✓ setRow works" << std::endl;

    // 设置列
    Vec<2> col({4.0f, 5.0f});
    M2.setCol(2, col);
    assert(M2(0, 2) == 4.0f);
    assert(M2(1, 2) == 5.0f);
    std::cout << "  ✓ setCol works" << std::endl;
}

void test_matrix_stats() {
    std::cout << "Testing matrix statistics..." << std::endl;

    Mat<2, 2> M1;
    M1(0, 0) = 1.0f; M1(0, 1) = 2.0f;
    M1(1, 0) = 3.0f; M1(1, 1) = 4.0f;

    // 求和
    assert(M1.sum() == 10.0f);
    std::cout << "  ✓ Sum works" << std::endl;

    // 最大/最小值
    assert(M1.maxCoeff() == 4.0f);
    assert(M1.minCoeff() == 1.0f);
    std::cout << "  ✓ Max/Min works" << std::endl;

    // 范数
    [[maybe_unused]] float norm = M1.norm();
    assert(std::abs(norm - 5.477226f) < 1e-5f);
    std::cout << "  ✓ Norm works" << std::endl;

    // 平方范数
    [[maybe_unused]] float sqNorm = M1.squaredNorm();
    assert(std::abs(sqNorm - 30.0f) < 1e-5f);
    std::cout << "  ✓ Squared norm works" << std::endl;

    // 迹
    assert(M1.trace() == 5.0f);
    std::cout << "  ✓ Trace works" << std::endl;
}

void test_matrix_transpose() {
    std::cout << "Testing matrix transpose..." << std::endl;

    Mat<2, 3> M1;
    M1(0, 0) = 1.0f; M1(0, 1) = 2.0f; M1(0, 2) = 3.0f;
    M1(1, 0) = 4.0f; M1(1, 1) = 5.0f; M1(1, 2) = 6.0f;

    [[maybe_unused]] Mat<3, 2> M1_T = transpose(M1);

    assert(M1_T.rows() == 3);
    assert(M1_T.cols() == 2);
    assert(M1_T(0, 0) == 1.0f);
    assert(M1_T(2, 1) == 6.0f);

    // 验证转置
    assert(M1_T(0, 1) == M1(1, 0));
    assert(M1_T(1, 0) == M1(0, 1));
    assert(M1_T(1, 1) == M1(1, 1));
    assert(M1_T(2, 0) == M1(0, 2));

    std::cout << "  ✓ Transpose works" << std::endl;
}

void test_matrix_special() {
    std::cout << "Testing special matrices..." << std::endl;

    // 单位矩阵
    [[maybe_unused]] Mat<3, 3> I = identity<float, 3>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            assert((i == j) ? I(i, j) == 1.0f : I(i, j) == 0.0f);
        }
    }
    std::cout << "  ✓ Identity matrix works" << std::endl;

    // 对角矩阵
    Vec<3> diag({1.0f, 2.0f, 3.0f});
    [[maybe_unused]] Mat<3, 3> D = diagonal_matrix(diag);
    assert(D(0, 0) == 1.0f);
    assert(D(1, 1) == 2.0f);
    assert(D(2, 2) == 3.0f);
    assert(D(0, 1) == 0.0f);
    assert(D(1, 0) == 0.0f);
    std::cout << "  ✓ Diagonal matrix works" << std::endl;
}

void test_vector_matrix_ops() {
    std::cout << "Testing vector-matrix operations..." << std::endl;

    // 2x3 矩阵
    Mat<2, 3> A;
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    Vec<3> v({1.0f, 2.0f, 3.0f});

    // A (2x3) * v (3) -> (2): 行点积 {14, 32}
    Vec<2> result1 = matTimesVec(A, v);
    assert(result1[0] == 14.0f);
    assert(result1[1] == 32.0f);
    std::cout << "  ✓ mat * vec works" << std::endl;

    // 3x3 情形: A3 * v = {14, 32, 50} (原始数值, 第三行参与)
    Mat<3, 3> A3;
    A3.setRow(0, Vec<3>({1.0f, 2.0f, 3.0f}));
    A3.setRow(1, Vec<3>({4.0f, 5.0f, 6.0f}));
    A3.setRow(2, Vec<3>({7.0f, 8.0f, 9.0f}));
    Vec<3> result3 = matTimesVec(A3, v);
    assert(result3[0] == 14.0f);
    assert(result3[1] == 32.0f);
    assert(result3[2] == 50.0f);
    std::cout << "  ✓ 3x3 mat * vec works" << std::endl;

    // v^T * A^T (3x2) -> (2): 行向量乘矩阵 = (A v)^T = {14, 32}
    Vec<2> result2 = vecTimesMat(v, transpose(A));
    assert(result2[0] == 14.0f);
    assert(result2[1] == 32.0f);
    std::cout << "  ✓ vec * mat^T works" << std::endl;
}

int main() {
    std::cout << "=== AVBD Matrix Tests ===" << std::endl << std::endl;

    test_matrix_constructors();
    test_matrix_arithmetic();
    test_matrix_setters();
    test_matrix_stats();
    test_matrix_transpose();
    test_matrix_special();
    test_vector_matrix_ops();

    std::cout << "\n✓ All matrix tests passed!" << std::endl;

    return 0;
}
