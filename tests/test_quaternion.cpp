/**
 * Quaternion Test Suite
 * 四元数旋转正确性: 轴角旋转、欧拉角往返、旋转矩阵一致性、代数性质。
 * 另含 double 精度存储契约测试 (P3 前该测试暴露 float 存储缺陷, 用于钉住修复)。
 */

#include "test_framework.hpp"
#include "hlcl/quaternion.hpp"
#include <cmath>

using namespace hlcl;
using namespace hlcl::test;

namespace {

constexpr double kPi = 3.14159265358979323846;

template<typename T>
static bool near(T a, T b, T tol) {
    return std::abs(a - b) < tol;
}

// 逐分量比对旋转结果
static bool vec_near(const Vec3& a, const Vec3& b, float tol) {
    return near(a[0], b[0], tol) && near(a[1], b[1], tol) && near(a[2], b[2], tol);
}

static bool mat_near(const Mat<3, 3>& a, const Mat<3, 3>& b, float tol) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!near(a(i, j), b(i, j), tol)) return false;
    return true;
}

class QuaternionTestSuite : public TestSuite {
public:
    QuaternionTestSuite() : TestSuite("Quaternion Test Suite") {}

    void run() override {
        test_identity();
        test_rotate_axes();
        test_euler_roundtrip();
        test_rotation_matrix_consistency();
        test_composition();
        test_algebraic_properties();
        test_equality_tolerance();
        test_double_storage_precision();
    }

private:
    void test_identity() {
        std::cout << "Testing identity quaternion..." << std::endl;

        auto q = quaternionIdentity<float, Backend::CPU>();
        assert_true(near(q.w(), 1.0f, 1e-7f) && near(q.x(), 0.0f, 1e-7f) &&
                        near(q.y(), 0.0f, 1e-7f) && near(q.z(), 0.0f, 1e-7f),
                    "identity = (1,0,0,0)");
        assert_true(vec_near(q.rotate(Vec3({1.0f, 2.0f, 3.0f})),
                             Vec3({1.0f, 2.0f, 3.0f}), 1e-6f),
                    "identity rotates vectors unchanged");

        std::cout << "  ✓ Identity test passed" << std::endl;
    }

    void test_rotate_axes() {
        std::cout << "Testing axis rotations..." << std::endl;

        const float half = static_cast<float>(kPi) / 2.0f;

        // 绕 X 轴 +90°: y -> z, z -> -y
        auto qx = Quaternion<float, Backend::CPU>::fromEuler(Vec3({half, 0.0f, 0.0f}));
        assert_true(vec_near(qx.rotate(Vec3({0.0f, 1.0f, 0.0f})), Vec3({0.0f, 0.0f, 1.0f}), 1e-5f),
                    "X+90: y -> z");
        assert_true(vec_near(qx.rotate(Vec3({0.0f, 0.0f, 1.0f})), Vec3({0.0f, -1.0f, 0.0f}), 1e-5f),
                    "X+90: z -> -y");

        // 绕 Z 轴 +90°: x -> y
        auto qz = Quaternion<float, Backend::CPU>::fromEuler(Vec3({0.0f, 0.0f, half}));
        assert_true(vec_near(qz.rotate(Vec3({1.0f, 0.0f, 0.0f})), Vec3({0.0f, 1.0f, 0.0f}), 1e-5f),
                    "Z+90: x -> y");

        // 长度不变性
        assert_true(vec_near(qz.rotate(Vec3({3.0f, 4.0f, 0.0f})),
                             Vec3({-4.0f, 3.0f, 0.0f}), 1e-5f),
                    "Z+90 preserves length");

        std::cout << "  ✓ Axis rotation test passed" << std::endl;
    }

    void test_euler_roundtrip() {
        std::cout << "Testing Euler roundtrip..." << std::endl;

        const Vec3 euler({0.3f, -0.2f, 0.5f});
        auto q = Quaternion<float, Backend::CPU>::fromEuler(euler);
        Vec3 back = q.toEuler();
        assert_true(vec_near(back, euler, 1e-5f), "fromEuler -> toEuler roundtrip");

        assert_true(near(q.magnitude(), 1.0f, 1e-6f), "fromEuler yields unit quaternion");

        std::cout << "  ✓ Euler roundtrip test passed" << std::endl;
    }

    void test_rotation_matrix_consistency() {
        std::cout << "Testing rotation matrix consistency..." << std::endl;

        const float half = static_cast<float>(kPi) / 2.0f;
        auto qz = Quaternion<float, Backend::CPU>::fromEuler(Vec3({0.0f, 0.0f, half}));

        Mat<3, 3> r = qz.toRotationMatrix();
        // R(qz+90) 作用在 x 轴
        Vec3 col0({r(0, 0), r(1, 0), r(2, 0)});
        assert_true(vec_near(col0, Vec3({0.0f, 1.0f, 0.0f}), 1e-5f),
                    "R(q)*x == rotate(x)");

        // 正交性: R^T R = I
        Mat<3, 3> rt = transpose(r);
        Mat<3, 3> rtr = rt * r;
        Mat<3, 3> eye = identity<float, 3>();
        assert_true(mat_near(rtr, eye, 1e-5f), "R is orthogonal");

        std::cout << "  ✓ Rotation matrix consistency test passed" << std::endl;
    }

    void test_composition() {
        std::cout << "Testing quaternion composition..." << std::endl;

        const float half = static_cast<float>(kPi) / 2.0f;
        auto qz = Quaternion<float, Backend::CPU>::fromEuler(Vec3({0.0f, 0.0f, half}));
        auto qx = Quaternion<float, Backend::CPU>::fromEuler(Vec3({half, 0.0f, 0.0f}));

        // R(q1*q2) == R(q1)*R(q2)
        Mat<3, 3> r_qs = (qz * qx).toRotationMatrix();
        Mat<3, 3> r_prod = qz.toRotationMatrix() * qx.toRotationMatrix();
        assert_true(mat_near(r_qs, r_prod, 1e-5f), "R(q1*q2) == R(q1)*R(q2)");

        std::cout << "  ✓ Composition test passed" << std::endl;
    }

    void test_algebraic_properties() {
        std::cout << "Testing algebraic properties..." << std::endl;

        Quaternion<float, Backend::CPU> q(1.0f, 1.0f, 1.0f, 1.0f);
        assert_true(near(q.magnitude(), 2.0f, 1e-6f), "magnitude(1,1,1,1) == 2");

        // q * conj(q) = (|q|^2, 0, 0, 0)
        auto qc = q.conjugate();
        auto p = q * qc;
        assert_true(near(p.w(), 4.0f, 1e-5f) && near(p.x(), 0.0f, 1e-5f) &&
                        near(p.y(), 0.0f, 1e-5f) && near(p.z(), 0.0f, 1e-5f),
                    "q*conj(q) = (|q|^2,0,0,0)");

        // normalize 后单位长度且方向不变
        auto n = q.normalize();
        assert_true(near(n.magnitude(), 1.0f, 1e-6f), "normalize gives unit magnitude");
        assert_true(near(n.w(), n.x(), 1e-6f) && near(n.x(), n.y(), 1e-6f) &&
                        near(n.y(), n.z(), 1e-6f),
                    "normalize keeps components proportional");

        std::cout << "  ✓ Algebraic properties test passed" << std::endl;
    }

    void test_equality_tolerance() {
        std::cout << "Testing equality tolerance..." << std::endl;

        Quaternion<float, Backend::CPU> a(1.0f, 0.0f, 0.0f, 0.0f);
        Quaternion<float, Backend::CPU> b(1.0f, 1e-8f, 0.0f, 0.0f);
        Quaternion<float, Backend::CPU> c(0.9f, 0.1f, 0.0f, 0.0f);
        assert_true(a == b, "1e-8 perturbation counts as equal (1e-6 tolerance)");
        assert_true(a != c, "0.1 perturbation counts as unequal");

        std::cout << "  ✓ Equality tolerance test passed" << std::endl;
    }

    void test_double_storage_precision() {
        std::cout << "Testing double storage precision..." << std::endl;

        // double 四元数必须以 double 存储分量 (历史缺陷: 内部存成 float,
        // 1e-12 量级的分量会被舍入丢失)
        const double off = 1e-12;
        Quaternion<double, Backend::CPU> qd(1.0 + off, 2.0 + off, 3.0 + off, 4.0 + off);
        assert_true(qd.w() == 1.0 + off, "double storage: w keeps 1e-12 offset");
        assert_true(qd.x() == 2.0 + off, "double storage: x keeps 1e-12 offset");
        assert_true(qd.y() == 3.0 + off, "double storage: y keeps 1e-12 offset");
        assert_true(qd.z() == 4.0 + off, "double storage: z keeps 1e-12 offset");

        std::cout << "  ✓ Double storage precision test passed" << std::endl;
    }
};

} // namespace

int main() {
    TestRunner runner;
    runner.add_suite(std::make_unique<QuaternionTestSuite>());
    return runner.run_all();
}
