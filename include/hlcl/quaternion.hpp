#pragma once

#include "matrix.hpp"
#include "vector.hpp"
#include <cmath>
#include <algorithm>

namespace hlcl {

/// Quaternion representation for 3D rotations
/// Quaternion = [w, x, y, z] where w is the scalar part.
/// Components are stored at precision T (a previous revision stored a
/// Vec<4> = float vector, truncating Quaternion<double> to float accuracy).
template<typename T, Backend B>
class Quaternion {
public:
    using Vec3T = Vector<T, 3, B>;
    using Vec4T = Vector<T, 4, B>;

    Quaternion() noexcept : w_(T{1}), x_(T{0}), y_(T{0}), z_(T{0}) {}  // identity

    Quaternion(T w, T x, T y, T z) noexcept : w_(w), x_(x), y_(y), z_(z) {}

    // Initialize from Euler angles (intrinsic Z-Y-X: roll=x, pitch=y, yaw=z)
    static Quaternion fromEuler(const Vec3T& euler) {
        T cr = std::cos(euler[0] * T{0.5});
        T sr = std::sin(euler[0] * T{0.5});
        T cp = std::cos(euler[1] * T{0.5});
        T sp = std::sin(euler[1] * T{0.5});
        T cy = std::cos(euler[2] * T{0.5});
        T sy = std::sin(euler[2] * T{0.5});

        T w = cr * cp * cy + sr * sp * sy;
        T x = sr * cp * cy - cr * sp * sy;
        T y = cr * sp * cy + sr * cp * sy;
        T z = cr * cp * sy - sr * sp * cy;

        return Quaternion(w, x, y, z);
    }

    // Convert to Euler angles
    [[nodiscard]] Vec3T toEuler() const {
        T sqw = w_ * w_;
        T sqx = x_ * x_;
        T sqy = y_ * y_;
        T sqz = z_ * z_;

        T unit = sqw + sqx + sqy + sqz;
        if (unit > T{0}) {
            unit = T{1} / unit;
        }

        Vec3T euler;
        euler[0] = std::atan2(T{2} * (w_ * x_ + y_ * z_) * unit, T{1} - T{2} * (sqx + sqy) * unit);  // roll (x-axis rotation)
        euler[1] = std::asin(std::clamp(T{2} * (w_ * y_ - x_ * z_) * unit, T{-1}, T{1}));  // pitch (y-axis rotation)
        euler[2] = std::atan2(T{2} * (w_ * z_ + x_ * y_) * unit, T{1} - T{2} * (sqy + sqz) * unit);  // yaw (z-axis rotation)

        return euler;
    }

    // Multiply two quaternions: q * p
    [[nodiscard]] Quaternion operator*(const Quaternion& other) const noexcept {
        T w1 = w_, x1 = x_, y1 = y_, z1 = z_;
        T w2 = other.w_, x2 = other.x_, y2 = other.y_, z2 = other.z_;

        T w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        T x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        T y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        T z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

        return Quaternion(w, x, y, z);
    }

    // Multiply by scalar
    [[nodiscard]] Quaternion operator*(T scalar) const noexcept {
        return Quaternion(w_ * scalar, x_ * scalar, y_ * scalar, z_ * scalar);
    }

    // Multiply by scalar (left)
    [[nodiscard]] friend Quaternion operator*(T scalar, const Quaternion& q) noexcept {
        return q * scalar;
    }

    // Quaternion addition
    [[nodiscard]] Quaternion operator+(const Quaternion& other) const noexcept {
        return Quaternion(w_ + other.w_, x_ + other.x_, y_ + other.y_, z_ + other.z_);
    }

    // Quaternion subtraction
    [[nodiscard]] Quaternion operator-(const Quaternion& other) const noexcept {
        return Quaternion(w_ - other.w_, x_ - other.x_, y_ - other.y_, z_ - other.z_);
    }

    // Quaternion negation (single, non-mutating form; the previous
    // non-const `Quaternion& operator-()` mutated in place surprisingly)
    [[nodiscard]] Quaternion operator-() const noexcept {
        return Quaternion(-w_, -x_, -y_, -z_);
    }

    // Equality within tolerance (default 1e-6, kept for float compatibility)
    [[nodiscard]] bool almost_equal(const Quaternion& other, T tol = T{1e-6}) const {
        return std::abs(w_ - other.w_) < tol && std::abs(x_ - other.x_) < tol &&
               std::abs(y_ - other.y_) < tol && std::abs(z_ - other.z_) < tol;
    }
    [[nodiscard]] bool operator==(const Quaternion& other) const {
        return almost_equal(other);
    }
    [[nodiscard]] bool operator!=(const Quaternion& other) const {
        return !(*this == other);
    }

    // One first-order integration increment: dq = 0.5 * q * ω * dt.
    // (Renamed from `derivative`, which hid the dt scaling — this returns
    // a step, not a derivative.)
    [[nodiscard]] Quaternion integrationStep(const Vec3T& angularVel, T dt) const {
        Quaternion omega(T{0}, angularVel[0], angularVel[1], angularVel[2]);
        return *this * omega * (T{0.5} * dt);
    }

    // Update quaternion using first-order integration
    Quaternion integrate(const Vec3T& angularVel, T dt) {
        *this = *this + integrationStep(angularVel, dt);
        *this = normalize();
        return *this;
    }

    // Update quaternion using second-order (midpoint) integration
    Quaternion integrateRK2(const Vec3T& angularVel, T dt) {
        Quaternion dq1 = integrationStep(angularVel, dt * T{0.5});
        Quaternion q_half = (*this + dq1).normalize();

        Quaternion dq2 = q_half.integrationStep(angularVel, dt * T{0.5});
        *this = *this + dq2 * T{0.5};
        *this = normalize();

        return *this;
    }

    // Convert to a 3x3 rotation matrix (q is normalized internally if needed)
    [[nodiscard]] Matrix<T, 3, 3, B> toRotationMatrix() const {
        Quaternion n = normalize();
        T w = n.w_, x = n.x_, y = n.y_, z = n.z_;
        Matrix<T, 3, 3, B> r;
        r(0, 0) = T{1} - T{2} * (y * y + z * z);
        r(0, 1) = T{2} * (x * y - w * z);
        r(0, 2) = T{2} * (x * z + w * y);
        r(1, 0) = T{2} * (x * y + w * z);
        r(1, 1) = T{1} - T{2} * (x * x + z * z);
        r(1, 2) = T{2} * (y * z - w * x);
        r(2, 0) = T{2} * (x * z - w * y);
        r(2, 1) = T{2} * (y * z + w * x);
        r(2, 2) = T{1} - T{2} * (x * x + y * y);
        return r;
    }

    // Rotate a vector by this quaternion: v' = q * v * q^(-1)
    [[nodiscard]] Vec3T rotate(const Vec3T& v) const {
        Quaternion v_quat(T{0}, v[0], v[1], v[2]);
        Quaternion qv = *this * v_quat;
        Quaternion r = qv * conjugate();

        return Vec3T({r.x_, r.y_, r.z_});
    }

    // Get conjugate (q_conj = [w, -x, -y, -z])
    [[nodiscard]] Quaternion conjugate() const noexcept {
        return Quaternion(w_, -x_, -y_, -z_);
    }

    // Normalize quaternion (zero quaternion returned unchanged)
    [[nodiscard]] Quaternion normalize() const {
        T len = std::sqrt(w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_);
        if (len > T{0}) {
            T inv_len = T{1} / len;
            return Quaternion(w_ * inv_len, x_ * inv_len, y_ * inv_len, z_ * inv_len);
        }
        return *this;
    }

    // Get quaternion as Vec4
    [[nodiscard]] Vec4T asVec4() const {
        return Vec4T({w_, x_, y_, z_});
    }

    // Get scalar part (w) and vector parts
    [[nodiscard]] T w() const noexcept { return w_; }
    [[nodiscard]] T x() const noexcept { return x_; }
    [[nodiscard]] T y() const noexcept { return y_; }
    [[nodiscard]] T z() const noexcept { return z_; }

    // Get vector part (x, y, z)
    [[nodiscard]] Vec3T vec() const {
        return Vec3T({x_, y_, z_});
    }

    // Get magnitude
    [[nodiscard]] T magnitude() const {
        return std::sqrt(w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_);
    }

private:
    T w_, x_, y_, z_;
};

// Quaternion identity
template<typename T, Backend B>
[[nodiscard]] Quaternion<T, B> quaternionIdentity() {
    return Quaternion<T, B>(T{1}, T{0}, T{0}, T{0});
}

// Quaternion zero
template<typename T, Backend B>
[[nodiscard]] Quaternion<T, B> quaternionZero() {
    return Quaternion<T, B>(T{0}, T{0}, T{0}, T{0});
}

} // namespace hlcl
