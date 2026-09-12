#pragma once

#include "matrix.hpp"
#include "vector.hpp"
#include <cmath>
#include <algorithm>

namespace hlcl {

/// Quaternion representation for 3D rotations
/// Quaternion = [w, x, y, z] where w is the scalar part
template<typename T, Backend B>
class Quaternion {
public:
    using Vec4 = Vec<4, B>;

    Quaternion() : data_(Vec4({T{1}, T{0}, T{0}, T{0}})) {}  // identity

    Quaternion(T w, T x, T y, T z) : data_(Vec4({w, x, y, z})) {}

    // Initialize from Euler angles (intrinsic Z-Y-X: roll=x, pitch=y, yaw=z)
    static Quaternion fromEuler(const Vec3& euler) {
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
    Vec3 toEuler() const {
        T sqw = data_[0] * data_[0];
        T sqx = data_[1] * data_[1];
        T sqy = data_[2] * data_[2];
        T sqz = data_[3] * data_[3];

        T unit = sqw + sqx + sqy + sqz;
        if (unit > T{0}) {
            unit = T{1} / unit;
        }

        Vec3 euler;
        euler[0] = std::atan2(T{2} * (data_[0] * data_[1] + data_[2] * data_[3]) * unit, T{1} - T{2} * (sqx + sqy) * unit);  // roll (x-axis rotation)
        euler[1] = std::asin(std::clamp(T{2} * (data_[0] * data_[2] - data_[1] * data_[3]) * unit, T{-1}, T{1}));  // pitch (y-axis rotation)
        euler[2] = std::atan2(T{2} * (data_[0] * data_[3] + data_[1] * data_[2]) * unit, T{1} - T{2} * (sqy + sqz) * unit);  // yaw (z-axis rotation)

        return euler;
    }

    // Multiply two quaternions: q * p
    Quaternion operator*(const Quaternion& other) const {
        T w1 = data_[0], x1 = data_[1], y1 = data_[2], z1 = data_[3];
        T w2 = other.data_[0], x2 = other.data_[1], y2 = other.data_[2], z2 = other.data_[3];

        T w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        T x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        T y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        T z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

        return Quaternion(w, x, y, z);
    }

    // Multiply by scalar
    Quaternion operator*(T scalar) const {
        return Quaternion(data_[0] * scalar, data_[1] * scalar, data_[2] * scalar, data_[3] * scalar);
    }

    // Multiply by scalar (right)
    friend Quaternion operator*(T scalar, const Quaternion& q) {
        return q * scalar;
    }

    // Quaternion addition
    Quaternion operator+(const Quaternion& other) const {
        return Quaternion(
            data_[0] + other.data_[0],
            data_[1] + other.data_[1],
            data_[2] + other.data_[2],
            data_[3] + other.data_[3]
        );
    }

    // Quaternion subtraction
    Quaternion operator-(const Quaternion& other) const {
        return Quaternion(
            data_[0] - other.data_[0],
            data_[1] - other.data_[1],
            data_[2] - other.data_[2],
            data_[3] - other.data_[3]
        );
    }

    // Negate quaternion
    Quaternion operator-() const {
        return Quaternion(-data_[0], -data_[1], -data_[2], -data_[3]);
    }

    // Quaternion negation
    Quaternion& operator-() {
        data_ = -data_;
        return *this;
    }

    // Quaternion equality (with tolerance)
    bool operator==(const Quaternion& other) const {
        return (data_ - other.data_).norm() < T{1e-6};
    }

    // Quaternion inequality
    bool operator!=(const Quaternion& other) const {
        return !(*this == other);
    }

    // Compute quaternion derivative: dq/dt = 0.5 * q * ω
    // where ω is the angular velocity quaternion
    Quaternion derivative(const Vec3& angularVel, T dt) const {
        // Angular velocity quaternion: ω = [0, wx, wy, wz]
        Quaternion omega(T{0}, angularVel[0], angularVel[1], angularVel[2]);

        // dq/dt = 0.5 * q * ω
        Quaternion dq = *this * omega * T{0.5};

        return dq * dt;
    }

    // Update quaternion using first-order integration
    Quaternion integrate(const Vec3& angularVel, T dt) {
        Quaternion dq = derivative(angularVel, dt);
        *this = *this + dq;
        *this = normalize();
        return *this;
    }

    // Update quaternion using second-order integration (more accurate)
    Quaternion integrateRK2(const Vec3& angularVel, T dt) {
        Quaternion dq1 = derivative(angularVel, dt * T{0.5});
        Quaternion q_half = *this + dq1;
        q_half = q_half.normalize();

        Quaternion dq2 = q_half.derivative(angularVel, dt * T{0.5});
        *this = *this + dq2 * T{0.5};
        *this = normalize();

        return *this;
    }
    // Rotate a vector by this quaternion

    // Convert to a 3x3 rotation matrix (q is normalized internally if needed)
    Matrix<T, 3, 3, B> toRotationMatrix() const {
        Quaternion n = normalize();
        T w = n.data_[0], x = n.data_[1], y = n.data_[2], z = n.data_[3];
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

    Vec3 rotate(const Vec3& v) const {
        // v' = q * v * q^(-1)
        // where q^(-1) = conjugate(q) / |q|^2
        Quaternion v_quat(T{0}, v[0], v[1], v[2]);
        Quaternion q_conj = conjugate();

        Quaternion qv = *this * v_quat;
        Quaternion r = qv * q_conj;

        return Vec3({r.data_[1], r.data_[2], r.data_[3]});
    }

    // Get conjugate (q_conj = [w, -x, -y, -z])
    Quaternion conjugate() const {
        return Quaternion(data_[0], -data_[1], -data_[2], -data_[3]);
    }

    // Normalize quaternion
    Quaternion normalize() const {
        T len = std::sqrt(data_[0] * data_[0] + data_[1] * data_[1] +
                          data_[2] * data_[2] + data_[3] * data_[3]);
        if (len > T{0}) {
            T inv_len = T{1} / len;
            return Quaternion(
                data_[0] * inv_len,
                data_[1] * inv_len,
                data_[2] * inv_len,
                data_[3] * inv_len
            );
        }
        return *this;
    }

    // Get quaternion as Vec4
    Vec4 asVec4() const {
        return data_;
    }

    // Get scalar part (w)
    T w() const { return data_[0]; }
    T x() const { return data_[1]; }
    T y() const { return data_[2]; }
    T z() const { return data_[3]; }

    // Get vector part (x, y, z)
    Vec3 vec() const {
        return Vec3(data_[1], data_[2], data_[3]);
    }

    // Get magnitude
    T magnitude() const {
        return data_.norm();
    }

private:
    Vec4 data_;
};

// Quaternion identity
template<typename T, Backend B>
Quaternion<T, B> quaternionIdentity() {
    return Quaternion<T, B>(T{1}, T{0}, T{0}, T{0});
}

// Quaternion zero
template<typename T, Backend B>
Quaternion<T, B> quaternionZero() {
    return Quaternion<T, B>(T{0}, T{0}, T{0}, T{0});
}

} // namespace hlcl
