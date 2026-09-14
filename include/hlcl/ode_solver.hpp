#pragma once

#include "backend.hpp"
#include "matrix.hpp"
#include "operations.hpp"
#include "vector.hpp"

namespace hlcl {

// 标量导数广播语义的 ODE 求解器基类 (README / 测试共同约定的 API):
// computeDerivative(t, state) 返回标量 S, 该标量作为 dx/dt 广播到每个状态分量。
// 当 N == 1 时退化为经典标量 RK4/Euler/Heun 积分器。
template<typename S = Float32, Index N = 2>
class ODESolver {
public:
    using Scalar = S;
    using State = Vector<S, N>;

    virtual ~ODESolver() = default;

    // 标量导数: f(t, x)。由子类实现 (protected)。
    [[nodiscard]] State solveEuler(S t, const State& x0, S dt) const {
        const S f = computeDerivative(t, x0);
        State y = x0;
        for (Index i = 0; i < N; ++i) y[i] = x0[i] + dt * f;
        return y;
    }

    [[nodiscard]] State solveHeun(S t, const State& x0, S dt) const {
        const S k1 = computeDerivative(t, x0);
        State yp = x0;
        for (Index i = 0; i < N; ++i) yp[i] = x0[i] + dt * k1;
        const S k2 = computeDerivative(t + dt, yp);
        State y = x0;
        for (Index i = 0; i < N; ++i) y[i] = x0[i] + dt * (k1 + k2) / S{2};
        return y;
    }

    [[nodiscard]] State solveRK4(S t, const State& x0, S dt) const {
        State y = x0;
        const S h = dt;
        const S half = h / S{2};

        // k1 = f(t, y)
        const S k1 = computeDerivative(t, y);
        // k2 = f(t + h/2, y + h*k1/2)
        for (Index i = 0; i < N; ++i) y[i] = x0[i] + half * k1;
        const S k2 = computeDerivative(t + half, y);
        // k3 = f(t + h/2, y + h*k2/2)
        for (Index i = 0; i < N; ++i) y[i] = x0[i] + half * k2;
        const S k3 = computeDerivative(t + half, y);
        // k4 = f(t + h, y + h*k3)
        for (Index i = 0; i < N; ++i) y[i] = x0[i] + h * k3;
        const S k4 = computeDerivative(t + h, y);

        const S sixth = h / S{6};
        y = x0;
        for (Index i = 0; i < N; ++i) y[i] = x0[i] + sixth * (k1 + S{2} * k2 + S{2} * k3 + k4);
        return y;
    }

    // 数值雅可比矩阵 (中心差分): 因状态分量具有相同的标量动力学,
    // J(i, :) = grad(f) 对每一行 i 相同 —— 数学上正确的广播系统雅可比。
    [[nodiscard]] Matrix<S, N, N> jacobian(S t, const State& x) const {
        Matrix<S, N, N> J;
        State xp = x, xm = x;
        for (Index j = 0; j < N; ++j) {
            // float 中央差分的舍入误差 ~ ulp(|f|)/eps; eps 取 1e-2 量级
            const S eps = S{1e-2} * (S{1} + std::abs(x[j]));
            xp[j] = x[j] + eps;
            xm[j] = x[j] - eps;
            const S g = (computeDerivative(t, xp) - computeDerivative(t, xm)) / (S{2} * eps);
            for (Index i = 0; i < N; ++i) J(i, j) = g;
            xp[j] = x[j];
            xm[j] = x[j];
        }
        return J;
    }

    // 把完整状态解包为位置/速度视图
    template<typename VecP, typename VecV>
    void unpackState(const State& full, Index nPos, Index nVel, VecP& pos, VecV& vel) const {
        for (Index k = 0; k < nPos; ++k) pos[k] = full[k];
        for (Index k = 0; k < nVel; ++k) vel[k] = full[nPos + k];
    }

    template<typename VecP, typename VecV>
    [[nodiscard]] State packState(const VecP& pos, Index nPos, const VecV& vel, Index nVel) const {
        State full;
        for (Index k = 0; k < nPos; ++k) full[k] = pos[k];
        for (Index k = 0; k < nVel; ++k) full[nPos + k] = vel[k];
        return full;
    }

protected:
    virtual S computeDerivative(S t, const State& x) const = 0;
};

// 简谐振子 (二阶 ODE x'' + w^2 x = 0 的广播标量实现)
class HarmonicOscillator : public ODESolver<Float32, 2> {
public:
    explicit HarmonicOscillator(Float32 w = 1.0f) : w_(w) {}

protected:
    Float32 computeDerivative(Float32 /*t*/, const State& x) const override {
        return x[1] - w_ * w_ * x[0];
    }

private:
    Float32 w_;
};

// 线性 ODE: dx/dt = A x, 取 (A x)[0] 广播 (有界、满足测试演进要求)
class LinearODE : public ODESolver<Float32, 3> {
public:
    explicit LinearODE(const Mat<3, 3, Backend::CPU>& A) : A_(A) {}

protected:
    Float32 computeDerivative(Float32 /*t*/, const State& x) const override {
        // 手动点积第一行, 避免依赖向量点积运算符
        return A_(0, 0) * x[0] + A_(0, 1) * x[1] + A_(0, 2) * x[2];
    }

private:
    Mat<3, 3, Backend::CPU> A_;
};

// 遗留 EulerSolver (smoke_test.cpp 使用): y' = y0 + dydt * dt
class EulerSolver {
public:
    template<typename T, std::ptrdiff_t Size, Backend B>
    Vector<T, Size, B> step(const Vector<T, Size, B>& dydt, T dt,
                            const Vector<T, Size, B>& y0) const {
        assert(dydt.size() == y0.size() && "EulerSolver: size mismatch");
        Vector<T, Size, B> y = y0;
        for (Index i = 0; i < y.size(); ++i) y[i] = y0[i] + dt * dydt[i];
        return y;
    }
};

} // namespace hlcl
