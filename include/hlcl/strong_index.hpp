#pragma once

// 强类型行列索引: Matrix 的行/列在类型层面隔离, 互换即编译错误。
//
//   Matrix<float, 3, 4> m;
//   m(row_index{1}, col_index{2});   // OK
//   m(col_index{1}, row_index{2});   // 编译错误: 行列互换
//   m(1, 2);                         // 编译错误: int 不隐式转换
//   m(1_r, 2_c);                     // 字面量语法 (推荐)
//
// 设计要点:
//   * 构造 explicit —— int 不得隐式退化, 循环内显式包裹:
//       for (Index r = 0; r < m.rows(); ++r) m(row_index{r}, col_index{c});
//   * 底层统一 Index (std::ptrdiff_t, 与 Eigen::Index 同义);
//     单一构造函数, 避免重载歧义。
//   * 两类型互不可构造/不可转换 —— 行列互换在编译期被拒绝。

#include "backend.hpp"

#include <cassert>
#include <cstddef>

namespace hlcl {

struct row_index {
    Index value;
    explicit constexpr row_index(Index v) noexcept : value(v) { assert(v >= 0 && "row_index 负值"); }
};

struct col_index {
    Index value;
    explicit constexpr col_index(Index v) noexcept : value(v) { assert(v >= 0 && "col_index 负值"); }
};

static_assert(!std::is_constructible_v<col_index, row_index>,
              "col_index 不得由 row_index 构造 (行列互换必须编译失败)");
static_assert(!std::is_constructible_v<row_index, col_index>,
              "row_index 不得由 col_index 构造");
static_assert(!std::is_convertible_v<int, row_index> &&
                  !std::is_convertible_v<int, col_index>,
              "int 不得隐式转换为索引类型");

constexpr row_index operator""_r(unsigned long long v) noexcept {
    return row_index{static_cast<Index>(v)};
}

constexpr col_index operator""_c(unsigned long long v) noexcept {
    return col_index{static_cast<Index>(v)};
}

} // namespace hlcl
