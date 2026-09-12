#pragma once

namespace hlcl {

// 后端抽象接口
//   CPU     —— 宿主机数学 (主模板)
//   GPU     —— AdaptiveCpp (SYCL) 后端; 默认构建下同为宿主机数学
//   Kompute —— Kompute (Vulkan compute) 后端; float 内核真正派发 GPU 设备
enum class Backend {
    CPU,
    GPU,
    Kompute
};

// 浮点标量类型别名
using Float32 = float;
using Float64 = double;

// 版本常量 (test_ode.cpp 断言 1.1.0)
constexpr int HLCL_VERSION_MAJOR = 1;
constexpr int HLCL_VERSION_MINOR = 1;
constexpr int HLCL_VERSION_PATCH = 0;

} // namespace hlcl
