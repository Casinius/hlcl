-- hlcl — 仿 Eigen 风格 header-only 代数库 (独立子项目, 以内嵌包形式分发)
-- 向量/矩阵/四元数/ODE 求解器与 SYCL GPU / Vulkan compute 内核层 (编译期开关
-- HLCL_GPU_ENABLED / HLCL_KOMPUTE_ENABLED, 见 tests/xmake.lua), 纯 C++17,
-- 零第三方依赖。
--
-- 三种消费方式:
--   * 本仓库内 (avbd 引擎/测试): includes() 引入本文件, target 依赖 add_deps("hlcl")
--   * 其他 xmake 项目: 复制本文件中的 package("hlcl") 定义, 或经包仓库分发后
--     add_requires("hlcl")
-- 本包定义同时在本文件内给出 (内嵌包), `xmake require -y hlcl` 即可验证安装。
local dir = os.scriptdir()

-- ============ 独立构建判定 ============
-- hlcl 可独立构建/测试 (cd hlcl && xmake ...): 此时选项、mypack 包仓库与
-- 后端依赖由本文件自持; 被 avbd 根 xmake.lua includes 时, 这些由根提供,
-- 本文件只按 has_config 取值。
local standalone = (path.normalize(os.projectdir()) == path.normalize(dir))

if standalone then
    option("gpu")
        set_default(false)
        set_showmenu(true)
        set_description("Enable SYCL GPU tests via AdaptiveCpp (acpp)")
    option_end()
    option("opencl")
        set_default(false)
        set_showmenu(true)
        set_description("Use AdaptiveCpp OpenCL backend instead of OpenMP (requires --gpu=y)")
    option_end()
    option("kompute")
        set_default(false)
        set_showmenu(true)
        set_description("Enable Kompute (Vulkan compute) backend tests")
    option_end()
    option("coverage")
        set_default(false)
        set_showmenu(true)
        set_description("Instrument all targets with gcov (--coverage)")
    option_end()

    if has_config("coverage") then
        add_cxflags("-fprofile-arcs", "-ftest-coverage", {force = true})
        add_ldflags("-fprofile-arcs", {force = true})
    end

    -- mypack 仓库在 hlcl 的上两级 (adaptive_cpp/)
    add_repositories("mypack ../..")
    if has_config("kompute") then
        add_requires("kompute v0.8.0")
        add_requires("glslang", {configs = {binaryonly = true}})
    end
    if has_config("gpu") then
        add_requires("adaptive_cpp", {configs = {opencl = has_config("opencl")}})
    end
end

-- ============ 内嵌包定义 (子项目包) ============
package("hlcl")
    set_kind("library", {headeronly = true})
    set_homepage("https://example.invalid/hlcl")
    set_description("hlcl — Eigen-style header-only linear algebra library (Vec/Mat/Quaternion/ODE, optional SYCL & Vulkan compute backends)")
    set_license("MIT")

    add_configs("gpu",     {description = "Install with HLCL_GPU_ENABLED (SYCL backend surface).", default = false, type = "boolean"})
    add_configs("kompute", {description = "Install with HLCL_KOMPUTE_ENABLED (Vulkan compute backend surface).", default = false, type = "boolean"})

    on_load(function (package)
        package:add("includedirs", "include")
        if package:config("gpu") then
            package:add("defines", "HLCL_GPU_ENABLED")
        end
        if package:config("kompute") then
            package:add("defines", "HLCL_KOMPUTE_ENABLED")
            package:add("deps", "kompute v0.8.0")
        end
    end)

    -- 内嵌本地包: 直接从本子项目复制头文件 (无需远端 URL)
    on_install(function (package)
        os.cp(path.join(os.projectdir(), "hlcl", "include"), package:installdir())
    end)

    on_test(function (package)
        assert(package:has_cxxtypes("hlcl::Vec<3>", {
            includes = "hlcl/vector.hpp",
            configs = {languages = "c++17"},
            defines = package:config("gpu") and "HLCL_GPU_ENABLED" or nil,
        }))
    end)
package_end()

-- ============ 库 target (本仓库内开发/构建用) ============
target("hlcl")
    set_kind("headeronly")
    add_headerfiles(path.join(dir, "include/hlcl/**.hpp"))
    add_includedirs(path.join(dir, "include"), {interface = true})

-- ============ Kompute 内核 SPIR-V 生成 ============
-- xmake run kompute_shaders: 将 shaders/*.comp 编译为 SPIR-V 并生成
-- include/hlcl/kompute_shaders.hpp (uint32_t 词数组)。生成产物入库,
-- 普通构建不需要 glslang; 仅内核 GLSL 变更后手动重跑。
if has_config("kompute") then
    target("kompute_shaders")
        set_kind("phony")
        add_packages("glslang")
        on_run(function (target)
            local pkg = target:pkg("glslang")
            -- binaryonly 实例直接落在 installdir 根下 (bin/ 仅软链占位)
            local root = pkg and pkg:installdir() or ""
            local cand = {path.join(root, "glslang"), path.join(root, "glslangValidator"),
                path.join(root, "bin", "glslang"), path.join(root, "bin", "glslangValidator")}
            local glslang = "glslang"
            for _, c in ipairs(cand) do
                if os.isfile(c) then glslang = c break end
            end

            local shaderdir = path.join(os.scriptdir(), "shaders")
            local outhpp = path.join(os.scriptdir(), "include/hlcl/kompute_shaders.hpp")
            local tmpdir = os.tmpdir() .. "/hlcl_kompute_spv"
            os.mkdir(tmpdir)

            local function compile(src, spv)
                os.runv(glslang, {"-V", "--target-env", "vulkan1.1", "-o", spv, src})
            end

            local function spv_to_words(spv)
                -- SPIR-V 小端: b0|b1<<8|b2<<16|b3<<24; xmake 的 bit 库是
                -- 32 位截断实现, 位或会丢高位, 故用算术组合
                local data = io.readfile(spv, {encoding = "binary"})
                local words = {}
                for i = 1, #data, 4 do
                    local b1, b2, b3, b4 = data:byte(i, i + 3)
                    words[#words + 1] = b1 + (b2 or 0) * 256 + (b3 or 0) * 65536
                        + (b4 or 0) * 16777216
                end
                return words
            end

            local out = {
                "// 由 `xmake run kompute_shaders` 生成 — 请勿手改; 源在 shaders/*.comp",
                "// SPIR-V (Vulkan 1.1 目标), 以 uint32_t 词数组内联, 供 Kompute 后端使用。",
                "#pragma once",
                "#include <cstdint>",
                ""}
            local order = {"matmul", "inverse", "determinant", "vec_add",
                "vec_sub", "vec_scale", "vec_norm", "vec_divide"}
            for _, name in ipairs(order) do
                local spv = path.join(tmpdir, name .. ".spv")
                compile(path.join(shaderdir, name .. ".comp"), spv)
                local words = spv_to_words(spv)
                out[#out + 1] = ("inline constexpr uint32_t KP_SPV_%s[] = {"
                    ):format(name:upper())
                for i = 1, #words, 8 do
                    local line = {}
                    for j = i, math.min(i + 7, #words) do
                        line[#line + 1] = ("0x%08x"):format(words[j])
                    end
                    out[#out + 1] = "    " .. table.concat(line, ", ") .. ","
                end
                out[#out + 1] = "};"
                out[#out + 1] = ""
                print(format("  %-12s %d words", name, #words))
            end
            io.writefile(outhpp, table.concat(out, "\n") .. "\n")
            print("==> 生成 " .. outhpp)
        end)
end

-- ============ hlcl 自有测试 (低级测试, 库资产) ============
-- 独立构建入口: cd hlcl && xmake && xmake run run_hlcl_tests
-- 被 avbd 根 includes 时同样定义, 供 run_all_tests / run_coverage_tests 聚合。
local tests_dir = path.join(dir, "tests")

function _hlcl_test(name)
    target(name)
        set_kind("binary")
        add_files(path.join(tests_dir, name .. ".cpp"))
        add_deps("hlcl")
end

_hlcl_test("test_vector")
_hlcl_test("test_matrix")
_hlcl_test("test_ode")
_hlcl_test("fuzz_test")
_hlcl_test("test_special_values")
_hlcl_test("test_error_handling")
_hlcl_test("test_concurrent")

-- ============ GPU 测试 (仅 --gpu=y) ============
if has_config("gpu") then
    local acpp_targets = has_config("opencl") and "--acpp-targets=opencl" or "--acpp-targets=omp"

    function _gpu_core_test(name)
        target(name)
            set_kind("binary")
            add_files(path.join(tests_dir, name .. ".cpp"))
            add_deps("hlcl")
            add_packages("adaptive_cpp")
            add_defines("HLCL_GPU_ENABLED")
            -- 编译与链接都要传: 链接阶段 acpp 依据它附加对应运行时
            add_cxflags(acpp_targets, {force = true})
            add_ldflags(acpp_targets, {force = true})
            on_load(function (target)
                -- 经包内 acpp 驱动编译/链接; 链接命名 acpp-clang:
                -- 让 xmake 按 clang 语义驱动该程序
                local pkg = target:pkg("adaptive_cpp")
                if not (pkg and pkg:installdir() and os.isfile(path.join(pkg:installdir(), "bin", "acpp"))) then
                    wprint("adaptive_cpp 包尚未就绪 (首次配置会自动安装), %s 将在包就绪后以 acpp 驱动重新构建",
                        target:name())
                    return
                end
                local acpp = path.join(pkg:installdir(), "bin", "acpp")
                local link = path.join(target:autogendir(), "acpp-clang")
                os.mkdir(path.directory(link))
                os.tryrm(link)
                os.ln(acpp, link)
                target:set("toolset", "cxx", link)
                target:set("toolset", "ld",  link)
            end)
    end

    _gpu_core_test("test_gpu_matrix_multiply")
    _gpu_core_test("test_gpu_core")
    _gpu_core_test("test_host_device_copy")
end

-- ============ Kompute (Vulkan compute) 测试 (仅 --kompute=y) ============
if has_config("kompute") then
    function _kompute_test(name)
        target(name)
            set_kind("binary")
            add_files(path.join(tests_dir, name .. ".cpp"))
            add_deps("hlcl")
            add_packages("kompute")
            add_defines("HLCL_KOMPUTE_ENABLED")
    end

    _kompute_test("test_kompute_core")
    _kompute_test("test_kompute_matrix_multiply")
end

-- ============ 聚合运行器 ============
function _add_runner(runner_name, list)
    target(runner_name)
        set_kind("binary")
        add_files(path.join(tests_dir, "runner_tpl.cpp"))
        add_defines("TARGETS=" .. table.concat(list, ","))
        add_deps(list)
end

_add_runner("run_hlcl_tests", { "test_vector", "test_matrix", "test_ode", "fuzz_test" })
_add_runner("run_hlcl_edge",  { "test_special_values", "test_error_handling", "test_concurrent" })
if has_config("gpu") then
    _add_runner("run_hlcl_gpu", { "test_gpu_core", "test_gpu_matrix_multiply", "test_host_device_copy" })
end
if has_config("kompute") then
    _add_runner("run_hlcl_kompute", { "test_kompute_core", "test_kompute_matrix_multiply" })
end
