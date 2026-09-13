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

-- C++20 (concepts/span 等): 根项目同步声明; 此处保证独立构建时也成立。
set_languages("c++20")
set_warnings("all", "extra", "pedantic")

-- ============ 独立构建判定 ============
-- hlcl 可独立构建/测试 (cd hlcl && xmake ...): 此时选项、mypack 包仓库与
-- 后端依赖由本文件自持; 被 avbd 根 xmake.lua includes 时, 这些由根提供,
-- 本文件只按 has_config 取值。
local standalone = (path.normalize(os.projectdir()) == path.normalize(dir))

-- GPU 后端取值: 选项仅在独立构建时定义 (被 avbd 根 includes 时缺省 omp)
function _gpu_backend()
    local v = get_config("gpu_backend") or has_config("gpu_backend")
    if type(v) ~= "string" then v = "omp" end
    return v
end

function _cuda_arch()
    local v = get_config("cuda_arch") or has_config("cuda_arch")
    if type(v) ~= "string" then v = "sm_75" end
    return v
end

if standalone then
    option("gpu")
        set_default(false)
        set_showmenu(true)
        set_description("Enable SYCL GPU tests via AdaptiveCpp (acpp)")
    option_end()
    option("gpu_backend")
        set_default("omp")
        set_showmenu(true)
        set_values("omp", "opencl", "cuda")
        set_description("AdaptiveCpp kernel backend for --gpu=y",
                        "  omp    — OpenMP CPU host (default, no GPU needed)",
                        "  opencl — OpenCL devices",
                        "  cuda   — NVIDIA CUDA (needs driver + CUDA toolkit)")
    option_end()
    option("cuda_arch")
        set_default("sm_75")
        set_showmenu(true)
        set_description("SM arch for --gpu_backend=cuda",
                        "  sm_75 GTX16xx, sm_80/86 A100/RTX30xx, sm_89 RTX40xx")
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

    -- mypack 仓库: 自维护的 adaptive_cpp 包 (官方 xmake-repo 尚未收录)。
    -- 解析顺序: HLCL_MYPACK 环境变量 > avbd 根布局 (../..) > 本机已知位置;
    -- 旧写法 "../.." 在项目目录迁移后失效 (相对路径按 cwd 解析)。
    local mypack
    local candidates = {}
    if os.getenv("HLCL_MYPACK") then
        table.insert(candidates, os.getenv("HLCL_MYPACK"))
    end
    table.insert(candidates, path.join(dir, "..", ".."))
    table.insert(candidates, "/home/cyan/Documents/xmake_mypack/adaptive_cpp")
    table.insert(candidates, "/home/cyan/xmake-repo")
    for _, p in ipairs(candidates) do
        if p and os.isfile(path.join(p, "packages", "a", "adaptive_cpp", "xmake.lua")) then
            mypack = p
            break
        end
    end
    if mypack then
        add_repositories("mypack " .. mypack)
    elseif has_config("gpu") then
        wprint("未找到 adaptive_cpp 包仓库 (mypack): 设 HLCL_MYPACK=<仓库路径> 后重试; GPU 测试将不可构建")
    end
    if has_config("kompute") then
        add_requires("kompute v0.8.0")
        add_requires("glslang", {configs = {binaryonly = true}})
    end
    if not has_config("gpu") and _gpu_backend() ~= "omp" then
        raise("--gpu_backend=%s 需要同时指定 --gpu=y", _gpu_backend())
    end
    if has_config("gpu") then
        add_requires("adaptive_cpp", {configs = {
            opencl = (_gpu_backend() == "opencl"),
            cuda   = (_gpu_backend() == "cuda")}})
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
            configs = {languages = "c++20"},
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
        -- 裸 assert 的测试文件在文件头 #undef NDEBUG 保证断言存活 (见 tests/)
end

_hlcl_test("test_vector")
_hlcl_test("test_matrix")
_hlcl_test("test_ode")
_hlcl_test("test_quaternion")
_hlcl_test("fuzz_test")
_hlcl_test("test_special_values")
_hlcl_test("test_error_handling")
_hlcl_test("test_concurrent")

-- ============ GPU 测试 (仅 --gpu=y) ============
-- AdaptiveCpp (acpp) 驱动编译/链接, 内核后端经 --gpu_backend 选择:
--   omp    — OpenMP CPU host (默认; 无 GPU 也能跑)
--   opencl — OpenCL 设备
--   cuda   — NVIDIA CUDA (构建机需 NVIDIA 驱动 + CUDA toolkit)
if has_config("gpu") then
    local acpp_targets = ({
        ["omp"]    = "--acpp-targets=omp",
        ["opencl"] = "--acpp-targets=opencl",
        ["cuda"]   = "--acpp-targets=cuda:" .. _cuda_arch(),
    })[_gpu_backend()]
    if not acpp_targets then
        raise("未知 --gpu_backend=%s (可选: omp | opencl | cuda)", _gpu_backend())
    end

    -- acpp 是 clang 包装驱动: 以名为 acpp-clang 的符号链接呈现, xmake 即按
    -- clang 语义驱动编译/链接 (文件名含 clang 是识别依据)。各目标在自己
    -- autogen 目录建链, 规避并行构建竞争。
    local function _use_acpp_driver(target)
        local pkg = target:pkg("adaptive_cpp")
        local acpp = pkg and pkg:installdir() and path.join(pkg:installdir(), "bin", "acpp")
        if not (acpp and os.isfile(acpp)) then
            raise("adaptive_cpp 包未就绪: 先完成包安装 (xmake -y), 再构建 %s", target:name())
        end
        local link = path.join(target:autogendir(), "acpp-clang")
        os.mkdir(path.directory(link))
        os.tryrm(link)
        os.ln(acpp, link)
        target:set("toolset", "cxx", link)
        target:set("toolset", "ld",  link)
    end

    local function _gpu_test(name)
        target(name)
            set_kind("binary")
            add_files(path.join(tests_dir, name .. ".cpp"))
            add_deps("hlcl")
            add_packages("adaptive_cpp")
            add_defines("HLCL_GPU_ENABLED")
            -- 编译与链接都要传: 链接阶段 acpp 依据它附加对应运行时
            add_cxflags(acpp_targets, {force = true})
            add_ldflags(acpp_targets, {force = true})
            on_load(_use_acpp_driver)
    end

    _gpu_test("test_gpu_matrix_multiply")
    _gpu_test("test_gpu_core")
    _gpu_test("test_host_device_copy")
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
_add_runner("run_hlcl_edge",  { "test_special_values", "test_error_handling", "test_concurrent", "test_quaternion" })
if has_config("gpu") then
    _add_runner("run_hlcl_gpu", { "test_gpu_core", "test_gpu_matrix_multiply", "test_host_device_copy" })
end
if has_config("kompute") then
    _add_runner("run_hlcl_kompute", { "test_kompute_core", "test_kompute_matrix_multiply" })
end
