-- hlcl — 仿 Eigen 风格 header-only 代数库 (本目录自包含)
-- 向量/矩阵/四元数/ODE 求解器与 SYCL GPU / Vulkan compute 内核层 (编译期开关
-- HLCL_GPU_ENABLED / HLCL_KOMPUTE_ENABLED), 纯 C++17, 零第三方依赖。
--
-- 消费方式:
--   * 本目录内构建/测试: xmake && xmake run run_hlcl_tests
--   * 其他 xmake 项目: add_requires("hlcl") (内嵌包定义见本文件)
-- 本包定义同时在本文件内给出 (内嵌包), `xmake require -y hlcl` 即可验证安装。
local dir = os.scriptdir()

-- C++20 (concepts/span 等): 根项目同步声明; 此处保证独立构建时也成立。
set_languages("c++20")
set_warnings("all", "extra", "pedantic")

-- ============ 独立构建判定 ============
-- hlcl 可独立构建/测试 (cd hlcl && xmake ...): 此时选项与后端依赖由本文件
-- 自持; 被 avbd 根 xmake.lua includes 时, 这些由根提供, 本文件只按 has_config 取值。
local standalone = (path.normalize(os.projectdir()) == path.normalize(dir))

-- GPU 后端取值: 选项仅在独立构建时定义 (被 avbd 根 includes 时缺省 omp);
-- has_config 在部分 xmake 版本对字符串选项返回布尔, 统一经 get_config 归一
function _gpu_backend()
    local v = get_config("gpu_backend") or has_config("gpu_backend")
    if type(v) ~= "string" then v = "omp" end
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
        set_values("omp", "cuda")
        set_description("AdaptiveCpp kernel backend for --gpu=y",
                        "  omp  — OpenMP CPU host (default, no GPU needed)",
                        "  cuda — NVIDIA CUDA direct PTX (needs driver + CUDA toolkit)")
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

    if has_config("kompute") then
        add_requires("kompute v0.8.0")
        add_requires("glslang", {configs = {binaryonly = true}})
    end
    if has_config("gpu") then
        -- 官方 xmake-repo 包 (本人提交维护), 不依赖任何私有仓库
        add_requires("adaptive_cpp", {configs = {cuda = (_gpu_backend() == "cuda")}})
    end
    -- 取值硬校验 (set_values 在此 xmake 版本不强制拒绝)
    local gpu_backend = _gpu_backend()
    if gpu_backend ~= "omp" and gpu_backend ~= "cuda" then
        raise("--gpu_backend=%s 无效 (可选: omp | cuda)", gpu_backend)
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

    -- 内嵌本地包: 直接从本目录复制头文件 (无需远端 URL)
    on_install(function (package)
        os.cp(path.join(dir, "include"), package:installdir())
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

-- ============ AdaptiveCpp 工具链 (一等声明) ============
-- acpp 是 clang 包装驱动: 以独立 toolchain 声明, on_check 宽松放行 (首次
-- configure 时包可能尚未装好), on_load 定位二进制并注入 toolset 与后端
-- flags; 缺失时在构建期给出可操作报错, 不静默回退到默认编译器。
toolchain("acpp")
    set_kind("standalone")
    set_description("AdaptiveCpp (acpp) SYCL toolchain")
    on_check(function (toolchain)
        -- 与包绑定: 包里带了 adaptive_cpp 依赖, acpp 一定在包内
        for _, package in ipairs(toolchain:packages()) do
            local envs = package:envs()
            if envs and envs.PATH then
                local acpp = path.join(envs.PATH[1], "acpp")
                if os.isfile(acpp) then
                    toolchain:config_set("acpp", acpp)
                    return true
                end
            end
        end
        -- 允许系统安装的 acpp (发行版包), 不再读任何环境变量
        import("lib.detect.find_tool")
        local t = find_tool("acpp", {version = true})
        if t and t.program then
            toolchain:config_set("acpp", t.program)
            return true
        end
        return false
    end)
    on_load(function (toolchain)
        import("core.project.config")

        local acpp = toolchain:config("acpp")
        if not (acpp and os.isfile(acpp)) then
            raise("未找到 acpp 驱动: 确认 adaptive_cpp 包已安装 (xmake -y) 或系统装有 AdaptiveCpp")
        end

        -- xmake 按可执行文件名加载对应工具脚本 (编译器语义);
        -- acpp 是 clang 包装驱动, 故以 clang 命名的链接呈现给它。
        -- 在此统一建一处链接 (构建目录), 取代旧 per-target on_load hack。
        local link = path.join(os.projectdir(), "build", "acpp-clang")
        os.mkdir(path.directory(link))
        os.tryrm(link)
        os.ln(acpp, link)

        toolchain:set("toolset", "cc",  link)
        toolchain:set("toolset", "cxx", link)
        toolchain:set("toolset", "ld",  link)
        toolchain:set("toolset", "sh",  link)

        -- 后端 flags: 编译与链接都要传, 链接阶段 acpp 依据它附加对应运行时。
        -- 注意: toolchain 回调在独立沙箱执行, 无法访问脚本全局, 故内联取值。
        -- 仅保留直译后端: omp (主机直译) 与 cuda (PTX 直译, 需 CUDA toolkit)。
        local backend = config.get("gpu_backend")
        if type(backend) ~= "string" then backend = "omp" end
        local arch = config.get("cuda_arch")
        if type(arch) ~= "string" then arch = "sm_75" end
        local targets
        if backend == "cuda" then
            import("detect.sdks.find_cuda")
            local sdk = find_cuda()
            if not (sdk and sdk.sdkdir) then
                raise("gpu_backend=cuda 需要 CUDA toolkit: 未检测到 SDK, 请安装 CUDA 或 xmake f --cuda=<SDK目录>")
            end
            targets = "--acpp-targets=cuda:" .. arch .. " --acpp-cuda-path=" .. sdk.sdkdir
        else
            targets = "--acpp-targets=omp"
        end
        toolchain:add("cxflags", targets)
        toolchain:add("ldflags", targets)
    end)
toolchain_end()

-- ============ GPU 测试 (仅 --gpu=y) ============
-- AdaptiveCpp (acpp) 直译后端经 --gpu_backend 选择 (omp | cuda);
-- 工具链与后端 flags 全部由 toolchain("acpp") 声明, target 保持纯声明式。
if has_config("gpu") then
    local function _gpu_test(name)
        target(name)
            set_kind("binary")
            add_files(path.join(tests_dir, name .. ".cpp"))
            add_deps("hlcl")
            add_packages("adaptive_cpp")
            add_defines("HLCL_GPU_ENABLED")
            -- acpp@包名: 把 adaptive_cpp 包实例绑定进工具链 (驱动即包内 bin/acpp)
            set_toolchains("acpp@adaptive_cpp")
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
