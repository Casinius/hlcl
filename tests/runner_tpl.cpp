// 通用测试运行器: 按编译期 TARGETS 列表逐一执行同一 targetdir 内的兄弟二进制。
// 各别名目标 (smoke/unit/fuzz/benchmark/run_all_tests/...) 以不同 TARGETS 编译本文件。
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define STR_IMPL(...) #__VA_ARGS__
#define STR(...) STR_IMPL(__VA_ARGS__)

#ifndef TARGETS
#define TARGETS ""
#endif

int main(int argc, char** argv) {
    (void)argc;
    std::string self = argv[0];
    std::string dir = self.substr(0, self.find_last_of('/'));
    if (dir.empty()) dir = ".";

    std::string list = STR(TARGETS);
    std::vector<std::string> names;
    std::string::size_type pos = 0;
    while ((pos = list.find(',')) != std::string::npos) {
        names.push_back(list.substr(0, pos));
        list.erase(0, pos + 1);
    }
    if (!list.empty()) names.push_back(list);

    int passed = 0;
    int failed = 0;
    for (const std::string& name : names) {
        std::printf("==== running %s ====\n", name.c_str());
        std::fflush(stdout);
        int rc = std::system((dir + "/" + name).c_str());
        if (rc == 0) {
            ++passed;
            std::printf("[PASS] %s\n", name.c_str());
        } else {
            ++failed;
            std::printf("[FAIL] %s\n", name.c_str());
        }
        std::fflush(stdout);
    }
    std::printf("%d/%zu subtests passed\n", passed, names.size());
    return failed == 0 ? 0 : 1;
}
