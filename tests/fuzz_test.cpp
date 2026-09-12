#include "test_framework.hpp"
#include "test_fuzz_suite.hpp"
#include "hlcl/core.hpp"

using namespace hlcl;
using namespace hlcl::test;

int main() {
    TestRunner runner;

    runner.add_suite(std::make_unique<FuzzTestSuite>());

    int code = runner.run_all();

    return code;
}
