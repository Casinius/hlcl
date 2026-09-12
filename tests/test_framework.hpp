#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace hlcl {
namespace test {

struct TestResult {
    std::string test_name;
    bool passed = true;
    std::vector<std::string> failures;
};

class TestSuite {
public:
    TestSuite() : TestSuite("Unnamed Test Suite") {}
    explicit TestSuite(const std::string& name) : name_(name) {}
    virtual ~TestSuite() = default;
    virtual std::string getName() const { return name_; }
    virtual void run() = 0;

    void assert_true(bool condition, const std::string& message = "condition failed") {
        if (!condition) {
            current_result_.failures.push_back(message);
            current_result_.passed = false;
            std::cerr << "[FAIL] " << current_result_.test_name << ": " << message << std::endl;
        }
    }

    // 数值容差断言: 失败条件 |a - b| > tol
    template<typename T>
    void assert_equal(T a, T b, T tol, const std::string& message = "values differ") {
        const T diff = (a > b) ? (a - b) : (b - a);
        if (diff > tol) {
            current_result_.failures.push_back(message);
            current_result_.passed = false;
            std::cerr << "[FAIL] " << current_result_.test_name << ": " << message
                      << " (|a-b| = " << diff << " > tol " << tol << ")" << std::endl;
        }
    }

    TestResult& getCurrentResult() { return current_result_; }

protected:
    std::string name_;
    TestResult current_result_;
};

class TestRunner {
public:
    TestRunner() = default;
    ~TestRunner() = default;

    void add_suite(TestSuite* suite) {
        suites_.push_back(suite);
    }

    void add_suite(std::unique_ptr<TestSuite> suite) {
        owned_suites_.push_back(std::move(suite));
        suites_.push_back(owned_suites_.back().get());
    }

    int run_all() {
        int passed = 0;
        int failed = 0;

        std::cout << "\n========== Running Test Suites ==========" << std::endl;

        for (auto* suite : suites_) {
            if (suite) {
                suite->run();
                if (suite->getCurrentResult().passed) {
                    ++passed;
                    std::cout << "[PASS] " << suite->getName() << std::endl;
                } else {
                    ++failed;
                    std::cout << "[FAIL] " << suite->getName() << std::endl;
                }
            }
        }

        std::cout << "\n========== Test Results ==========" << std::endl;
        std::cout << "Total Suites: " << suites_.size() << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "========================================\n" << std::endl;

        return failed == 0 ? 0 : 1;
    }

private:
    std::vector<TestSuite*> suites_;
    std::vector<std::unique_ptr<TestSuite>> owned_suites_;
};

} // namespace test
} // namespace hlcl

// 全局别名, 兼容遗留写法 `test::TestRunner` / `using namespace avbd::test;`
namespace test = hlcl::test;
