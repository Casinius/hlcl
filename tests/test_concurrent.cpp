/**
 * Concurrent & Thread Safety Test Suite (移植到真实 avbd API)
 * 多线程下独立 Vec/Mat 运算与共享容器保护的确定性验证。
 * 注意: 本套件不共享同一引擎求解器 (引擎非线程安全), 每线程独立对象。
 */

#include "test_framework.hpp"
#include "hlcl/core.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

using namespace hlcl;
using namespace hlcl::test;

class ConcurrentTestSuite : public TestSuite {
public:
    ConcurrentTestSuite() : TestSuite("Concurrent Test Suite") {}

    void run() override {
        test_vector_thread_safety();
        test_matrix_thread_safety();
        test_atomic_operations();
        test_queue_thread_safety();
        test_parallel_vector_computation();
        test_parallel_vector_reduction();
    }

private:
    static Vec3 one() { return Vec3({1.0f, 1.0f, 1.0f}); }
    static Vec3 two() { return Vec3({4.0f, 5.0f, 6.0f}); }

    void test_vector_thread_safety() {
        std::cout << "Testing vector operations thread safety..." << std::endl;

        const int num_threads = 8;
        const int iterations_per_thread = 2000;

        std::mutex mtx;
        std::vector<Vec3> results;
        std::atomic<int> counter(0);

        auto worker = [&]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                Vec3 r = one() + two();          // {5, 6, 7}
                Vec3 r2 = 2.0f * one();          // {2, 2, 2}
                float dot = one() * two();       // 1*4+1*5+1*6 = 15
                std::lock_guard<std::mutex> lock(mtx);
                results.push_back(r);
                results.push_back(r2);
                counter += (dot == 15.0f) ? 1 : 0;
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

        assert_true(counter == num_threads * iterations_per_thread,
                    "all dot products correct (atomic counter)");
        assert_true(results.size() == static_cast<size_t>(num_threads * iterations_per_thread * 2),
                    "all results recorded");

        // 每个结果内容正确且完整 (无撕裂/损坏)
        for (size_t i = 0; i < results.size(); i += 2) {
            const Vec3& r = results[i];
            assert_true(r.x() == 5.0f && r.y() == 6.0f && r.z() == 7.0f,
                        "vector add result intact");
            const Vec3& s = results[i + 1];
            assert_true(s.x() == 2.0f && s.y() == 2.0f && s.z() == 2.0f,
                        "vector scale result intact");
        }

        std::cout << "  ✓ Vector operations thread safety test passed" << std::endl;
    }

    void test_matrix_thread_safety() {
        std::cout << "Testing matrix operations thread safety..." << std::endl;

        const int num_threads = 8;
        const int iterations_per_thread = 2000;

        std::mutex mtx;
        std::atomic<int> ok_count(0);

        auto worker = [&]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                // I * diag(2) = diag(2); 每线程局部对象
                Mat<3, 3, Backend::CPU> diag2;
                diag2.setDiagonal({2.0f, 2.0f, 2.0f});
                Mat<3, 3, Backend::CPU> r = identity<float, 3>() * diag2;
                if (r(0, 0) == 2.0f && r(1, 1) == 2.0f && r(2, 2) == 2.0f &&
                    r(0, 1) == 0.0f) {
                    ++ok_count;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

        assert_true(ok_count == num_threads * iterations_per_thread,
                    "matrix product results all correct");

        std::cout << "  ✓ Matrix operations thread safety test passed" << std::endl;
    }

    void test_atomic_operations() {
        std::cout << "Testing atomic operations..." << std::endl;

        const int num_threads = 8;
        const int iterations_per_thread = 10000;
        std::atomic<int> counter(0);

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < iterations_per_thread; ++j) {
                    counter++;
                }
            });
        }
        for (auto& t : threads) t.join();

        assert_true(counter == num_threads * iterations_per_thread,
                    "atomic counter totals exactly");

        std::cout << "  ✓ Atomic operations test passed" << std::endl;
    }

    void test_queue_thread_safety() {
        std::cout << "Testing queue thread safety..." << std::endl;

        const int producers = 4;
        const int consumers = 4;
        const int items_per_producer = 500;

        std::mutex mtx;
        std::condition_variable cv;
        std::queue<Vec3> q;
        int done_producers = 0;
        std::atomic<int> pushed(0);
        std::atomic<int> popped(0);

        auto producer = [&]() {
            for (int i = 0; i < items_per_producer; ++i) {
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    q.push(one());
                }
                pushed++;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                ++done_producers;
            }
            cv.notify_all();
        };

        auto consumer = [&]() {
            for (;;) {
                Vec3 v;
                bool got = false;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&]() { return !q.empty() || done_producers == producers; });
                    if (q.empty() && done_producers == producers) break;
                    if (!q.empty()) {
                        v = q.front();
                        q.pop();
                        got = true;
                    }
                }
                if (got) {
                    if (v.x() == 1.0f && v.y() == 1.0f && v.z() == 1.0f) popped++;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < producers; ++i) threads.emplace_back(producer);
        for (int i = 0; i < consumers; ++i) threads.emplace_back(consumer);
        for (auto& t : threads) t.join();

        assert_true(pushed == producers * items_per_producer, "queue: all items pushed");
        assert_true(popped == producers * items_per_producer, "queue: all items popped intact");

        std::lock_guard<std::mutex> lock(mtx);
        assert_true(q.empty(), "queue empty at end");

        std::cout << "  ✓ Queue thread safety test passed" << std::endl;
    }

    void test_parallel_vector_computation() {
        std::cout << "Testing parallel vector computation..." << std::endl;

        const int num_vectors = 1000;
        std::vector<Vec3> vectors;
        for (int i = 0; i < num_vectors; ++i) {
            vectors.push_back(Vec3({static_cast<float>(i + 1) * 0.1f,
                                    static_cast<float>(i + 1) * 0.2f,
                                    static_cast<float>(i + 1) * 0.3f}));
        }

        std::mutex mtx;
        std::vector<double> lengths;
        const int chunk = 100;
        std::vector<std::thread> threads;

        for (int start = 0; start < num_vectors; start += chunk) {
            threads.emplace_back([&, start]() {
                for (int j = start; j < start + chunk && j < num_vectors; ++j) {
                    double len = vectors[j].norm();
                    std::lock_guard<std::mutex> lock(mtx);
                    lengths.push_back(len);
                }
            });
        }
        for (auto& t : threads) t.join();

        assert_true(lengths.size() == static_cast<size_t>(num_vectors),
                    "parallel length computed for every vector");
        for (double len : lengths) {
            assert_true(len > 0.0, "all lengths positive");
        }

        std::cout << "  ✓ Parallel vector computation test passed" << std::endl;
    }

    void test_parallel_vector_reduction() {
        std::cout << "Testing parallel vector reduction..." << std::endl;

        const int num_vectors = 1000;
        std::vector<Vec3> vectors;
        for (int i = 0; i < num_vectors; ++i) {
            vectors.push_back(Vec3({static_cast<float>(i), static_cast<float>(i),
                                    static_cast<float>(i)}));
        }

        std::mutex mtx;
        std::vector<Vec3> partial_sums;
        const int chunk = 100;
        std::vector<std::thread> threads;

        for (int start = 0; start < num_vectors; start += chunk) {
            threads.emplace_back([&, start]() {
                Vec3 acc;
                acc.setZero();
                for (int j = start; j < start + chunk && j < num_vectors; ++j) {
                    acc += vectors[j];
                }
                std::lock_guard<std::mutex> lock(mtx);
                partial_sums.push_back(acc);
            });
        }
        for (auto& t : threads) t.join();

        Vec3 total;
        total.setZero();
        for (const Vec3& p : partial_sums) total += p;

        // sum(i, i=0..999) = 999*1000/2 = 499500 (float 精确可表示, < 2^24)
        const float expected = (num_vectors - 1) * num_vectors / 2.0f;
        assert_true(total.x() == expected && total.y() == expected && total.z() == expected,
                    "parallel reduction exact total");

        std::cout << "  ✓ Parallel vector reduction test passed" << std::endl;
    }
};

int main() {
    TestRunner runner;

    runner.add_suite(std::make_unique<ConcurrentTestSuite>());

    int code = runner.run_all();

    return code;
}
