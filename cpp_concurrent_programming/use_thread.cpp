#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <format>
#include <thread>
#include <vector>
#include <random>
#include <cstdint>


#include <chrono>
#include <iostream>
#include <thread>

class Timer {
private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    bool is_running = false;

public:
    // 开始计时
    void start() {
        start_time = std::chrono::steady_clock::now();
        is_running = true;
    }
    
    // 停止计时
    void stop() {
        if (is_running) {
            end_time = std::chrono::steady_clock::now();
            is_running = false;
        }
    }
    
    // 获取经过的时间（秒）
    double elapsed_seconds() const {
        auto current_end = is_running ? std::chrono::steady_clock::now() : end_time;
        auto duration = current_end - start_time;
        return std::chrono::duration<double>(duration).count();
    }
    
    // 重置计时器
    void reset() {
        is_running = false;
    }
    
    // 检查是否正在运行
    bool running() const {
        return is_running;
    }
};

void hello()
{
    std::thread t{ [] { std::cout << "Hello World" << std::endl; } };
    t.join();
}

void count_available_concurrency()
{
    auto n = std::thread::hardware_concurrency();
    std::cout << std::format("available concurrency: {}", n) << std::endl;
}

template <typename ForwardIt>
auto sum(ForwardIt begin, ForwardIt end)
{
    using ValueType = std::iter_value_t<ForwardIt>;
    uint32_t thread_num = std::thread::hardware_concurrency();
    std::ptrdiff_t distance = std::distance(begin, end);

    if (distance > 1024000) {
        std::size_t chunk_size = distance / thread_num;
        std::size_t remainder  = distance % thread_num;

        std::vector<ValueType> sum_per_chunk(thread_num);
        std::vector<std::thread> thread_vec;

        ForwardIt start = begin;
        for (uint32_t i = 0; i < thread_num; i++) {
            ForwardIt end = std::next(start, chunk_size + (i < remainder ? 1 : 0));
            thread_vec.emplace_back(
                [start, end, i, &sum_per_chunk] {
                    sum_per_chunk[i] = std::accumulate(start, end, sum_per_chunk[i]);
                }
            );
            start = end;
        }

        for (auto& t: thread_vec) {
            t.join();
        }

        return std::accumulate(sum_per_chunk.begin(), sum_per_chunk.end(), ValueType{});
    }

    return std::accumulate(begin, end, ValueType{});
}

template <typename ForwardIt>
auto plain_sum(ForwardIt begin, ForwardIt end)
{
    return std::accumulate(begin, end, std::iter_value_t<ForwardIt>{});
}

std::vector<uint64_t> generate_random_u64_list(std::size_t n)
{
    std::random_device seed;
    std::mt19937_64 gen(seed());
    std::uniform_int_distribution<uint64_t> dis(0, 10000);

    auto random_numbers = std::vector<uint64_t>{};
    std::ranges::generate_n(std::back_inserter(random_numbers), n, [&] { return dis(gen); });

    return random_numbers;
}


int main()
{
    Timer timer;

    timer.start();
    auto random_numbers = generate_random_u64_list(1000000000);
    timer.stop();
    double elapsed_generate = timer.elapsed_seconds();
    timer.reset();
    
    timer.start();
    auto sum_1 = ::plain_sum(random_numbers.begin(), random_numbers.end());
    timer.stop();
    double elapsed_plain_sum = timer.elapsed_seconds();
    timer.reset();
    
    timer.start();
    auto sum_2 = ::sum(random_numbers.begin(), random_numbers.end());
    timer.stop();
    double elapsed_sum = timer.elapsed_seconds();
    timer.reset();

    std::cout << "number size: " << random_numbers.size() << std::endl;
    if (sum_1 == sum_2) {
        std::cout << std::format("ok\n"
                                 "elapsed_generate:  {}\n"
                                 "elapsed_plain_sum: {}\n"
                                 "elapsed_sum:       {}\n",
                                 elapsed_generate, elapsed_plain_sum, elapsed_sum)
                  << std::endl;
    } else {
        std::cout << "unequal\n"
                  << "plain_sum: " << sum_1 << "\n"
                  << "sum: " << sum_2 << std::endl;
    }

    return 0;
}
