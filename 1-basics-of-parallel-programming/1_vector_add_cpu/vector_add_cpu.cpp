#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  const size_t N = 100'000'000;
  const int T = argc > 1 ? std::atoi(argv[1]) : 4;
  std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N), c_ref(N);

  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < N; ++i) c_ref[i] = a[i] + b[i];
  auto t1 = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  for (int t = 0; t < T; ++t)
    threads.emplace_back([&, t]() {
    size_t lo = t * N / T, hi = (t + 1) * N / T;
    for (size_t i = lo; i < hi; ++i) c[i] = a[i] + b[i];
      });
  for (auto& thread : threads) thread.join();
  auto t2 = std::chrono::steady_clock::now();

  float err = 0;
  for (size_t i = 0; i < N; ++i) err = std::fmax(err, std::fabs(c[i] - c_ref[i]));
  auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
  printf("serial: %.1f ms, %d threads: %.1f ms, speedup %.2fx, max error: %g\n",
    ms(t1 - t0), T, ms(t2 - t1), ms(t1 - t0) / ms(t2 - t1), err);
}