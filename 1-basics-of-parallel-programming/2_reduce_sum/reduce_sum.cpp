// Include headers for timing, I/O, standard library, threading, and data structures
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  // Initialize array size N, thread count T (from command line or default 4), and data vector filled with 0.5
  const size_t N = 100'000'000;
  const int T = argc > 1 ? std::atoi(argv[1]) : 4;
  std::vector<float> data(N, 0.5f);

  // Compute sequential sum as reference result
  auto t_seq_0 = std::chrono::steady_clock::now();
  double seq = 0;
  for (size_t i = 0; i < N; ++i) seq += data[i];
  auto t_seq_1 = std::chrono::steady_clock::now();
  double seq_time = std::chrono::duration<double, std::milli>(t_seq_1 - t_seq_0).count();

  // Fast: per-thread partial sums (map), combine at end (reduce).
  // Create vector to store partial sums from each thread
  std::vector<double> partial(T, 0.0);
  // Create vector to hold thread objects
  std::vector<std::thread> ts;
  // Record start time
  auto t0 = std::chrono::steady_clock::now();
  // Launch T threads, each computing partial sum for their range
  for (int t = 0; t < T; ++t)
    ts.emplace_back([&, t]() {
    // Calculate range boundaries for this thread
    size_t lo = t * N / T, hi = (t + 1) * N / T;
    // Initialize local accumulator
    double s = 0;
    // Sum assigned range of data
    for (size_t i = lo; i < hi; ++i) s += data[i];
    // Store partial sum in shared vector
    partial[t] = s;
      });
  // Wait for all threads to complete
  for (auto& th : ts) th.join();
  // Reduce: combine all partial sums into final parallel result
  double par = 0;
  for (double p : partial) par += p;
  // Record end time
  auto t1 = std::chrono::steady_clock::now();

  // Calculate relative error between parallel and sequential results
  double rel = std::abs(par - seq) / seq;
  // Print results with sequential sum, parallel sum, relative error, and execution time
  printf("seq=%.1f par=%.1f rel_err=%g seq_time=%.1f ms par_time=%.1f ms\n", seq, par, rel,
    seq_time, std::chrono::duration<double, std::milli>(t1 - t0).count());
  // Return success if error is small, otherwise failure
  return rel < 1e-6 ? 0 : 1;
}