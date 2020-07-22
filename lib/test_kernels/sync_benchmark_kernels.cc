// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- sync_benchmark_kernels.cc ------------------------------------------===//
//
// This file implements ops for benchmarking BEFInterpreter.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <chrono>
#include <ctime>

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/test_kernels.h"

namespace tfrt {

// TODO(jingdong): Refactor benchmark_kernels.cc to also use this class for
// keeping track of benchmark stats.
class BenchmarkStats {
 public:
  BenchmarkStats(string_view name, int num_warmup_runs, int max_count,
                 std::chrono::microseconds benchmark_duration)
      : name_{name},
        num_warmup_runs_{num_warmup_runs},
        max_count_{max_count},
        benchmark_duration_{benchmark_duration} {}

  void StartRun() {
    ++cur_count_;

    // Start recording CPU time.
    cur_start_cpu_ = std::clock();

    cur_start_walltime_ = std::chrono::steady_clock::now();
  }

  void StopRun() {
    // Do not collect the runtime statistics if we are still in the warm up
    // period.
    if (cur_count_ < num_warmup_runs_) return;

    // Stop the wall clock timer.
    auto cur_stop_walltime_ = std::chrono::steady_clock::now();

    // Stop the CPU timer.
    std::clock_t cur_stop_cpu_ = std::clock();

    // Collect the wall clock duration.
    auto duration_walltime_ = cur_stop_walltime_ - cur_start_walltime_;
    run_times_walltime_.push_back(duration_walltime_);

    // Collect the CPU duration in microseconds.
    // First cast to integer that represents microseconds with truncation, as
    // does std::chrono::duration_cast. Then cast to std::chrono::microseconds.
    std::clock_t duration_cpu_raw = cur_stop_cpu_ - cur_start_cpu_;
    auto duration_cpu_ = static_cast<std::chrono::nanoseconds>(
        static_cast<int64_t>(1e9 * duration_cpu_raw / CLOCKS_PER_SEC));

    run_times_cpu_.push_back(duration_cpu_);

    total_duration_walltime_ += duration_walltime_;
  }

  // Return if we should we run more rounds.
  bool MoreRun() const {
    return cur_count_ < max_count_ + num_warmup_runs_ &&
           total_duration_walltime_ < benchmark_duration_;
  }

  // Summarize the benchmark results.
  void Summarize() {
    std::sort(run_times_walltime_.begin(), run_times_walltime_.end());
    std::sort(run_times_cpu_.begin(), run_times_cpu_.end());

    auto percentile =
        [](double p, const std::vector<std::chrono::nanoseconds>& run_times) {
          assert(p >= 0.0 && p <= 1.0);
          return run_times[run_times.size() * p];
        };

    // BM: prefix is added to make grepping results from lit output easier.
    std::string prefix;
    llvm::raw_string_ostream(prefix) << "BM:" << name_ << ':';

    tfrt::outs() << prefix
                 << "Duration(ns): " << total_duration_walltime_.count()
                 << '\n';
    tfrt::outs() << prefix << "Count: " << run_times_walltime_.size() << '\n';
    tfrt::outs() << prefix
                 << "Time Min(ns): " << run_times_walltime_.front().count()
                 << '\n';
    tfrt::outs() << prefix << "Time 50%(ns): "
                 << percentile(0.5, run_times_walltime_).count() << '\n';
    tfrt::outs() << prefix << "Time 95%(ns): "
                 << percentile(0.95, run_times_walltime_).count() << '\n';
    tfrt::outs() << prefix << "Time 99%(ns): "
                 << percentile(0.99, run_times_walltime_).count() << '\n';

    // Log CPU time statistics.
    tfrt::outs() << prefix << "CPU Min(ns): " << run_times_cpu_.front().count()
                 << '\n';
    tfrt::outs() << prefix
                 << "CPU 50%(ns): " << percentile(0.5, run_times_cpu_).count()
                 << '\n';
    tfrt::outs() << prefix
                 << "CPU 95%(ns): " << percentile(0.95, run_times_cpu_).count()
                 << '\n';
    tfrt::outs() << prefix
                 << "CPU 99%(ns): " << percentile(0.99, run_times_cpu_).count()
                 << '\n';
    tfrt::outs().flush();
  }

 private:
  const std::string name_;
  const int num_warmup_runs_;
  const int max_count_;
  int cur_count_ = 0;
  const std::chrono::nanoseconds benchmark_duration_;
  std::chrono::nanoseconds total_duration_walltime_{};
  std::chrono::time_point<std::chrono::steady_clock> cur_start_walltime_{};
  std::clock_t cur_start_cpu_;
  std::vector<std::chrono::nanoseconds> run_times_walltime_;
  // CPU run times in microseconds.
  std::vector<std::chrono::nanoseconds> run_times_cpu_;
};

// This op benchmarks the input BEF function by running the function in a loop
// up to a max count or max time as specified in the function's attributes.
//
// Attributes:
// duration_secs: Benchmark duration in seconds.
// max_count: Max run count of input function.
// num_warmup_runs: Number of warm up runs before benchmarking starts.
// fn_const: The input function to be benchmarked.
static Error TestSyncBenchmark(RemainingSyncArguments args,
                               Attribute<int32_t> duration_secs,
                               Attribute<int32_t> max_count,
                               Attribute<int32_t> num_warmup_runs,
                               Attribute<Function> fn_const,
                               const ExecutionContext& exec_ctx) {
  const Function* fn = &(*fn_const);

  if (fn->function_kind() != FunctionKind::kSyncBEFFunction) {
    return MakeStringError(
        "SyncBenchmark op requires the input function be a sync function");
  }

  if (fn->num_results() != 0) {
    return MakeStringError(
        "SyncBenchmark op requires the input function have zero return "
        "value");
  }

  if (fn->num_arguments() != args.size()) {
    return MakeStringError(
        "Incorrect number of arguments for the target function for the "
        "SyncBenchmark op");
  }

  BenchmarkStats bm_stats{fn->name(), *num_warmup_runs, *max_count,
                          std::chrono::seconds(*duration_secs)};

  SmallVector<Value*, 16> func_args;
  for (auto i = 0; i < args.size(); ++i) {
    func_args.emplace_back(args[i]);
  }

  while (bm_stats.MoreRun()) {
    bm_stats.StartRun();
    // TODO(jingdong): Expose BEFInterpreter and SyncBEFFunction so we can
    // factor the warm up cost out of the benchmark to make the benchmark
    // results more accurate.
    auto error = ExecuteSyncBEFFunction(*fn, exec_ctx, func_args, {});
    bm_stats.StopRun();
    if (error) return error;
  }

  bm_stats.Summarize();

  return Error::success();
}

void RegisterSyncBenchmarkKernels(KernelRegistry* registry) {
  registry->AddSyncKernel("tfrt_test.sync_benchmark",
                          TFRT_SYNC_KERNEL(TestSyncBenchmark));
}
}  // namespace tfrt
