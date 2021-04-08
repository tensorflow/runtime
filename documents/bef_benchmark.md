# TFRT Benchmark Framework

<!--* freshness: {
  owner: 'jingdong'
  reviewed: '2021-04-07'
} *-->

<!-- TOC -->

This document describes the design, usage, and implementation of the benchmark
kernel in TFRT for benchmarking BEF programs.

## Design Philosophy

The TFRT `test.benchmark` kernel is designed to allow the user to benchmark
arbitrary BEF functions from MLIR code, without the need to compile the
benchmark code into a binary. This is akin to the testing approach adopted by
the TFRT project where we write testing code in MLIR instead of compiling the
test code into a binary, but different from the
[Google Benchmark](https://github.com/google/benchmark) approach where the
target code is hard-coded in C++ and needs to be compiled into a benchmark
binary. The benefit of defining benchmark functions in MLIR instead of
hard-coding in C++ is that it allows easy experimentations of different BEF
functions without a proliferation of benchmark binaries.

## Usage

To use the benchmark kernel, the user needs to take the following two steps:

1.  define the target function to be benchmarked in MLIR,
2.  run the benchmark with the benchmark driver,
    [`bef_perf.py`](https://github.com/tensorflow/runtime/blob/master/mlir_tests/bef_perf/bef_perf.py).

### Define the function to be benchmarked in MLIR

Example:

```c++
 func @benchmark() {
   %c = tfrt.constant.i32 42

   tfrt_test.benchmark "add.i32" (%c : i32)
     duration_secs = 1,    // benchmark duration in seconds
     max_count = 100,      // the maximum repeated runs for the benchmark
     num_warmup_runs = 10  // the number of warm up runs
   {
     // The MLIR code to be benchmarked goes here.
     // The following code benchmarks the tfrt.constant.i32 and tfrt.add.i32 kernel
     %x = tfrt.add.i32 %c, %c
     tfrt.return %x : i32    // the benchmarked function needs to return exactly one value
   }

   tfrt.return  // The return statement is necessary for any bef function.
 }
```

NOTE: The benchmarked function needs to return exactly one value. The exact
value returned is not important, but is only used to notify the benchmark kernel
of the completion of one run, so it can start the next run.

### Run the benchmark code with bef_perf.py.

[`bef_perf.py`](https://github.com/tensorflow/runtime/blob/master/mlir_tests/bef_perf/bef_perf.py)
is located in
[`mlir_tests/bef_perf/bef_perf.py`](https://github.com/tensorflow/runtime/blob/master/mlir_tests/bef_perf/).
See the example commands below to build `bef_perf` and pass the names of
generated `.mlir` files as command line arguments.

Benchmark `.mlir` files are produced by
[`gen_benchmark_mlir.py`](https://github.com/tensorflow/runtime/blob/master/mlir_tests/bef_perf/gen_benchmark_mlir.py),
which runs from genrules in
[`mlir_tests/bef_perf/BUILD`](https://github.com/tensorflow/runtime/blob/master/mlir_tests/bef_perf/BUILD).
You can find the generated files in `bazel-bin/mlir_tests/bef_perf/*.mlir`. You
can also run `gen_benchmark_mlir.py` manually.

```shell
$ bef_perf=mlir_tests/bef_perf
$ bazel build $bef_perf/...
$ bazel-bin/$bef_perf/bef_perf bazel-bin/$bef_perf/*.mlir
```

```shell
----------------------------------------
Running benchmarks in bazel-bin/mlir_tests/bef_perf/fully_parallel.mlir
Running benchmarks in bazel-bin/mlir_tests/bef_perf/fully_serial.mlir
Running benchmarks in bazel-bin/mlir_tests/bef_perf/star.mlir
----------------------------------------
CPU Info: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
Num cores: 56
Frequency: 2600.0 MHz
                          Duration(us)       Count      Time Min(us)   Time 50%(us)   Time 95%(us)   Time 99%(us)    CPU Min(us)    CPU 50%(us)    CPU 95%(us)    CPU 99%(us)
BM_full_parallel_100         1000006         58564            15             17             17             27             15             17             18             27
BM_full_serial_100            916219         100001           8              9              9              15             9              10             10             16
BM_star_100                  1000010         49242            18             20             20             36             18             20             21             36
```

## Implementation Details

The benchmark framework consists of several main components, the
`test.benchmark` kernel, the benchmark driver, `bef_perf.py`, and the CPU
profile generator, `bef_profile.py` .

### tfrt_test.benchmark kernel

The `test.benchmark` kernel is implemented in
[`lib/test_kernels/benchmark_kernels.cc`](https://github.com/tensorflow/runtime/blob/master/lib/test_kernels/benchmark_kernels.cc)
The kernel takes the following arguments as constant attributes:

*   benchmark duration in seconds,
*   maximum run count,
*   the name of the benchmark,
*   the number of warm up runs before the benchmark,
*   the BEF function to be benchmarked.

The benchmark kernel runs the target function to be benchmarked repeatedly until
either benchmark duration or max run count is reached. Due to the requirement of
asynchronous execution for TFRT kernels, the repeated kernel executions are
implemented with the continuation passing style, i.e. the start of the next
execution is scheduled in the completion callback of the previous execution.
This is the reason why the benchmarked function must have exactly one return
value.

At the end of the execution, the benchmark kernel prints a summary of the run
statistics, including the minimum run time, the median run time, the 90th
percentile run time, etc.

### Benchmark driver: `bef_perf.py`

`bef_perf.py` is the benchmark driver program that takes a set of MLIR files
containing benchmark functions and performs the following steps:

1.  Run the input MLIR files using `tfrt_translate` and `bef_executor`,
1.  Parse the output of the `test.benchmark` kernel and print it in a tabular
    format.
