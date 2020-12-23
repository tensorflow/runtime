load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_py_binary")
load("@tf_runtime//mlir_tests/bef_perf:gen_benchmark.bzl", "gen_benchmark")

licenses(["notice"])

package(default_visibility = [
    # copybara:uncomment "//third_party/tensorflow/core/runtime_fallback:__subpackages__",
    "@tf_runtime//:__subpackages__",
    # copybara:uncomment "//third_party/tf_runtime_google:__subpackages__",
])

py_library(
    name = "benchmark_utils_lib",
    srcs = ["benchmark_utils.py"],
    srcs_version = "PY3",
    deps = ["@py-cpuinfo//:cpuinfo"],
)

tfrt_py_binary(
    name = "bef_perf",
    testonly = True,
    srcs = ["bef_perf.py"],
    data = [
        "@tf_runtime//tools:bef_executor",
        "@tf_runtime//tools:tfrt_translate",
    ],
    python_version = "PY3",
    deps = [
        ":benchmark_utils_lib",
    ],
)

sh_test(
    name = "bef_perf_test",
    size = "small",
    srcs = ["bef_perf_test.sh"],
    data = [
        ":bef_perf",
        ":fully_parallel.mlir",
        ":fully_serial.mlir",
        ":star.mlir",
    ],
)

py_library(
    name = "gen_benchmark_mlir_lib",
    srcs = ["gen_benchmark_mlir_lib.py"],
    srcs_version = "PY3",
)

tfrt_py_binary(
    name = "gen_benchmark_mlir",
    srcs = ["gen_benchmark_mlir.py"],
    main = "gen_benchmark_mlir.py",
    python_version = "PY3",
    deps = [":gen_benchmark_mlir_lib"],
)

gen_benchmark(benchmark_name = "fully_serial")

gen_benchmark(benchmark_name = "fully_parallel")

gen_benchmark(benchmark_name = "star")

bzl_library(
    name = "gen_benchmark_bzl",
    srcs = ["gen_benchmark.bzl"],
    visibility = ["//visibility:private"],
)
