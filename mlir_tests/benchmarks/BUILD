# MLIR benchmarks using the C++ benchmark framework

licenses(["notice"])

cc_test(
    name = "basic_benchmark_test",
    srcs = ["basic_benchmark_test.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:mlir_runner_util",
        "@tf_runtime//cpp_tests:common",
    ],
)
