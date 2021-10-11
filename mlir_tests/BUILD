package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Note: run_lit.sh does not exist in open-source environment.
exports_files(["run_lit.sh"])

# LLVM lit configurations for lit test in OSS environment.
filegroup(
    name = "litcfgs",
    srcs = glob(["lit*py"]),
)

cc_test(
    name = "mlir_runner_util_test",
    testonly = 1,
    srcs = ["mlir_runner_util_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:mlir_runner_util",
        "@tf_runtime//cpp_tests:common",
    ],
)
