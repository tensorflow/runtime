load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_library(
    name = "common",
    testonly = True,
    hdrs = [
        "include/tfrt/cpp_tests/error_util.h",
        "include/tfrt/cpp_tests/test_util.h",
    ],
    visibility = [
        ":__subpackages__",
        "@tf_runtime//backends/gpu/cpp_tests:__subpackages__",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/parallel_for_test",
    srcs = [
        "host_context/parallel_for_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/concurrent_vector_test",
    srcs = [
        "support/concurrent_vector_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/dense_host_tensor_test",
    srcs = [
        "support/dense_host_tensor_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "support/error_test",
    srcs = [
        "support/error_test.cc",
    ],
    deps = [
        ":common",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:support",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/latch_test",
    srcs = [
        "support/latch_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "core_runtime/driver_test",
    srcs = [
        "core_runtime/driver.cc",
        "core_runtime/driver.h",
        "core_runtime/driver_test.cc",
    ],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:test_ops_alwayslink",
    ],
)

tfrt_cc_test(
    name = "core_runtime/op_attrs_test",
    srcs = [
        "core_runtime/op_attrs_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "host_runtime/async_value_test",
    srcs = ["host_runtime/async_value_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/aligned_buffer_test",
    srcs = ["support/aligned_buffer_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_runtime/concurrent_work_queue_test",
    srcs = ["host_runtime/concurrent_work_queue_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

# Options to pass to 'bazel test' that affect what's measured:
# --copt=-DTFRT_DISABLE_TRACING:            strip tracing code.
# --copt=-DTFRT_BM_DISABLE_TRACING_REQUEST: do not request tracing.
tfrt_cc_test(
    name = "tracing",
    srcs = [
        "tracing/tracing_benchmark.cc",
        "tracing/tracing_test.cc",
    ],
    deps = [
        ":common",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:support",
        "@tf_runtime//:support",
        "@tf_runtime//:tracing",
    ],
)
