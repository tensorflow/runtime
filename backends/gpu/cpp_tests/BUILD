load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "stream/wrapper_test",
    srcs = [
        "stream/blas_wrapper_test.cc",
        "stream/common.h",
        "stream/dnn_wrapper_test.cc",
        "stream/instantiate_suite.cc",
        "stream/runtime_wrapper_test.cc",
        "stream/solver_wrapper_test.cc",
        "stream/stream_wrapper_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
        "@tf_runtime//backends/gpu:gpu_stream_wrapper",
        "@tf_runtime//cpp_tests:common",
    ],
)

tfrt_cc_test(
    name = "memory/sub_allocator_test",
    srcs = [
        "memory/sub_allocator_test.cc",
    ],
    tags = [
        "noasan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//backends/gpu:gpu_memory",
        "@tf_runtime//cpp_tests:common",
    ],
)

tfrt_cc_test(
    name = "memory/block_allocator_test",
    srcs = [
        "memory/block_allocator_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//backends/gpu:gpu_memory",
        "@tf_runtime//cpp_tests:common",
    ],
)

tfrt_cc_test(
    name = "event_manager_test",
    srcs = [
        "event_manager_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//backends/gpu:gpu_event_manager",
        "@tf_runtime//backends/gpu:gpu_stream_wrapper",
        "@tf_runtime//cpp_tests:common",
    ],
)

tfrt_cc_test(
    name = "memory/gpu_buffer_test",
    srcs = [
        "memory/gpu_buffer_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//backends/gpu:gpu_memory",
        "@tf_runtime//cpp_tests:common",
    ],
)

tfrt_cc_test(
    name = "module_table_test",
    srcs = [
        "module_table_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
        "@tf_runtime//backends/gpu:gpu_stream_wrapper",
        "@tf_runtime//backends/gpu:module_table",
        "@tf_runtime//cpp_tests:common",
    ],
)
