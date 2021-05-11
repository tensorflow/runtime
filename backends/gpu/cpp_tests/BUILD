load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "wrapper/wrapper_test",
    srcs = [
        "wrapper/blas_wrapper_test.cc",
        "wrapper/common.h",
        "wrapper/cufft_wrapper_test.cc",
        "wrapper/dnn_wrapper_test.cc",
        "wrapper/driver_wrapper_test.cc",
        "wrapper/instantiate_suite.cc",
        "wrapper/runtime_wrapper_test.cc",
        "wrapper/solver_wrapper_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//backends/gpu:gpu_wrapper",
        "@tf_runtime//cpp_tests:common",
        "@tf_runtime//:support",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "@tf_runtime//:is_build_env_google": ["//third_party/absl/debugging:leak_check"],
        "//conditions:default": [],
    }),
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
        "@tf_runtime//backends/gpu:gpu_wrapper",
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
        "@tf_runtime//backends/gpu:gpu_wrapper",
        "@tf_runtime//cpp_tests:common",
    ],
)
