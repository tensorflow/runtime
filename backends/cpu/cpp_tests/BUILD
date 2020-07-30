load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "kernels/cwise_binary_kernels_test",
    srcs = ["kernels/cwise_binary_kernels_test.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:tf_bcast",
        "@tf_runtime//backends/cpu:cpu_kernels",
    ],
)

tfrt_cc_test(
    name = "ops/tf/buffer_forwarding_test",
    srcs = ["ops/tf/buffer_forwarding_test.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:buffer_forwarding",
    ],
)
