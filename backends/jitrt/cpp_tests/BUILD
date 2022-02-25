load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "jit_executable_test",
    srcs = ["jit_executable_test.cc"],
    tags = ["no_oss"],  # TODO(b/197262906)
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//backends/jitrt",
        "@tf_runtime//backends/jitrt:jitrt_compiler",
    ],
)

tfrt_cc_test(
    name = "return_value_converter_test",
    srcs = ["return_value_converter_test.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:dtype",
        "@tf_runtime//backends/jitrt",
    ],
)

tfrt_cc_test(
    name = "symbolic_shapes_resolver_test",
    srcs = ["symbolic_shapes_resolver_test.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//backends/jitrt",
        "@tf_runtime//backends/jitrt:constraints",
    ],
)

tfrt_cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//backends/jitrt:types",
    ],
)
