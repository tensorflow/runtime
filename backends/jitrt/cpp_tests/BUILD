load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "aot_compilation_test",
    srcs = ["aot_compilation_test.cc"],
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
    name = "custom_call_test",
    srcs = ["custom_call_test.cc"],
    tags = ["no_oss"],  # TODO(b/197262906)
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
        "@tf_runtime//backends/jitrt",
        "@tf_runtime//backends/jitrt:custom_call",
        "@tf_runtime//backends/jitrt:custom_call_testlib",
        "@tf_runtime//backends/jitrt:execution_engine",
        "@tf_runtime//backends/jitrt:jitrt_compiler",
    ],
)

tfrt_cc_test(
    name = "end_to_end_example_test",
    srcs = ["end_to_end_example_test.cc"],
    tags = ["no_oss"],  # TODO(b/197262906)
    deps = [
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TosaToLinalg",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/jitrt",
        "@tf_runtime//backends/jitrt:jitrt_compiler",
    ],
)

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
