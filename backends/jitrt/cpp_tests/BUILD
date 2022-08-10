load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

# TODO(b/240450920): We are in the process of migrating JitRt backend to XLA
# and while we are doing this we can't keep it buildable/testable in OSS.

# copybara:uncomment_begin
#
# tfrt_cc_test(
#     name = "aot_compilation_test",
#     srcs = ["aot_compilation_test.cc"],
#     tags = ["no_oss"],  # TODO(b/197262906)
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//backends/jitrt",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
#     ],
# )
#
# tfrt_cc_test(
#     name = "custom_call_test",
#     srcs = ["custom_call_test.cc"],
#     tags = ["no_oss"],  # TODO(b/197262906)
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:Support",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:support",
#         "@tf_runtime//backends/jitrt",
#         "@tf_runtime//backends/jitrt:custom_call",
#         "@tf_runtime//backends/jitrt:custom_call_testlib",
#         "@tf_runtime//backends/jitrt:diagnostics",
#         "@tf_runtime//backends/jitrt:execution_engine",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
#     ],
# )
#
# tfrt_cc_test(
#     name = "diagnostics_test",
#     srcs = ["diagnostics_test.cc"],
#     tags = ["no_oss"],  # TODO(b/197262906)
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:Support",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:support",
#         "@tf_runtime//backends/jitrt:diagnostics",
#     ],
# )
#
# tfrt_cc_test(
#     name = "end_to_end_example_test",
#     srcs = ["end_to_end_example_test.cc"],
#     tags = ["no_oss"],  # TODO(b/197262906)
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:BufferizationTransforms",
#         "@llvm-project//mlir:FuncTransforms",
#         "@llvm-project//mlir:LLVMDialect",
#         "@llvm-project//mlir:LinalgTransforms",
#         "@llvm-project//mlir:TosaDialect",
#         "@llvm-project//mlir:TosaToLinalg",
#         "@llvm-project//mlir:Transforms",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#         "@tf_runtime//backends/jitrt",
#         "@tf_runtime//backends/jitrt:custom_call_registry",
#         "@tf_runtime//backends/jitrt:custom_call_testlib",
#         "@tf_runtime//backends/jitrt:custom_call_to_llvm",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
#         "@tf_runtime//backends/jitrt:xla",
#     ],
# )
#
# tfrt_cc_test(
#     name = "jit_executable_test",
#     srcs = ["jit_executable_test.cc"],
#     tags = ["no_oss"],  # TODO(b/197262906)
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//backends/jitrt",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
#     ],
# )
#
# tfrt_cc_test(
#     name = "return_value_converter_test",
#     srcs = ["return_value_converter_test.cc"],
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:mlir_c_runner_utils",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//backends/jitrt",
#     ],
# )
#
# tfrt_cc_test(
#     name = "symbolic_shapes_resolver_test",
#     srcs = ["symbolic_shapes_resolver_test.cc"],
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "//third_party/tensorflow/compiler/xla/runtime:arguments",
#         "//third_party/tensorflow/compiler/xla/runtime:types",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//backends/jitrt:constraints",
#         "@tf_runtime//backends/jitrt:symbolic_shape",
#         "@tf_runtime//backends/jitrt:xla",
#     ],
# )
#
# copybara:uncomment_end
