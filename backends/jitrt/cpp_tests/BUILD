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
#         "//third_party/tensorflow/compiler/xla/runtime:compiler",
#         "//third_party/tensorflow/compiler/xla/runtime:jit_executable",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//backends/jitrt:arguments",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
#     ],
# )
#
# tfrt_cc_test(
#     name = "calling_convention_test",
#     srcs = ["calling_convention_test.cc"],
#     tags = ["no_oss"],  # TODO(b/197262906)
#     deps = [
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:BufferizationTransforms",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:calling_convention",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
#         "//third_party/tensorflow/compiler/xla/runtime:compiler",
#         "//third_party/tensorflow/compiler/xla/runtime:jit_executable",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//backends/jitrt:arguments",
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
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
#         "//third_party/tensorflow/compiler/xla/runtime:arguments",
#         "//third_party/tensorflow/compiler/xla/runtime:custom_call",
#         "//third_party/tensorflow/compiler/xla/runtime:diagnostics",
#         "//third_party/tensorflow/compiler/xla/runtime:executable",
#         "//third_party/tensorflow/compiler/xla/runtime:execution_engine",
#         "//third_party/tensorflow/compiler/xla/runtime:jit_executable",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:support",
#         "@tf_runtime//backends/jitrt:custom_call_testlib",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
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
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:BufferizationTransforms",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:FuncTransforms",
#         "@llvm-project//mlir:LLVMDialect",
#         "@llvm-project//mlir:LinalgTransforms",
#         "@llvm-project//mlir:TosaDialect",
#         "@llvm-project//mlir:TosaToLinalg",
#         "@llvm-project//mlir:Transforms",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:calling_convention",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
#         "//third_party/tensorflow/compiler/xla/runtime:arguments",
#         "//third_party/tensorflow/compiler/xla/runtime:custom_call_registry",
#         "//third_party/tensorflow/compiler/xla/runtime:executable",
#         "//third_party/tensorflow/compiler/xla/runtime:jit_executable",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#         "@tf_runtime//backends/jitrt:custom_call_testlib",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
#         "@tf_runtime//backends/jitrt:results",
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
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
#         "//third_party/tensorflow/compiler/xla/runtime:jit_executable",
#         "//third_party/tensorflow/compiler/xla/runtime:symbolic_shape",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//backends/jitrt:arguments",
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
#         "@tf_runtime//backends/jitrt:results",
#     ],
# )
#
# copybara:uncomment_end
