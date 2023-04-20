load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

# TODO(b/240450920): We are in the process of migrating JitRt backend to XLA
# and while we are doing this we can't keep it buildable/testable in OSS.

# copybara:uncomment_begin
#
# td_library(
#     name = "jitrt_ops_td_files",
#     srcs = [
#         "include/tfrt/jitrt/opdefs/jitrt_base.td",
#         "include/tfrt/jitrt/opdefs/jitrt_ops.td",
#     ],
#     compatible_with = ["//buildenv/target:gce"],
#     includes = ["include"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "@tf_runtime//:CoreRTTdFiles",
#         "@tf_runtime//:OpBaseTdFiles",
#     ],
# )
#
# gentbl_cc_library(
#     name = "jitrt_opdefs_inc_gen",
#     compatible_with = ["//buildenv/target:gce"],
#     tbl_outs = [
#         (
#             [
#                 "-gen-dialect-decls",
#                 "-dialect=jitrt",
#             ],
#             "include/tfrt/jitrt/opdefs/jitrt_ops_dialect.h.inc",
#         ),
#         (
#             [
#                 "-gen-dialect-defs",
#                 "-dialect=jitrt",
#             ],
#             "include/tfrt/jitrt/opdefs/jitrt_ops_dialect.cpp.inc",
#         ),
#         (
#             ["-gen-op-decls"],
#             "include/tfrt/jitrt/opdefs/jitrt_ops.h.inc",
#         ),
#         (
#             ["-gen-op-defs"],
#             "include/tfrt/jitrt/opdefs/jitrt_ops.cpp.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "include/tfrt/jitrt/opdefs/jitrt_ops.td",
#     deps = [":jitrt_ops_td_files"],
# )
#
# tfrt_cc_library(
#     name = "jitrt_opdefs",
#     srcs = ["lib/opdefs/jitrt_ops.cc"],
#     hdrs = ["include/tfrt/jitrt/opdefs/jitrt_ops.h"],
#     compatible_with = ["//buildenv/target:gce"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":jitrt_opdefs_inc_gen",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:SideEffectInterfaces",
#         "@llvm-project//mlir:Support",
#         "@tf_runtime//:basic_kernels_opdefs",
#         "@tf_runtime//:core_runtime_opdefs",
#         "@tf_runtime//:tensor_opdefs",
#     ],
# )
#
# tfrt_cc_library(
#     name = "arguments",
#     srcs = ["lib/arguments.cc"],
#     hdrs = ["include/tfrt/jitrt/arguments.h"],
#     compatible_with = ["//buildenv/target:gce"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "@llvm-project//llvm:Support",
#         "//third_party/tensorflow/compiler/xla/runtime:arguments",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#     ],
# )
#
# tfrt_cc_library(
#     name = "jitrt_compiler",
#     srcs = ["lib/jitrt_compiler.cc"],
#     hdrs = ["include/tfrt/jitrt/jitrt_compiler.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":codegen_transforms",
#         "@llvm-project//mlir:AMXToLLVMIRTranslation",
#         "@llvm-project//mlir:AffineDialect",
#         "@llvm-project//mlir:AffineToStandard",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:ArithTransforms",
#         "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
#         "@llvm-project//mlir:ArmSVEToLLVMIRTranslation",
#         "@llvm-project//mlir:AsyncDialect",
#         "@llvm-project//mlir:AsyncToLLVM",
#         "@llvm-project//mlir:AsyncTransforms",
#         "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
#         "@llvm-project//mlir:ComplexToLLVM",
#         "@llvm-project//mlir:ControlFlowDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:FuncToLLVM",
#         "@llvm-project//mlir:FuncTransforms",
#         "@llvm-project//mlir:LLVMCommonConversion",
#         "@llvm-project//mlir:LLVMToLLVMIRTranslation",
#         "@llvm-project//mlir:LinalgToLLVM",
#         "@llvm-project//mlir:LinalgTransforms",
#         "@llvm-project//mlir:MathDialect",
#         "@llvm-project//mlir:MathToLLVM",
#         "@llvm-project//mlir:MathToLibm",
#         "@llvm-project//mlir:MemRefToLLVM",
#         "@llvm-project//mlir:MemRefTransforms",
#         "@llvm-project//mlir:Pass",
#         "@llvm-project//mlir:ReconcileUnrealizedCasts",
#         "@llvm-project//mlir:SCFToControlFlow",
#         "@llvm-project//mlir:TensorInferTypeOpInterfaceImpl",
#         "@llvm-project//mlir:ToLLVMIRTranslation",
#         "@llvm-project//mlir:Transforms",
#         "@llvm-project//mlir:VectorToLLVM",
#         "@llvm-project//mlir:X86VectorToLLVMIRTranslation",
#         "//third_party/tensorflow/compiler/xla/mlir/math/transforms:passes",
#         "//third_party/tensorflow/compiler/xla/mlir/memref/transforms:passes",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/ir:rt",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:passes",
#         "//third_party/tensorflow/compiler/xla/runtime:compiler",
#     ],
#     alwayslink = 1,
# )
#
# tfrt_cc_library(
#     name = "async_task_runner",
#     hdrs = ["include/tfrt/jitrt/async_task_runner.h"],
#     # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "//third_party/tensorflow/compiler/xla/runtime:async_runtime",
#         "@tf_runtime//:hostcontext",
#     ],
# )
#
# tfrt_cc_library(
#     name = "results",
#     srcs = ["lib/results.cc"],
#     hdrs = ["include/tfrt/jitrt/results.h"],
#     # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:mlir_c_runner_utils",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/utils:async_runtime_api",
#         "//third_party/tensorflow/compiler/xla/runtime:async_runtime",
#         "//third_party/tensorflow/compiler/xla/runtime:results",
#         "//third_party/tensorflow/compiler/xla/runtime:types",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#     ],
# )
#
# tfrt_cc_library(
#     name = "custom_call_testlib",
#     srcs = ["lib/custom_calls/custom_call_testlib.cc"],
#     hdrs = ["include/tfrt/jitrt/custom_calls/custom_call_testlib.h"],
#     compatible_with = ["//buildenv/target:gce"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":arguments",
#         "@llvm-project//llvm:OrcJIT",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/ir/tests:testlib",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
#         "//third_party/tensorflow/compiler/xla/runtime:custom_call",
#         "//third_party/tensorflow/compiler/xla/runtime:custom_call_registry",
#         "//third_party/tensorflow/compiler/xla/runtime:executable",
#         "//third_party/tensorflow/compiler/xla/runtime:type_id",
#         "@tf_runtime//:support",
#         "@tf_runtime//third_party/llvm_derived:raw_ostream",
#     ],
# )
#
# tfrt_cc_library(
#     name = "jitrt_kernels",
#     srcs = ["lib/kernels/jitrt_kernels.cc"],
#     alwayslink_static_registration_src = "lib/kernels/jitrt_kernels_registration.cc",
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":arguments",
#         ":async_task_runner",
#         ":custom_call_testlib",
#         ":jitrt_compiler",
#         ":results",
#         "//third_party/tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
#         "//third_party/tensorflow/compiler/xla/runtime:arguments",
#         "//third_party/tensorflow/compiler/xla/runtime:async_runtime",
#         "//third_party/tensorflow/compiler/xla/runtime:compiler",
#         "//third_party/tensorflow/compiler/xla/runtime:custom_call",
#         "//third_party/tensorflow/compiler/xla/runtime:custom_call_registry",
#         "//third_party/tensorflow/compiler/xla/runtime:diagnostics",
#         "//third_party/tensorflow/compiler/xla/runtime:executable",
#         "//third_party/tensorflow/compiler/xla/runtime:jit_executable",
#         "//third_party/tensorflow/compiler/xla/runtime:types",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#     ],
# )
#
# tfrt_cc_library(
#     name = "jitrt_corert_kernels",
#     srcs = ["lib/kernels/jitrt_corert_kernels.cc"],
#     alwayslink_static_registration_src = "lib/kernels/jitrt_corert_kernels_registration.cc",
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":arguments",
#         ":async_task_runner",
#         ":jitrt_compiler",
#         ":results",
#         "//third_party/tensorflow/compiler/xla/runtime:arguments",
#         "//third_party/tensorflow/compiler/xla/runtime:async_runtime",
#         "//third_party/tensorflow/compiler/xla/runtime:custom_call",
#         "//third_party/tensorflow/compiler/xla/runtime:diagnostics",
#         "//third_party/tensorflow/compiler/xla/runtime:executable",
#         "//third_party/tensorflow/compiler/xla/runtime:jit_executable",
#         "//third_party/tensorflow/compiler/xla/runtime:types",
#         "@tf_runtime//:core_runtime",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#     ],
# )
#
# # Note: run_lit.sh does not exist in open-source environment.
# exports_files(["tools/run_lit.sh"])
#
# gentbl_cc_library(
#     name = "codegen_transforms_passes_inc_gen",
#     compatible_with = ["//buildenv/target:non_prod"],
#     tbl_outs = [
#         (
#             [
#                 "-gen-pass-decls",
#                 "-name=CodegenTransforms",
#             ],
#             "include/tfrt/jitrt/transforms/codegen_gen_passes.h.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "include/tfrt/jitrt/transforms/codegen_passes.td",
#     deps = ["@llvm-project//mlir:PassBaseTdFiles"],
# )
#
# tfrt_cc_library(
#     name = "codegen_transforms",
#     srcs = ["lib/transforms/cost_driven_async_parallel_for.cc"],
#     hdrs = ["include/tfrt/jitrt/transforms/codegen_passes.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     deps = [
#         ":codegen_transforms_passes_inc_gen",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:AsyncDialect",
#         "@llvm-project//mlir:AsyncTransforms",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MathDialect",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Pass",
#         "@llvm-project//mlir:SCFDialect",
#         "@llvm-project//mlir:Transforms",
#         "@llvm-project//mlir:VectorDialect",
#     ],
# )
#
# copybara:uncomment_end
