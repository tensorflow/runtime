load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_binary", "tfrt_cc_library")

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
#     compatible_with = ["//buildenv/target:non_prod"],
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
#     compatible_with = ["//buildenv/target:non_prod"],
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
#     compatible_with = ["//buildenv/target:non_prod"],
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
#     name = "memory_mapper",
#     srcs = ["lib/memory_mapper.cc"],
#     hdrs = ["include/tfrt/jitrt/memory_mapper.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "@llvm-project//llvm:ExecutionEngine",
#         "@llvm-project//llvm:Support",
#     ],
# )
#
# tfrt_cc_library(
#     name = "execution_engine",
#     srcs = ["lib/execution_engine.cc"],
#     hdrs = ["include/tfrt/jitrt/execution_engine.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "@llvm-project//llvm:ARMAsmParser",
#         "@llvm-project//llvm:ARMCodeGen",
#         "@llvm-project//llvm:ExecutionEngine",
#         "@llvm-project//llvm:OrcJIT",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//llvm:X86AsmParser",
#         "@llvm-project//llvm:X86CodeGen",
#         "@llvm-project//mlir:ExecutionEngineUtils",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:ToLLVMIRTranslation",
#         "@tf_runtime//:support",
#     ],
# )
#
# tfrt_cc_library(
#     name = "jitrt",
#     srcs = ["lib/jitrt.cc"],
#     hdrs = [
#         "include/tfrt/jitrt/jitrt.h",
#         "include/tfrt/jitrt/runtime.h",
#     ],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":async_values_cache",
#         ":custom_call",
#         ":custom_call_registry",
#         ":diagnostics",
#         ":execution_engine",
#         ":memory_mapper",
#         ":results",
#         ":xla",
#         "@llvm-project//llvm:Core",
#         "@llvm-project//llvm:OrcJIT",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:LLVMCommonConversion",
#         "@llvm-project//mlir:Parser",
#         "@llvm-project//mlir:Pass",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:Transforms",
#         "@llvm-project//mlir:mlir_c_runner_utils",
#         "//third_party/tensorflow/compiler/xla/mlir/transforms/runtime:passes",
#         "//third_party/tensorflow/compiler/xla/mlir/transforms/runtime:specialization",
#         "//third_party/tensorflow/compiler/xla/mlir/transforms/runtime:type_converter",
#         "//third_party/tensorflow/compiler/xla/mlir/utils/runtime:async_runtime_api",
#         "//third_party/tensorflow/compiler/xla/mlir/utils/runtime:constraints",
#         "//third_party/tensorflow/compiler/xla/runtime:arguments",
#         "//third_party/tensorflow/compiler/xla/runtime:async_runtime",
#         "//third_party/tensorflow/compiler/xla/runtime:constraints",
#         "//third_party/tensorflow/compiler/xla/runtime:symbolic_shape",
#         "//third_party/tensorflow/compiler/xla/runtime:types",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
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
#         ":custom_call_to_llvm",
#         ":rt_conversion",
#         "@llvm-project//mlir:AMXToLLVMIRTranslation",
#         "@llvm-project//mlir:AffineDialect",
#         "@llvm-project//mlir:AffineToStandard",
#         "@llvm-project//mlir:ArithmeticDialect",
#         "@llvm-project//mlir:ArithmeticTransforms",
#         "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
#         "@llvm-project//mlir:ArmSVEToLLVMIRTranslation",
#         "@llvm-project//mlir:AsyncDialect",
#         "@llvm-project//mlir:AsyncToLLVM",
#         "@llvm-project//mlir:AsyncTransforms",
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
#         "//third_party/tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
#         "//third_party/tensorflow/compiler/xla/mlir/transforms/math:passes",
#         "//third_party/tensorflow/compiler/xla/mlir/transforms/memref:passes",
#         "//third_party/tensorflow/compiler/xla/mlir/transforms/runtime:passes",
#     ],
#     alwayslink = 1,
# )
#
# tfrt_cc_library(
#     name = "results",
#     srcs = ["lib/results.cc"],
#     hdrs = ["include/tfrt/jitrt/results.h"],
#     # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":xla",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:mlir_c_runner_utils",
#         "//third_party/tensorflow/compiler/xla/mlir/utils/runtime:async_runtime_api",
#         "//third_party/tensorflow/compiler/xla/runtime:async_runtime",
#         "//third_party/tensorflow/compiler/xla/runtime:types",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#     ],
# )
#
# # Generates 'async_values_cache.h' based on the :std_thread flag.
# #
# # Internally we use more efficient AsyncValuesCache implemented on
# # top of the RCU primitive. In open source we rely on mutex for
# # synchronizing access to shared data strucures.
# #
# # This avoids a (non-transitive) copts setting to include one or the other
# # header file by the preprocessor.
# genrule(
#     name = "async_values_cache_hdr",
#     srcs = select({
#         "@tf_runtime//:use_std_thread": ["include/tfrt/jitrt/async_values_cache_std.h"],
#         "//conditions:default": ["include/tfrt/jitrt/async_values_cache_rcu.h"],
#     }),
#     outs = ["include/tfrt/jitrt/async_values_cache.h"],
#     cmd = "cp $< $@",
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
# )
#
# tfrt_cc_library(
#     name = "async_values_cache",
#     hdrs = ["include/tfrt/jitrt/async_values_cache.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "@llvm-project//llvm:Support",
#         "@tf_runtime//:hostcontext",
#     ] + select({
#         "@tf_runtime//:use_std_thread": [
#             "@tf_runtime//:support",
#         ],
#         "//conditions:default": [
#             "//concurrent/rcu:view",
#         ],
#     }),
# )
#
# tfrt_cc_library(
#     name = "custom_call",
#     srcs = ["lib/custom_call.cc"],
#     hdrs = ["include/tfrt/jitrt/custom_call.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":diagnostics",
#         "@eigen_archive//:eigen3",
#         "@llvm-project//llvm:Support",
#         "//third_party/tensorflow/compiler/xla/runtime:logical_result",
#         "//third_party/tensorflow/compiler/xla/runtime:type_id",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:support",
#     ],
# )
#
# tfrt_cc_library(
#     name = "custom_call_registry",
#     srcs = ["lib/custom_call_registry.cc"],
#     hdrs = ["include/tfrt/jitrt/custom_call_registry.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":custom_call",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:Support",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:support",
#     ],
# )
#
# tfrt_cc_library(
#     name = "diagnostics",
#     srcs = ["lib/diagnostics.cc"],
#     hdrs = ["include/tfrt/jitrt/diagnostics.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:Support",
#     ],
# )
#
# tfrt_cc_library(
#     name = "xla",
#     hdrs = ["include/tfrt/jitrt/xla.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
# )
#
# td_library(
#     name = "custom_call_testlib_td_files",
#     srcs = [
#         "include/tfrt/jitrt/custom_calls/custom_call_testlib.td",
#         "include/tfrt/jitrt/custom_calls/custom_call_testlib_attrs.td",
#         "include/tfrt/jitrt/custom_calls/custom_call_testlib_enums.td",
#         "include/tfrt/jitrt/custom_calls/custom_call_testlib_types.td",
#     ],
#     compatible_with = ["//buildenv/target:non_prod"],
#     includes = ["include"],
#     deps = [
#         "@llvm-project//mlir:OpBaseTdFiles",
#     ],
# )
#
# gentbl_cc_library(
#     name = "custom_call_testlib_inc_gen",
#     compatible_with = ["//buildenv/target:non_prod"],
#     tbl_outs = [
#         (
#             ["-gen-attrdef-decls"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_attrs.h.inc",
#         ),
#         (
#             ["-gen-attrdef-defs"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_attrs.cc.inc",
#         ),
#         (
#             ["-gen-dialect-decls"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_dialect.h.inc",
#         ),
#         (
#             ["-gen-dialect-defs"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_dialect.cc.inc",
#         ),
#         (
#             ["-gen-enum-decls"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_enums.h.inc",
#         ),
#         (
#             ["-gen-enum-defs"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_enums.cc.inc",
#         ),
#         (
#             ["-gen-typedef-decls"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_types.h.inc",
#         ),
#         (
#             ["-gen-typedef-defs"],
#             "include/tfrt/jitrt/custom_calls/custom_call_testlib_types.cc.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "include/tfrt/jitrt/custom_calls/custom_call_testlib.td",
#     deps = [":custom_call_testlib_td_files"],
# )
#
# tfrt_cc_library(
#     name = "custom_call_testlib",
#     srcs = ["lib/custom_calls/custom_call_testlib.cc"],
#     hdrs = ["include/tfrt/jitrt/custom_calls/custom_call_testlib.h"],
#     alwayslink_static_registration_src = "lib/custom_calls/custom_call_testlib_registration.cc",
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":custom_call",
#         ":custom_call_registry",
#         ":custom_call_testlib_inc_gen",
#         ":custom_call_to_llvm",
#         ":jitrt",
#         "@llvm-project//llvm:OrcJIT",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
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
#         ":custom_call_testlib",
#         ":jitrt",
#         ":jitrt_compiler",
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
#         ":jitrt",
#         ":jitrt_compiler",
#         "@tf_runtime//:core_runtime",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#     ],
# )
#
# tfrt_cc_binary(
#     name = "jitrt_opt",
#     srcs = ["tools/jitrt_opt.cc"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":codegen_transforms",
#         ":custom_call_testlib",
#         ":jitrt_compiler",
#         ":rt_conversion",
#         "@llvm-project//mlir:MlirOptLib",
#         "@llvm-project//mlir:Transforms",
#         "//third_party/tensorflow/compiler/xla/mlir/transforms/runtime:passes",
#         "@tf_runtime//:init_tfrt_dialects",
#     ],
# )
#
# # Note: run_lit.sh does not exist in open-source environment.
# exports_files(["tools/run_lit.sh"])
#
# gentbl_cc_library(
#     name = "rt_conversion_passes_inc_gen",
#     compatible_with = ["//buildenv/target:non_prod"],
#     tbl_outs = [
#         (
#             [
#                 "-gen-pass-decls",
#                 "-name=RuntimeConversion",
#             ],
#             "include/tfrt/jitrt/conversion/rt_gen_passes.h.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "include/tfrt/jitrt/conversion/rt_passes.td",
#     deps = ["@llvm-project//mlir:PassBaseTdFiles"],
# )
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
#     name = "custom_call_to_llvm",
#     srcs = ["lib/conversion/custom_call_to_llvm.cc"],
#     hdrs = ["include/tfrt/jitrt/conversion/custom_call_to_llvm.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":custom_call",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithmeticDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:LLVMCommonConversion",
#         "@llvm-project//mlir:LLVMDialect",
#         "@llvm-project//mlir:Support",
#     ],
# )
#
# tfrt_cc_library(
#     name = "rt_conversion",
#     srcs = ["lib/conversion/rt_to_llvm.cc"],
#     hdrs = ["include/tfrt/jitrt/conversion/rt_passes.h"],
#     compatible_with = ["//buildenv/target:non_prod"],
#     deps = [
#         ":custom_call",
#         ":custom_call_to_llvm",
#         ":rt_conversion_passes_inc_gen",
#         ":xla",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithmeticDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:FuncToLLVM",
#         "@llvm-project//mlir:FuncTransforms",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:LLVMCommonConversion",
#         "@llvm-project//mlir:LLVMDialect",
#         "@llvm-project//mlir:Pass",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:Transforms",
#         "//third_party/tensorflow/compiler/xla/mlir/ir/runtime:rt_ops",
#     ],
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
#         "@llvm-project//mlir:ArithmeticDialect",
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
