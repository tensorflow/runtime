load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@tf_runtime//:build_defs.bzl", "if_google", "tfrt_cc_binary", "tfrt_cc_library")

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

tfrt_cc_library(
    name = "async_runtime",
    srcs = [
        "lib/async_runtime.cc",
    ],
    hdrs = [
        "include/tfrt/jitrt/async_runtime.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_async_runtime_api",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + if_google([
        # TODO(b/161569340): Short-term fix. Remove.
        "//third_party/tensorflow/core/platform:types",
        "//third_party/tensorflow/core/platform:mutex",
    ]),
)

tfrt_cc_library(
    name = "async_runtime_api",
    srcs = [
        "lib/async_runtime_api.cc",
    ],
    hdrs = [
        "include/tfrt/jitrt/async_runtime_api.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":async_runtime",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

td_library(
    name = "jitrt_ops_td_files",
    srcs = [
        "include/tfrt/jitrt/opdefs/jitrt_base.td",
        "include/tfrt/jitrt/opdefs/jitrt_ops.td",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@tf_runtime//:CoreRTTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "jitrt_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=jitrt",
            ],
            "include/tfrt/jitrt/opdefs/jitrt_ops_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=jitrt",
            ],
            "include/tfrt/jitrt/opdefs/jitrt_ops_dialect.cpp.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/tfrt/jitrt/opdefs/jitrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/jitrt/opdefs/jitrt_ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/jitrt/opdefs/jitrt_ops.td",
    deps = [":jitrt_ops_td_files"],
)

tfrt_cc_library(
    name = "jitrt_opdefs",
    srcs = ["lib/opdefs/jitrt_ops.cc"],
    hdrs = ["include/tfrt/jitrt/opdefs/jitrt_ops.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":jitrt_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

tfrt_cc_library(
    name = "jitrt",
    srcs = ["lib/jitrt.cc"],
    hdrs = [
        "include/tfrt/jitrt/jitrt.h",
        "include/tfrt/jitrt/runtime.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":async_runtime",
        ":async_runtime_api",
        ":rt_transforms",
        ":support",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:MlirJitRunner",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//:tracing",
    ],
)

tfrt_cc_library(
    name = "jitrt_compiler",
    srcs = ["lib/jitrt_compiler.cc"],
    hdrs = ["include/tfrt/jitrt/jitrt_compiler.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":codegen_transforms",
        ":rt_conversion",
        ":rt_opdefs",
        ":rt_transforms",
        "@llvm-project//mlir:AMXToLLVMIRTranslation",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
        "@llvm-project//mlir:ArmSVEToLLVMIRTranslation",
        "@llvm-project//mlir:AsyncToLLVM",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MathToLibm",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:TensorInferTypeOpInterfaceImpl",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:X86VectorToLLVMIRTranslation",
    ],
)

tfrt_cc_library(
    name = "support",
    hdrs = ["include/tfrt/jitrt/support.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "jitrt_kernels",
    srcs = ["lib/jitrt_kernels.cc"],
    alwayslink_static_registration_src = "lib/jitrt_kernels_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":jitrt",
        ":jitrt_compiler",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "jitrt_corert_kernels",
    srcs = ["lib/jitrt_corert_kernels.cc"],
    alwayslink_static_registration_src = "lib/jitrt_corert_kernels_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":jitrt",
        ":jitrt_compiler",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_binary(
    name = "jitrt_opt",
    srcs = ["tools/jitrt_opt.cc"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":codegen_transforms",
        ":rt_conversion",
        ":rt_transforms",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:init_tfrt_dialects",
    ],
)

# Note: run_lit.sh does not exist in open-source environment.
exports_files(["tools/run_lit.sh"])

td_library(
    name = "rt_ops_td_files",
    srcs = [
        "include/tfrt/jitrt/opdefs/rt_base.td",
        "include/tfrt/jitrt/opdefs/rt_ops.td",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "rt_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=rt",
            ],
            "include/tfrt/jitrt/opdefs/rt_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=rt",
            ],
            "include/tfrt/jitrt/opdefs/rt_dialect.cpp.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/tfrt/jitrt/opdefs/rt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/jitrt/opdefs/rt_ops.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "include/tfrt/jitrt/opdefs/rt_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/tfrt/jitrt/opdefs/rt_types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/jitrt/opdefs/rt_ops.td",
    deps = [":rt_ops_td_files"],
)

gentbl_cc_library(
    name = "rt_conversion_passes_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=RuntimeConversion",
            ],
            "include/tfrt/jitrt/conversion/rt_gen_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/jitrt/conversion/rt_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "rt_transforms_passes_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=RuntimeTransforms",
            ],
            "include/tfrt/jitrt/transforms/rt_gen_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/jitrt/transforms/rt_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "codegen_transforms_passes_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CodegenTransforms",
            ],
            "include/tfrt/jitrt/transforms/codegen_gen_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/jitrt/transforms/codegen_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

tfrt_cc_library(
    name = "rt_opdefs",
    srcs = ["lib/opdefs/rt_ops.cc"],
    hdrs = ["include/tfrt/jitrt/opdefs/rt_ops.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":rt_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "rt_conversion",
    srcs = ["lib/conversion/rt_to_llvm.cc"],
    hdrs = ["include/tfrt/jitrt/conversion/rt_passes.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":rt_conversion_passes_inc_gen",
        ":rt_opdefs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

tfrt_cc_library(
    name = "rt_transforms",
    srcs = ["lib/transforms/convert_to_kernel_function.cc"],
    hdrs = ["include/tfrt/jitrt/transforms/rt_passes.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":rt_opdefs",
        ":rt_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "codegen_transforms",
    srcs = [
        "lib/transforms/aligned_allocations.cc",
        "lib/transforms/cost_driven_async_parallel_for.cc",
        "lib/transforms/math_optimization.cc",
    ],
    hdrs = ["include/tfrt/jitrt/transforms/codegen_passes.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":codegen_transforms_passes_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:X86Vector",
    ],
)
