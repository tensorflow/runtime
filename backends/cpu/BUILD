load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@tf_runtime//:build_defs.bzl", "if_google", "tfrt_cc_library")
# copybara:uncomment load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

exports_files(["LICENSE"])

tfrt_cc_library(
    name = "core_runtime",
    srcs = [
        "lib/core_runtime/cpu_op_handler.cc",
        "lib/core_runtime/cpu_op_registry.cc",
        "lib/core_runtime/cpu_op_registry_impl.h",
        "lib/core_runtime/null_op_handler.cc",
        "lib/core_runtime/op_handler_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/cpu/core_runtime/cpu_op_handler.h",
        "include/tfrt/cpu/core_runtime/cpu_op_registry.h",
        "include/tfrt/cpu/core_runtime/null_op_handler.h",
        "lib/core_runtime/op_handler_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/core_runtime/static_registration.cc",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "test_ops",
    srcs = [
        "lib/ops/test/btf_kernels.cc",
        "lib/ops/test/coo_host_tensor_kernels.cc",
        "lib/ops/test/example_ops.cc",
        "lib/ops/test/mnist_tensor_kernels.cc",
        "lib/ops/test/resnet_tensor_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/cpu/ops/test/cpu_ops_and_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/ops/test/static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":core_runtime",
        ":cpu_kernels",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:test_metadata_functions",
        "@tf_runtime//backends/common:tf_metadata_functions",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_library(
    name = "async_runtime",
    srcs = [
        "lib/jit/async_runtime.cc",
    ],
    hdrs = [
        "include/tfrt/cpu/jit/async_runtime.h",
    ],
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
        "lib/jit/async_runtime_api.cc",
    ],
    hdrs = [
        "include/tfrt/cpu/jit/async_runtime_api.h",
    ],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":async_runtime",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_library(
    name = "buffer_forwarding",
    srcs = ["lib/ops/tf/buffer_forwarding.cc"],
    hdrs = ["lib/ops/tf/buffer_forwarding.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "type_dispatch",
    hdrs = ["lib/ops/tf/type_dispatch.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//backends/common:eigencompat",
    ],
)

tfrt_cc_library(
    name = "tf_ops",
    srcs = [
        "lib/ops/tf/concat_op.cc",
        "lib/ops/tf/concat_op.h",
        "lib/ops/tf/constant_ops.cc",
        "lib/ops/tf/constant_ops.h",
        "lib/ops/tf/cpu_ops.cc",
        "lib/ops/tf/cwise_binary_ops.cc",
        "lib/ops/tf/cwise_binary_ops.h",
        "lib/ops/tf/cwise_unary_ops.cc",
        "lib/ops/tf/cwise_unary_ops.h",
        "lib/ops/tf/matmul_fusion_ops.cc",
        "lib/ops/tf/matmul_fusion_ops.h",
        "lib/ops/tf/matmul_ops.cc",
        "lib/ops/tf/matmul_ops.h",
        "lib/ops/tf/shape_ops.cc",
        "lib/ops/tf/shape_ops.h",
        "lib/ops/tf/softmax_ops.cc",
        "lib/ops/tf/softmax_ops.h",
        "lib/ops/tf/tile_op.cc",
        "lib/ops/tf/tile_op.h",
    ],
    hdrs = [
        "include/tfrt/cpu/ops/tf/cpu_ops.h",
    ],
    alwayslink_static_registration_src = "lib/ops/tf/static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":buffer_forwarding",
        ":core_runtime",
        ":cpu_kernels",
        ":type_dispatch",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_metadata_functions",
    ],
)

tfrt_cc_library(
    name = "cpu_kernels",
    srcs = [
        "lib/kernels/tf/concat_kernels.cc",
        "lib/kernels/tf/const_kernels.cc",
        "lib/kernels/tf/cwise_binary_kernels.cc",
        "lib/kernels/tf/cwise_unary_kernels.cc",
        "lib/kernels/tf/fused_matmul_kernels.cc",
        "lib/kernels/tf/softmax_kernels.cc",
        "lib/kernels/tf/tile_kernels.cc",
        "lib/kernels/tile_kernel.cc",
    ],
    hdrs = [
        "lib/kernels/concat_kernel.h",
        "lib/kernels/cpu_kernels.h",
        "lib/kernels/cwise_binary_kernels.h",
        "lib/kernels/cwise_unary_kernels.h",
        "lib/kernels/fused_matmul_kernel.h",
        "lib/kernels/matmul_kernel.h",
        "lib/kernels/softmax_kernel.h",
        "lib/kernels/tile_kernel.h",
    ],
    alwayslink_static_registration_src = "lib/kernels/tf/static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@dnnl//:dnnl_single_threaded",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_bcast",
    ],
)

filegroup(
    name = "cpurt_ops_td_files",
    srcs = [
        "include/tfrt/cpu/jit/opdefs/cpurt_base.td",
        "include/tfrt/cpu/jit/opdefs/cpurt_ops.td",
        "@tf_runtime//:OpBaseTdFiles",
        "@tf_runtime//:include/tfrt/core_runtime/opdefs/corert_base.td",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
)

gentbl_cc_library(
    name = "cpurt_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = [
        "../../include",
        "include",
    ],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/cpu/jit/opdefs/cpurt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/cpu/jit/opdefs/cpurt_ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/cpu/jit/opdefs/cpurt_ops.td",
    td_srcs = [":cpurt_ops_td_files"],
)

tfrt_cc_library(
    name = "cpurt_opdefs",
    srcs = ["lib/jit/opdefs/cpurt_ops.cc"],
    hdrs = ["include/tfrt/cpu/jit/opdefs/cpurt_ops.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":cpurt_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

tfrt_cc_library(
    name = "cpurt",
    srcs = ["lib/jit/cpurt.cc"],
    hdrs = ["include/tfrt/cpu/jit/cpurt.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":async_runtime",
        ":async_runtime_api",
        ":cpurt_support",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AMXToLLVMIRTranslation",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
        "@llvm-project//mlir:ArmSVEToLLVMIRTranslation",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:AsyncToLLVM",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MathTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MlirJitRunner",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:X86VectorToLLVMIRTranslation",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "cpurt_support",
    hdrs = ["include/tfrt/cpu/jit/cpurt_support.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "cpurt_kernels",
    srcs = ["lib/jit/cpurt_kernels.cc"],
    alwayslink_static_registration_src = "lib/jit/cpurt_kernels_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":cpurt",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "cpurt_corert_kernels",
    srcs = ["lib/jit/cpurt_corert_kernels.cc"],
    alwayslink_static_registration_src = "lib/jit/cpurt_corert_kernels_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":cpurt",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

# copybara:uncomment_begin
# # temporarily remove. TODO(donglin): enable proto and image to OSS.
# tfrt_cc_library(
#     name = "proto",
#     srcs = [
#         "lib/kernels/proto/proto_kernels.cc",
#     ],
#     alwayslink_static_registration_src = "lib/kernels/proto/static_registration.cc",
#     defines = ["GOOGLE_PROTOBUF_NO_RTTI=1"],
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         ":lib_cc_proto",
#         "@llvm-project//llvm:Support",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tracing",
#     ],
# )
#
# cc_proto_library(
#     name = "lib_cc_proto",
#     deps = [":lib_proto"],
# )
#
# proto_library(
#     name = "lib_proto",
#     srcs = ["include/tfrt/cpu/kernels/proto/example.proto"],
#     cc_api_version = 2,
# )
#
# tfrt_cc_library(
#     name = "image",
#     srcs = [
#         "lib/kernels/image/image_kernels.cc",
#         "lib/kernels/image/jpeg/jpeg_handle.cc",
#         "lib/kernels/image/jpeg/jpeg_handle.h",
#         "lib/kernels/image/jpeg/jpeg_mem.cc",
#         "lib/kernels/image/jpeg/jpeg_mem.h",
#         "lib/kernels/image/resize_bilinear_op.cc",
#         "lib/kernels/image/resize_bilinear_op.h",
#     ],
#     alwayslink_static_registration_src = "lib/kernels/image/static_registration.cc",
#     visibility = ["@tf_runtime//:friends"],
#     deps = [
#         "//third_party/libjpeg_turbo:jpeg",
#         "@llvm-project//llvm:Support",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//:hostcontext",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#         "@tf_runtime//:tracing",
#     ],
# )
# copybara:uncomment_end
