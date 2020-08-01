load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

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
    hdrs = [
        "lib/kernels/concat_kernel.h",
        "lib/kernels/cpu_kernels.h",
        "lib/kernels/cwise_binary_kernels.h",
        "lib/kernels/cwise_unary_kernels.h",
        "lib/kernels/matmul_kernel.h",
        "lib/kernels/softmax_kernel.h",
        "lib/kernels/tile_kernel.h",
    ],
    deps = [
        "@mkl_dnn//:mkldnn_single_threaded",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_bcast",
    ],
)
