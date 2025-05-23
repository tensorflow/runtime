# copybara:uncomment load("//third_party/protobuf/bazel:proto_library.bzl", "proto_library")
load("@tf_runtime//:build_defs.bzl", "if_google", "tfrt_cc_library")
# copybara:uncomment load("//tools/build_defs/license:license.bzl", "license")
# copybara:uncomment load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    default_visibility = [":__subpackages__"],
)

# copybara:uncomment_begin(Internal license rules)
# license(
#     name = "license",
#     package_name = "cpu",
# )
# copybara:uncomment_end

licenses(["notice"])

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
    visibility = ["//visibility:public"],
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
    visibility = if_google(
        ["@tf_runtime//:friends"],
        ["//visibility:public"],
    ),
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
    visibility = if_google(
        ["@tf_runtime//:friends"],
        ["//visibility:public"],
    ),
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
        "lib/kernels/tile_kernel.cc",
    ],
    hdrs = [
        "lib/kernels/cpu_kernels.h",
        "lib/kernels/cwise_binary_kernels.h",
        "lib/kernels/cwise_unary_kernels.h",
        "lib/kernels/fused_matmul_kernel.h",
        "lib/kernels/matmul_kernel.h",
        "lib/kernels/softmax_kernel.h",
        "lib/kernels/tile_kernel.h",
    ],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_bcast",
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
