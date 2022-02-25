load(
    "@tf_runtime//:build_defs.bzl",
    "if_google",
    "if_oss",
    "tfrt_cc_library",
)

package(
    # copybara:uncomment default_compatible_with = ["//buildenv/target:non_prod"],
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

# Add "--define disable_eigen_mkldnn_contraction_kernel=true" to your build
# command to disable MKL-DNN sgemm in Eigen tensor contractions (matrix
# multiplications and convolutions). The MKL-DNN kernels are generated at
# runtime and use avx/avx2/fma/avx512 based on cpu status registers
# (https://en.wikipedia.org/wiki/CPUID).
#
# When MKL-DNN contraction kernel is disabled, default kernel is:
#   Eigen::internal::gebp_kernel (general block-panel kernel).
config_setting(
    name = "disable_eigen_mkldnn_contraction_kernel",
    define_values = {"disable_eigen_mkldnn_contraction_kernel": "true"},
    visibility = ["//visibility:public"],
)

tfrt_cc_library(
    name = "tf_metadata_functions",
    srcs = ["lib/ops/tf/metadata_functions.cc"],
    hdrs = [
        "include/tfrt/common/ops/tf/metadata_functions.h",
    ],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":eigencompat",
        ":tf_bcast",
        ":tf_dnn_ops_util",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "test_metadata_functions",
    srcs = [
        "lib/ops/test/metadata/test_ops.cc",
    ],
    hdrs = [
        "include/tfrt/common/ops/test/metadata_functions.h",
    ],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":eigencompat",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "tf_dnn_ops_util",
    srcs = ["lib/ops/tf/dnn_ops_util.cc"],
    hdrs = ["include/tfrt/common/ops/tf/dnn_ops_util.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:shape_functions",
    ],
)

tfrt_cc_library(
    name = "tf_bcast",
    srcs = ["lib/ops/tf/bcast.cc"],
    hdrs = ["include/tfrt/common/ops/tf/bcast.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "eigentype",
    hdrs = ["include/tfrt/common/compat/eigen/eigen_dtype.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@eigen_archive//:eigen3",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
    ] + if_google([
        # TODO(b/161569340): Short-term fix. Remove.
        "//third_party/tensorflow/core/platform:types",
        "//third_party/tensorflow/core/platform:mutex",
    ]),
)

# Depending on a build configuration Eigen kernels library might use different
# contraction kernel (small matrix multiplication kernel used to multiply
# together blocks of the original tensors) implementation.
#
# 1) Default:
#    Use MKL-DNN single threaded sgemm. The MKL-DNN kernels are generated at
#    runtime and use avx/avx2/fma/avx512 based on cpu status registers
#    (https://en.wikipedia.org/wiki/CPUID).
#
# 2) Eigen: --define disable_eigen_mkldnn_contraction_kernel=true (no mkldnn)
#    Use Eigen contraction kernel: Eigen::internal::gebp_kernel.
#
# All kernels that use `tensor.contract(other_tensor)` must include
# `contraction_kernel.h` header.
tfrt_cc_library(
    name = "eigencompat",
    srcs = [
        "lib/compat/eigen/contraction_kernel.cc",
    ],
    hdrs = [
        "include/tfrt/common/compat/eigen/contraction_kernel.h",
        "include/tfrt/common/compat/eigen/contraction_output_kernel.h",
        "include/tfrt/common/compat/eigen/eigen_dtype.h",
        "include/tfrt/common/compat/eigen/eigen_evaluator.h",
        "include/tfrt/common/compat/eigen/eigen_kernel.h",
        "include/tfrt/common/compat/eigen/partial_packets.h",
        "include/tfrt/common/compat/eigen/spatial_convolution.h",
        "include/tfrt/common/compat/eigen/spatial_convolution_data_mapper.h",
        "include/tfrt/common/compat/eigen/tensor_types.h",
        "include/tfrt/common/compat/eigen/thread_pool_device.h",
    ],
    defines = if_oss([
        # TODO(b/161569340): Short-term fix. Remove.
        "EIGEN_MUTEX=std::mutex",
        "EIGEN_MUTEX_LOCK=std::unique_lock<std::mutex>",
        "EIGEN_CONDVAR=std::condition_variable",
        "EIGEN_AVOID_STL_ARRAY",
    ]) + select({
        ":disable_eigen_mkldnn_contraction_kernel": [],
        "@tf_runtime//:linux_x86_64": [
            # Custom contraction kernel defines.
            "TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL",
            "TFRT_EIGEN_USE_MKLDNN_CONTRACTION_KERNEL",
        ],
        "//conditions:default": [],
    }),
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
    ] + if_google([
        # TODO(b/161569340): Short-term fix. Remove.
        "//third_party/tensorflow/core/platform:types",
        "//third_party/tensorflow/core/platform:mutex",
    ]) + select({
        ":disable_eigen_mkldnn_contraction_kernel": [],
        "@tf_runtime//:linux_x86_64": [
            # Custom contraction kernel dependencies.
            "@dnnl//:dnnl_single_threaded",
        ],
        "//conditions:default": [],
    }),
)

tfrt_cc_library(
    name = "shape_functions",
    srcs = ["lib/compat/eigen/kernels/shape_functions.cc"],
    hdrs = ["include/tfrt/common/compat/eigen/kernels/shape_functions.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "eigen_kernels",
    srcs = [
        "lib/compat/eigen/kernels/batch_norm_grad.cc",
        "lib/compat/eigen/kernels/conv2d_grad_filter.cc",
        "lib/compat/eigen/kernels/conv2d_grad_input.cc",
        "lib/compat/eigen/kernels/conv2d_shape_functions.cc",
        "lib/compat/eigen/kernels/cpu_kernels.cc",
        "lib/compat/eigen/kernels/matmul.cc",
        "lib/compat/eigen/kernels/zero_padding.h",
    ],
    hdrs = [
        # These headers are needed by the `eigen_ops` bazel target. Move them
        # to include/ if they are needed by other bazel targets.
        "lib/compat/eigen/kernels/conv2d_shape_functions.h",
        "lib/compat/eigen/kernels/batch_norm.h",
        "lib/compat/eigen/kernels/conv2d.h",
        "lib/compat/eigen/kernels/max_pooling.h",
        "lib/compat/eigen/kernels/zero_padding.h",
    ],
    alwayslink_static_registration_src = "lib/compat/eigen/kernels/static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":eigencompat",
        ":shape_functions",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "eigen_ops",
    srcs = [
        "lib/compat/eigen/ops/cpu_ops.cc",
    ],
    alwayslink_static_registration_src = "lib/compat/eigen/ops/static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":eigen_kernels",
        ":eigencompat",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
    ],
)
