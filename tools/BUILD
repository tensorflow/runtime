load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_binary", "tfrt_cc_library")

package(
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

package_group(
    name = "friends",
    packages = [
        "//...",
    ],
)

exports_files([
    "LICENSE",
])

tfrt_cc_library(
    name = "opt_lib",
    visibility = [":friends"],
    deps = [
        "@llvm-project//llvm:Support",
    ] + select({
        # TODO(chky): Create a separate opt tool for gpu.
        "@tf_runtime//:gpu_enabled": [
            "@tf_runtime//backends/gpu:cuda_opdefs_alwayslink",
            "@tf_runtime//backends/gpu:cuda_test_opdefs_alwayslink",
        ],
        "//conditions:default": [],
    }),
)

tfrt_cc_binary(
    name = "tfrt_opt",
    srcs = ["tfrt_opt/tfrt_opt.cc"],
    visibility = [":friends"],
    deps = [
        ":opt_lib",
        "@llvm-project//mlir:MlirOptLib",
        "@tf_runtime//:init_tfrt_dialects",
    ],
)

tfrt_cc_library(
    name = "translate_lib",
    visibility = [":friends"],
    deps = [
        "@tf_runtime//:basic_kernels_opdefs_alwayslink",
        "@tf_runtime//:distributed_kernels_opdefs_alwayslink",
        "@tf_runtime//:beftomlir_alwayslink",
        "@tf_runtime//:core_runtime_opdefs_alwayslink",
        "@tf_runtime//:data_opdefs_alwayslink",
        "@tf_runtime//:mlirtobef_alwayslink",
        "@tf_runtime//:tensor_opdefs_alwayslink",
        "@tf_runtime//:test_kernels_opdefs_alwayslink",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:tfrt_translate_main",
    ] + select({
        "@tf_runtime//:gpu_enabled": [
            "@tf_runtime//backends/gpu:cuda_opdefs_alwayslink",
            "@tf_runtime//backends/gpu:cuda_test_opdefs_alwayslink",
        ],
        "//conditions:default": [],
    }),
)

tfrt_cc_binary(
    name = "tfrt_translate",
    visibility = [":friends"],
    deps = [
        ":translate_lib",
    ],
)

tfrt_cc_binary(
    name = "btf_info",
    srcs = ["btf_info_tool/main.cc"],
    visibility = [":friends"],
    deps = [
        ":opt_lib",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "bef_executor_lib",
    srcs = [
        "bef_executor/main.cc",
    ],
    visibility = [":friends"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:bef_executor_driver",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:tracing",
    ],
)

# Op/Kernel dependencies belong here rather than directly adding them in
# tools:bef_executor. This ensures other bef_executor binaries can easily use
# the same kernels as tools:bef_executor
tfrt_cc_library(
    name = "bef_executor_kernels",
    visibility = [":friends"],
    deps = [
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:data_alwayslink",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//:test_kernels_alwayslink",
        "@tf_runtime//backends/common:eigen_kernels_alwayslink",
        "@tf_runtime//backends/common:eigen_ops_alwayslink",
        "@tf_runtime//backends/cpu:cpu_kernels_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:test_ops_alwayslink",
        "@tf_runtime//backends/cpu:tf_ops_alwayslink",
    ] + select({
        "@tf_runtime//:gpu_enabled": [
            "@tf_runtime//backends/gpu:cuda_alwayslink",
            "@tf_runtime//backends/gpu:cuda_test_alwayslink",
            "@tf_runtime//backends/gpu:gpu_op_handler_alwayslink",
            "@tf_runtime//backends/gpu:gpu_test_ops_alwayslink",
            "@tf_runtime//backends/gpu:gpu_tf_ops_alwayslink",
        ],
        "//conditions:default": [],
    }),
)

tfrt_cc_binary(
    name = "bef_executor",
    visibility = [":friends"],
    deps = [
        ":bef_executor_kernels",
        ":bef_executor_lib",
        "@tf_runtime//:dtype",
        "@tf_runtime//:simple_tracing_sink_alwayslink",
    ],
)

# bazel build -c opt --copt=-Os --define=disable_rtti_and_exceptions=true
#   --linkopt=-Wl,--strip-all --linkopt=-Wl,--icf=safe, --define=use_std_mutex=1, --copt=-DSTD_MUTEX
#   //code_size_test_driver
# Android
# bazel build --config=android_arm64 -c opt --copt=-Os
#   --define=disable_rtti_and_exceptions=true --linkopt=-Wl,--strip-all
#   --linkopt=-Wl,--icf=safe, --define=use_std_mutex=1, --copt=-DSTD_MUTEX
#   //code_size_test_driver
tfrt_cc_binary(
    name = "code_size_test_driver",
    srcs = [
        "code_size_test_app/main.cc",
    ],
    deps = [
        "@tf_runtime//:basic_kernels",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_binary(
    name = "bef_name",
    srcs = ["bef_name.cc"],
)

bzl_library(
    name = "mlir_to_bef_bzl",
    srcs = ["mlir_to_bef.bzl"],
    visibility = ["//visibility:private"],
)
