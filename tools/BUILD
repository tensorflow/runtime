load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_binary", "tfrt_cc_library")

package(
    # copybara:uncomment_begin
    # default_visibility = ["@tf_runtime//:__subpackages__"],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//:__subpackages__"],
    # copybara:comment_end
)

licenses(["notice"])

package_group(
    name = "friends",
    packages = [
        "//...",
        # copybara:uncomment_begin
        # # Authorized users go here.
        # "//learning/brain/mobile/lite/tf_runtime/...",
        # "//third_party/tf_runtime_google/...",
        # "//third_party/tensorflow/compiler/mlir/tfrt/...",
        # "//third_party/tensorflow/core/runtime_fallback/...",
        # copybara:uncomment_end
    ],
)

tfrt_cc_binary(
    name = "tfrt_opt",
    srcs = ["tfrt_opt/tfrt_opt.cc"],
    visibility = [":friends"],
    deps = [
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:print_stream_pass",
    ],
)

tfrt_cc_library(
    name = "translate_lib",
    visibility = [":friends"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:beftomlir_translate_alwayslink",
        "@tf_runtime//:mlirtobef_translate_alwayslink",
        "@tf_runtime//third_party/llvm_derived:tfrt_translate_main",
    ],
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
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

# copybara:uncomment_begin
# py_test(
#     name = "btf_info_test",
#     srcs = ["btf_info_tool/btf_info_test.py"],
#     python_version = "PY3",
#     deps = [
#         ":btf_info",
#         "@tf_runtime//:btf_writer",
#     ],
# )
# copybara:uncomment_end

tfrt_cc_library(
    name = "bef_executor_lib",
    testonly = True,
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
# the same kernels as tools:bef_executor[_lite]
#
# This build target contains kernels that are lightweight to build.
tfrt_cc_library(
    name = "bef_executor_lightweight_kernels",
    visibility = [":friends"],
    deps = [
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:data_alwayslink",
        "@tf_runtime//:io_alwayslink",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//:test_kernels_alwayslink",
        "@tf_runtime//backends/cpu:cpu_kernels_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:test_ops_alwayslink",
    ] + select({
        "@tf_runtime//:gpu_enabled": [
            "@tf_runtime//backends/gpu:gpu_kernels_alwayslink",
            "@tf_runtime//backends/gpu:gpu_op_handler_alwayslink",
        ],
        "//conditions:default": [],
    }),
)

# This build target contains kernels that are expensive to build.
tfrt_cc_library(
    name = "bef_executor_expensive_kernels",
    visibility = [":friends"],
    deps = [
        "@tf_runtime//backends/common:eigen_kernels_alwayslink",
        "@tf_runtime//backends/common:eigen_ops_alwayslink",
        "@tf_runtime//backends/cpu:tf_ops_alwayslink",
    ] + select({
        "@tf_runtime//:gpu_enabled": [
            "@tf_runtime//backends/gpu:gpu_test_ops_alwayslink",
            "@tf_runtime//backends/gpu:gpu_tf_ops_alwayslink",
        ],
        "//conditions:default": [],
    }),
)

# Op/Kernel dependencies belong here rather than directly adding them in
# tools:bef_executor. This ensures other bef_executor binaries can easily use
# the same kernels as tools:bef_executor
tfrt_cc_library(
    name = "bef_executor_jit_kernels",
    visibility = [":friends"],
    deps = [
        "@tf_runtime//backends/cpu:cpurt_corert_kernels_alwayslink",
        "@tf_runtime//backends/cpu:cpurt_kernels_alwayslink",
    ],
)

tfrt_cc_binary(
    name = "bef_executor_lite",
    testonly = True,
    visibility = [":friends"],
    deps = [
        ":bef_executor_jit_kernels",
        ":bef_executor_lib",
        ":bef_executor_lightweight_kernels",
        "@tf_runtime//:dtype",
        "@tf_runtime//:simple_tracing_sink_alwayslink",
    ],
)

# This one contains eigen-based CPU kernels that are expensive to build.
tfrt_cc_binary(
    name = "bef_executor",
    testonly = True,
    visibility = [":friends"],
    deps = [
        ":bef_executor_expensive_kernels",
        ":bef_executor_jit_kernels",
        ":bef_executor_lib",
        ":bef_executor_lightweight_kernels",
        "@tf_runtime//:dtype",
        "@tf_runtime//:simple_tracing_sink_alwayslink",
    ],
)

# Command to build the code_size_test_driver manually:
# bazel build --config=code_size_test tools:code_size_test_driver
#
# In order to build for Android, add '--config=android_arm64'.
tfrt_cc_binary(
    name = "code_size_test_driver",
    srcs = [
        "code_size_test_app/main.cc",
    ],
    deps = [
        "@tf_runtime//:basic_kernels",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

bzl_library(
    name = "mlir_to_bef_bzl",
    srcs = ["mlir_to_bef.bzl"],
    visibility = ["//visibility:private"],
)

tfrt_cc_binary(
    name = "bef_executor_debug_tracing",
    testonly = True,
    deps = [
        ":bef_executor_jit_kernels",
        ":bef_executor_lib",
        ":bef_executor_lightweight_kernels",
        ":bef_executor_lite",
        "@tf_runtime//:debug_tracing_sink",
        "@tf_runtime//:dtype",
    ],
)
