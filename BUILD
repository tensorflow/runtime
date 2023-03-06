load(":build_defs.bzl", "if_google", "if_oss", "make_variable", "tfrt_cc_library")

# copybara:uncomment load("//configlang/ncl/build_defs:ncl.bzl", "ncl_test")
# copybara:uncomment load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [":__subpackages__"],
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

config_setting(
    name = "windows",
    # Internal builds query the target OS.
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "windows"},
        {},
    ),
    # OSS builds query the CPU type.
    values = if_oss(
        {"cpu": "x64_windows"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_k8",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_haswell",
    values = {"cpu": "haswell"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "linux_x86_64",
    match_any = [
        ":linux_k8",
        ":linux_haswell",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin
# selects.config_setting_group(
#     name = "linux_x86_64-google",
#     match_all = [
#         # keep sorted
#         ":linux_x86_64",
#         "//tools/cc_target_os:linux-google",
#     ],
# )
# copybara:uncomment_end

# Add "--@tf_runtime//:eigen_mkldnn_contraction_kernel=false" to your build
# command to disable MKL-DNN sgemm in Eigen tensor contractions (matrix
# multiplications and convolutions). The MKL-DNN kernels are generated at
# runtime and use avx/avx2/fma/avx512 based on cpu status registers
# (https://en.wikipedia.org/wiki/CPUID).
#
# When MKL-DNN contraction kernel is disabled, default kernel is:
#   Eigen::internal::gebp_kernel (general block-panel kernel).
bool_flag(
    name = "eigen_mkldnn_contraction_kernel",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

# Flag to build tf_runtime with std::thread/mutex instead of ABSL's:
# bazel build --@tf_runtime//:std_thread
# This is the default and only valid option in open-source.
bool_flag(
    name = "std_thread",
    # copybara:uncomment_begin
    # build_setting_default = False,
    # copybara:uncomment_end_and_comment_begin
    build_setting_default = True,
    # copybara:comment_end
)

# Setting whether to use std::thread/mutex instead of ABSL's.
config_setting(
    name = "use_std_thread",
    flag_values = {":std_thread": "True"},
)

# To build tf_runtime without RTTI/exceptions, use:
# bazel build --no@tf_runtime//:rtti_and_exceptions
bool_flag(
    name = "rtti_and_exceptions",
    build_setting_default = True,
    visibility = ["//visibility:private"],
)

config_setting(
    name = "disable_rtti_and_exceptions",
    flag_values = {":rtti_and_exceptions": "False"},
    visibility = ["//visibility:public"],
)

# To build tf_runtime with GPU backend, use:
# bazel build --@tf_runtime//:enable_gpu
bool_flag(
    name = "enable_gpu",
    build_setting_default = False,
    visibility = ["//visibility:private"],
)

config_setting(
    name = "gpu_enabled_oss",
    flag_values = {":enable_gpu": "True"},
    visibility = ["//visibility:private"],
)

# Config setting to conditionally link GPU targets.
alias(
    name = "gpu_enabled",
    actual = if_google(
        ":linux_x86_64-google",
        ":gpu_enabled_oss",
    ),
)

# copybara:uncomment_begin
# ncl_test(
#     name = "tf_host_runtime_blueprint_test",
#     srcs = ["tf_host_runtime.blueprint"],
#     data = [
#         ":bluze.textproto",
#         "//devtools/blueprint/bluze/public:bluze_ncl",
#         "//devtools/blueprint/ncl:sanitizer",
#     ],
#     deps = ["//devtools/bazel/subteams/configurability/android_platforms_migration:flags_ncl"],
# )
#
# filegroup(
#     name = "copybara_library",
#     srcs = glob(
#         [
#             "*.bara.sky",
#             "*.dic",
#         ],
#         exclude = [
#             "copy.bara.sky",
#         ],
#     ),
#     visibility = ["//visibility:public"],
# )
#
# filegroup(
#     name = "copybara_config",
#     srcs = [
#         "copy.bara.sky",
#         "scrub.patch",
#     ],
#     data = [
#         ":copybara_library",
#         "//learning/brain/testing/copybara:all_bara_sky",
#     ],
# )
#
# copybara_config_test(
#     name = "copybara_config_test",
#     config = "copy.bara.sky",
#     tags = [
#         "noasan",
#         "nodfsan",
#         "nogotsan",
#         "nogpu",
#         "nomsan",
#         "nosan",
#         "notsan",
#         "noubsan",
#     ],
#     deps = [":copybara_config"],
# )
# copybara:uncomment_end

tfrt_cc_library(
    name = "profiled_allocator",
    srcs = ["lib/host_context/profiled_allocator.cc"],
    hdrs = ["include/tfrt/host_context/profiled_allocator.h"],
    visibility = ["//visibility:public"],
    deps = [":hostcontext"],
)

tfrt_cc_library(
    name = "async_value",
    srcs = [
        "lib/concurrency/async_value.cc",
        "lib/concurrency/async_value_ref.cc",
    ],
    hdrs = [
        "include/tfrt/concurrency/async_value.h",
        "include/tfrt/concurrency/async_value_ref.h",
        "include/tfrt/concurrency/chain.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":concurrent_vector",
        ":ref_count",
        ":support",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

tfrt_cc_library(
    name = "concurrent_vector",
    hdrs = ["include/tfrt/concurrency/concurrent_vector.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

tfrt_cc_library(
    name = "ref_count",
    hdrs = ["include/tfrt/concurrency/ref_count.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
)

tfrt_cc_library(
    name = "hostcontext",
    srcs = [
        "lib/host_context/async_dispatch.cc",
        "lib/host_context/concurrent_work_queue.cc",
        "lib/host_context/device.cc",
        "lib/host_context/diagnostic.cc",
        "lib/host_context/execution_context.cc",
        "lib/host_context/host_allocator.cc",
        "lib/host_context/host_buffer.cc",
        "lib/host_context/host_context.cc",
        "lib/host_context/host_context_ptr.cc",
        "lib/host_context/kernel_frame.cc",
        "lib/host_context/kernel_registry.cc",
        "lib/host_context/location.cc",
        "lib/host_context/native_function.cc",
        "lib/host_context/parallel_for.cc",
        "lib/host_context/shared_context.cc",
        "lib/host_context/single_threaded_work_queue.cc",
        "lib/host_context/test_fixed_size_allocator.cc",
        "lib/host_context/timer_queue.cc",
        "@tf_runtime//third_party/concurrent_work_queue:concurrent_work_queue_hdrs",
        "@tf_runtime//third_party/concurrent_work_queue:concurrent_work_queue_srcs",
    ],
    hdrs = [
        "include/tfrt/host_context/async_dispatch.h",
        "include/tfrt/host_context/async_value.h",
        "include/tfrt/host_context/async_value_ref.h",
        "include/tfrt/host_context/attribute_utils.h",
        "include/tfrt/host_context/chain.h",
        "include/tfrt/host_context/concurrent_work_queue.h",
        "include/tfrt/host_context/device.h",
        "include/tfrt/host_context/diagnostic.h",
        "include/tfrt/host_context/execution_context.h",
        "include/tfrt/host_context/function.h",
        "include/tfrt/host_context/host_allocator.h",
        "include/tfrt/host_context/host_buffer.h",
        "include/tfrt/host_context/host_context.h",
        "include/tfrt/host_context/host_context_ptr.h",
        "include/tfrt/host_context/kernel_frame.h",
        "include/tfrt/host_context/kernel_registry.h",
        "include/tfrt/host_context/kernel_utils.h",
        "include/tfrt/host_context/location.h",
        "include/tfrt/host_context/native_function.h",
        "include/tfrt/host_context/parallel_for.h",
        "include/tfrt/host_context/request_deadline_tracker.h",
        "include/tfrt/host_context/resource_context.h",
        "include/tfrt/host_context/shared_context.h",
        "include/tfrt/host_context/sync_kernel_frame.h",
        "include/tfrt/host_context/sync_kernel_utils.h",
        "include/tfrt/host_context/task_function.h",
        "include/tfrt/host_context/timer_queue.h",
        "include/tfrt/host_context/type_name.h",
        "include/tfrt/host_context/value.h",
    ],
    alwayslink_static_registration_src = "lib/host_context/static_registration.cc",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":async_value",
        ":bef",
        ":support",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:unique_any",
    ],
)

tfrt_cc_library(
    name = "dtype",
    srcs = [
        "lib/dtype/dtype.cc",
    ],
    hdrs = [
        "include/tfrt/dtype/dtype.def",
        "include/tfrt/dtype/dtype.h",
        "include/tfrt/dtype/dtype_formatter.h",
        "include/tfrt/dtype/quantized_types.h",
        "include/tfrt/support/bf16.h",
        "include/tfrt/support/fp16.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "bef",
    hdrs = [
        "include/tfrt/bef/bef.h",
        "include/tfrt/bef/bef_buffer.h",
        "include/tfrt/bef/bef_encoding.h",
        "include/tfrt/bef/bef_reader.h",
        "include/tfrt/bef/kernel.h",
        "include/tfrt/bef/span.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":dtype",
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

# Generates 'mutex.h' and `thread_environment.h` based on the :std_thread flag.
# This avoids a (non-transitive) copts setting to include one or the other
# header file by the preprocessor.
[
    genrule(
        name = out_name,
        srcs = select({
            ":use_std_thread": ["include/tfrt/support/" + std_name],
            "//conditions:default": ["include/tfrt/support/" + absl_name],
        }),
        outs = ["include/tfrt/support/" + out_name],
        cmd = "cp $< $@",
        # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
        visibility = ["//visibility:private"],
    )
    for (out_name, absl_name, std_name) in [
        ("mutex.h", "absl_mutex.h", "std_mutex.h"),
        ("thread_environment.h", "thread_environment_google.h", "thread_environment_std.h"),
    ]
]

tfrt_cc_library(
    name = "support",
    srcs = [
        "lib/support/alloc.cc",
        "lib/support/crc32c.cc",
        "lib/support/crc32c_accelerate.cc",
        "lib/support/error_util.cc",
        "lib/support/hash_util.cc",
        "lib/support/logging.cc",
        "lib/support/random_util.cc",
        "lib/support/stack_trace.cc",
        "lib/support/string_util.cc",
    ],
    hdrs = [
        "include/tfrt/support/aligned_buffer.h",
        "include/tfrt/support/alloc.h",
        "include/tfrt/support/bf16.h",
        "include/tfrt/support/byte_order.h",
        "include/tfrt/support/concurrent_vector.h",
        "include/tfrt/support/crc32c.h",
        "include/tfrt/support/error_type.def",
        "include/tfrt/support/error_util.h",
        "include/tfrt/support/forward_decls.h",
        "include/tfrt/support/fp16.h",
        "include/tfrt/support/hash_util.h",
        "include/tfrt/support/latch.h",
        "include/tfrt/support/logging.h",
        "include/tfrt/support/map_by_type.h",
        "include/tfrt/support/msan.h",
        "include/tfrt/support/mutex.h",
        "include/tfrt/support/op_registry_impl.h",
        "include/tfrt/support/philox_random.h",
        "include/tfrt/support/pointer_util.h",
        "include/tfrt/support/random_util.h",
        "include/tfrt/support/ranges.h",
        "include/tfrt/support/ranges_util.h",
        "include/tfrt/support/raw_coding.h",
        "include/tfrt/support/rc_array.h",
        "include/tfrt/support/ref_count.h",
        "include/tfrt/support/refcounted_callback.h",
        "include/tfrt/support/string_util.h",
        "include/tfrt/support/template_util.h",
        "include/tfrt/support/thread_annotations.h",
        "include/tfrt/support/thread_environment.h",
        "include/tfrt/support/thread_local.h",
        "include/tfrt/support/type_id.h",
        "include/tfrt/support/type_traits.h",
        "include/tfrt/support/variant.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":concurrent_vector",
        ":ref_count",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:unique_any",
    ] + select({
        ":use_std_thread": [],
        "//conditions:default": [
            # copybara:uncomment_begin(internal targets, remove for bazel query)
            # "@com_google_absl//absl/synchronization",
            # "@com_google_absl//absl/time",
            # "//thread",
            # copybara:uncomment_end
        ],
    }),
)

tfrt_cc_library(
    name = "kernel_runner",
    testonly = True,
    srcs = [
        "lib/utils/kernel_runner.cc",
    ],
    hdrs = [
        "include/tfrt/utils/kernel_runner.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_attr_encoder",
        ":hostcontext",
        ":support",
        ":tensor",
        "@llvm-project//llvm:Support",
    ],
)

# Change the maximum reported tracing levels with:
# --//third_party/tf_runtime:TFRT_MAX_TRACING_LEVEL=<value>
make_variable(
    name = "TFRT_MAX_TRACING_LEVEL",
    build_setting_default = "Verbose",
    values = [
        "None",
        "Default",
        "Verbose",
        "Debug",
    ],
)

tfrt_cc_library(
    name = "tracing",
    srcs = [
        "lib/tracing/tracing.cc",
    ],
    hdrs = [
        "include/tfrt/tracing/tracing.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    defines = ["TFRT_MAX_TRACING_LEVEL=$(TFRT_MAX_TRACING_LEVEL)"],
    toolchains = [":TFRT_MAX_TRACING_LEVEL"],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "simple_tracing_sink",
    srcs = ["lib/tracing/simple_tracing_sink.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

tfrt_cc_library(
    name = "debug_tracing_sink",
    srcs = ["lib/tracing/debug_tracing_sink.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

tfrt_cc_library(
    name = "chrome_tracing_sink",
    srcs = ["lib/tracing/chrome_tracing_sink.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

tfrt_cc_library(
    name = "nvtx_tracing_sink",
    srcs = ["lib/tracing/nvtx_tracing_sink.cc"],
    hdrs = ["include/tfrt/tracing/nvtx_tracing_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
    ] + if_google(
        ["@cuda_headers"],
        ["@nvtx_headers"],
    ),
    alwayslink = True,
)

tfrt_cc_library(
    name = "befexecutor",
    srcs = [
        "lib/bef_executor/bef_executor.cc",
        "lib/bef_executor/bef_file.cc",
        "lib/bef_executor/bef_file_impl.h",
        "lib/bef_executor/bef_interpreter.cc",
    ],
    hdrs = [
        "include/tfrt/bef/bef_encoding.h",
        "include/tfrt/bef_executor/bef_file.h",
        "include/tfrt/bef_executor/bef_interpreter.h",
        "include/tfrt/bef_executor/function_util.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_location",
        ":dtype",
        ":hostcontext",
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "metrics",
    srcs = [
        "lib/metrics/metrics.cc",
        "lib/metrics/metrics_registry.cc",
    ],
    hdrs = [
        "include/tfrt/metrics/common_metrics.h",
        "include/tfrt/metrics/gauge.h",
        "include/tfrt/metrics/histogram.h",
        "include/tfrt/metrics/metrics.h",
        "include/tfrt/metrics/metrics_registry.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "tensor",
    srcs = [
        "lib/tensor/btf.cc",
        "lib/tensor/btf_util.cc",
        "lib/tensor/conversion_registry.cc",
        "lib/tensor/coo_host_tensor.cc",
        "lib/tensor/coo_host_tensor_kernels.cc",
        "lib/tensor/dense_host_tensor.cc",
        "lib/tensor/dense_host_tensor_kernels.cc",
        "lib/tensor/dense_tensor_utils.cc",
        "lib/tensor/scalar_host_tensor.cc",
        "lib/tensor/string_host_tensor.cc",
        "lib/tensor/string_host_tensor_kernels.cc",
        "lib/tensor/tensor.cc",
        "lib/tensor/tensor_serialize_utils.cc",
        "lib/tensor/tensor_shape.cc",
        "lib/tensor/tensor_shape_kernels.cc",
        "lib/tensor/tensor_type_registration.cc",
    ],
    hdrs = [
        "include/tfrt/tensor/btf.h",
        "include/tfrt/tensor/btf_util.h",
        "include/tfrt/tensor/conversion_registry.h",
        "include/tfrt/tensor/conversion_utils.h",
        "include/tfrt/tensor/coo_host_tensor.h",
        "include/tfrt/tensor/dense_host_tensor.h",
        "include/tfrt/tensor/dense_host_tensor_kernels.h",
        "include/tfrt/tensor/dense_host_tensor_view.h",
        "include/tfrt/tensor/dense_tensor_utils.h",
        "include/tfrt/tensor/dense_view.h",
        "include/tfrt/tensor/host_tensor.h",
        "include/tfrt/tensor/scalar_host_tensor.h",
        "include/tfrt/tensor/string_host_tensor.h",
        "include/tfrt/tensor/string_host_tensor_kernels.h",
        "include/tfrt/tensor/tensor.h",
        "include/tfrt/tensor/tensor_metadata.h",
        "include/tfrt/tensor/tensor_serialize_utils.h",
        "include/tfrt/tensor/tensor_shape.h",
        "include/tfrt/tensor/tensor_type_registration.h",
    ],
    alwayslink_static_registration_src = "lib/tensor/static_registration.cc",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_attr_encoder",
        ":bef_emitter",
        ":dtype",
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_library(
    name = "basic_kernels",
    srcs = [
        "lib/basic_kernels/boolean_kernels.cc",
        "lib/basic_kernels/control_flow_kernels.cc",
        "lib/basic_kernels/device_kernels.cc",
        "lib/basic_kernels/float_kernels.cc",
        "lib/basic_kernels/integer_kernels.cc",
        "lib/basic_kernels/parallel_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/basic_kernels/basic_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/basic_kernels/static_registration.cc",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_library(
    name = "bef_emitter",
    srcs = [
        "lib/bef_converter/bef_emitter.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/bef_emitter.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":dtype",
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "mlirtobef",
    srcs = [
        "lib/bef_converter/mlir_to_bef/bef_attr_emitter.h",
        "lib/bef_converter/mlir_to_bef/bef_compilation_units.h",
        "lib/bef_converter/mlir_to_bef/bef_location_emitter.h",
        "lib/bef_converter/mlir_to_bef/bef_string_emitter.h",
        "lib/bef_converter/mlir_to_bef/mlir_to_bef.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/mlir_to_bef.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_attr_emitter",
        ":bef_attr_encoder",
        ":bef_emitter",
        ":bef_location_emitter",
        ":core_runtime_opdefs",
        ":dtype",
        ":stream_analysis",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "mlirtobef_translate",
    srcs = [
        "lib/bef_converter/mlir_to_bef/mlir_to_bef_translate.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/mlir_to_bef_translate.h",
    ],
    alwayslink_static_registration_src = "lib/bef_converter/mlir_to_bef/static_registration.cc",
    # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":init_tfrt_dialects",
        ":mlirtobef",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
    ],
)

tfrt_cc_library(
    name = "beftomlir",
    srcs = [
        "lib/bef_converter/bef_to_mlir/bef_attr_reader.h",
        "lib/bef_converter/bef_to_mlir/bef_location_reader.h",
        "lib/bef_converter/bef_to_mlir/bef_to_mlir.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/bef_to_mlir.h",
        "include/tfrt/host_context/attribute_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_attr_reader",
        ":bef_location",
        ":bef_location_reader",
        ":core_runtime_opdefs",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "beftomlir_translate",
    srcs = ["lib/bef_converter/bef_to_mlir/bef_to_mlir_translate.cc"],
    hdrs = ["include/tfrt/bef_converter/bef_to_mlir_translate.h"],
    alwayslink_static_registration_src = "lib/bef_converter/bef_to_mlir/static_registration.cc",
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":beftomlir",
        ":init_tfrt_dialects",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

tfrt_cc_library(
    name = "bef_attr_encoder",
    srcs = [
        "lib/bef_converter/bef_attr_encoder/bef_attr_encoder.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/bef_attr_encoder.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_emitter",
        ":dtype",
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "OpBaseTdFiles",
    srcs = [
        "include/tfrt/basic_kernels/opdefs/tfrt_base.td",
        "include/tfrt/tensor/opdefs/host_tensor.td",
        "include/tfrt/tensor/opdefs/tensor.td",
        "include/tfrt/tensor/opdefs/tensor_shape_base.td",
        "include/tfrt/tfrt_op_base.td",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

td_library(
    name = "CoreRTTdFiles",
    srcs = [
        "include/tfrt/core_runtime/opdefs/corert_base.td",
        "include/tfrt/core_runtime/opdefs/corert_traits.td",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

gentbl_cc_library(
    name = "basic_kernels_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/basic_kernels/opdefs/basic_kernels.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/basic_kernels/opdefs/basic_kernels_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/basic_kernels/opdefs/basic_kernels.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

tfrt_cc_library(
    name = "basic_kernels_opdefs",
    srcs = [
        "lib/basic_kernels/opdefs/basic_kernels.cc",
        "lib/basic_kernels/opdefs/tfrt_base.cc",
    ],
    hdrs = [
        "include/tfrt/basic_kernels/opdefs/basic_kernels.h",
        "include/tfrt/basic_kernels/opdefs/tfrt_base.h",
        "include/tfrt/basic_kernels/opdefs/types.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "tensor_shape_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tensor/opdefs/tensor_shape.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tensor/opdefs/tensor_shape.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=ts",
            ],
            "include/tfrt/tensor/opdefs/tensor_shape_dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=ts",
            ],
            "include/tfrt/tensor/opdefs/tensor_shape_dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/tensor_shape.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensor_shape_sync_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tensor/opdefs/tensor_shape_sync.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tensor/opdefs/tensor_shape_sync.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/tensor_shape_sync.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensor_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tensor/opdefs/tensor.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tensor/opdefs/tensor.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/tensor.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "host_tensor_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tensor/opdefs/host_tensor.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tensor/opdefs/host_tensor.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/host_tensor.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dense_host_tensor_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tensor/opdefs/dense_host_tensor.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tensor/opdefs/dense_host_tensor.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/dense_host_tensor.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dense_host_tensor_sync_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tensor/opdefs/dense_host_tensor_sync.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tensor/opdefs/dense_host_tensor_sync.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/dense_host_tensor_sync.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "coo_host_tensor_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tensor/opdefs/coo_host_tensor.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tensor/opdefs/coo_host_tensor.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/coo_host_tensor.td",
    deps = [
        ":OpBaseTdFiles",
    ],
)

tfrt_cc_library(
    name = "tensor_opdefs",
    srcs = [
        "lib/tensor/opdefs/coo_host_tensor.cc",
        "lib/tensor/opdefs/dense_host_tensor.cc",
        "lib/tensor/opdefs/dense_host_tensor_sync.cc",
        "lib/tensor/opdefs/host_tensor.cc",
        "lib/tensor/opdefs/tensor.cc",
        "lib/tensor/opdefs/tensor_shape.cc",
        "lib/tensor/opdefs/tensor_shape_sync.cc",
    ],
    hdrs = [
        "include/tfrt/tensor/opdefs/coo_host_tensor.h",
        "include/tfrt/tensor/opdefs/dense_host_tensor.h",
        "include/tfrt/tensor/opdefs/dense_host_tensor_sync.h",
        "include/tfrt/tensor/opdefs/host_tensor.h",
        "include/tfrt/tensor/opdefs/tensor.h",
        "include/tfrt/tensor/opdefs/tensor_shape.h",
        "include/tfrt/tensor/opdefs/tensor_shape_sync.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs",
        ":coo_host_tensor_opdefs_inc_gen",
        ":dense_host_tensor_opdefs_inc_gen",
        ":dense_host_tensor_sync_opdefs_inc_gen",
        ":host_tensor_opdefs_inc_gen",
        ":tensor_opdefs_inc_gen",
        ":tensor_shape_opdefs_inc_gen",
        ":tensor_shape_sync_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
    ],
)

gentbl_cc_library(
    name = "core_runtime_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/core_runtime/opdefs/core_runtime_opdefs.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/core_runtime/opdefs/core_runtime_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/core_runtime/opdefs/core_runtime.td",
    deps = [
        ":CoreRTTdFiles",
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

tfrt_cc_library(
    name = "core_runtime_opdefs",
    srcs = [
        "lib/core_runtime/opdefs/core_runtime.cc",
        "lib/core_runtime/opdefs/corert_utils.cc",
    ],
    hdrs = [
        "include/tfrt/core_runtime/opdefs/attributes.h",
        "include/tfrt/core_runtime/opdefs/core_runtime.h",
        "include/tfrt/core_runtime/opdefs/corert_utils.h",
        "include/tfrt/core_runtime/opdefs/traits.h",
        "include/tfrt/core_runtime/opdefs/types.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "core_runtime_sync_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/core_runtime/opdefs/sync/core_runtime_opdefs.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/core_runtime/opdefs/sync/core_runtime_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/core_runtime/opdefs/sync/core_runtime.td",
    deps = [
        ":CoreRTTdFiles",
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

tfrt_cc_library(
    name = "core_runtime_sync_opdefs",
    srcs = [
        "lib/core_runtime/opdefs/sync/core_runtime.cc",
    ],
    hdrs = [
        "include/tfrt/core_runtime/opdefs/sync/core_runtime.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs",
        ":core_runtime_sync_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "bef_executor_driver",
    srcs = [
        "lib/bef_executor_driver/bef_executor_driver.cc",
    ],
    hdrs = [
        "include/tfrt/bef_executor_driver/bef_executor_driver.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":befexecutor",
        ":core_runtime",
        ":hostcontext",
        ":metrics",
        ":profiled_allocator",
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_library(
    name = "core_runtime",
    srcs = [
        "lib/core_runtime/core_runtime.cc",
        "lib/core_runtime/core_runtime_op.cc",
        "lib/core_runtime/dispatch_utils.cc",
        "lib/core_runtime/execute_op_impl.cc",
        "lib/core_runtime/kernels.cc",
        "lib/core_runtime/logging_op_handler.cc",
        "lib/core_runtime/op_attrs.cc",
        "lib/core_runtime/tensor_handle.cc",
        "lib/core_runtime/test_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/core_runtime/core_runtime.h",
        "include/tfrt/core_runtime/core_runtime_op.h",
        "include/tfrt/core_runtime/dispatch_utils.h",
        "include/tfrt/core_runtime/execute_op_impl.h",
        "include/tfrt/core_runtime/kernels.h",
        "include/tfrt/core_runtime/logging_op_handler.h",
        "include/tfrt/core_runtime/op_args.h",
        "include/tfrt/core_runtime/op_attr_type.def",
        "include/tfrt/core_runtime/op_attr_type.h",
        "include/tfrt/core_runtime/op_attrs.h",
        "include/tfrt/core_runtime/op_handler.h",
        "include/tfrt/core_runtime/op_invocation.h",
        "include/tfrt/core_runtime/op_metadata_function.h",
        "include/tfrt/core_runtime/op_utils.h",
        "include/tfrt/core_runtime/tensor_handle.h",
    ],
    alwayslink_static_registration_src = "lib/core_runtime/static_registration.cc",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":dtype",
        ":hostcontext",
        ":support",
        ":tensor",
        ":tracing",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

# copybara:uncomment_begin
# py_library(
#     name = "btf_writer",
#     srcs = ["utils/btf_writer.py"],
#     srcs_version = "PY3",
#     visibility = ["//visibility:public"],
#     deps = ["//third_party/py/numpy"],
# )
# copybara:uncomment_end

tfrt_cc_library(
    name = "test_kernels",
    srcs = [
        "lib/test_kernels/async_kernels.cc",
        "lib/test_kernels/async_test_kernels.cc",
        "lib/test_kernels/atomic_test_kernels.cc",
        "lib/test_kernels/benchmark_kernels.cc",
        "lib/test_kernels/simple_kernels.cc",
        "lib/test_kernels/simple_test_kernels.cc",
        "lib/test_kernels/test_native_functions.cc",
        "lib/test_kernels/tutorial_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/test_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/test_kernels/static_registration.cc",
    visibility = ["//visibility:public"],
    deps = [
        ":befexecutor",
        ":hostcontext",
        ":support",
        ":tensor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

gentbl_cc_library(
    name = "test_kernels_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/test_kernels/opdefs/test_kernels.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/test_kernels/opdefs/test_kernels_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/test_kernels/opdefs/test_kernels.td",
    deps = [
        ":CoreRTTdFiles",
        ":OpBaseTdFiles",
        ":compiler_td_files",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

tfrt_cc_library(
    name = "test_kernels_opdefs",
    srcs = [
        "lib/test_kernels/opdefs/test_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/test_kernels/opdefs/test_kernels.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs",
        ":compiler_tfrt_op_interfaces",
        ":compiler_tfrt_traits",
        ":core_runtime_opdefs",
        ":tensor_opdefs",
        ":test_kernels_opdefs_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

gentbl_cc_library(
    name = "test_kernels_sync_opdefs_inc_gen",
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/test_kernels/opdefs/test_kernels_sync.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/test_kernels/opdefs/test_kernels_sync_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/test_kernels/opdefs/test_kernels_sync.td",
    deps = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

tfrt_cc_library(
    name = "test_kernels_sync_opdefs",
    srcs = [
        "lib/test_kernels/opdefs/test_kernels_sync.cc",
    ],
    hdrs = [
        "include/tfrt/test_kernels/opdefs/test_kernels_sync.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs",
        ":tensor_opdefs",
        ":test_kernels_sync_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "io",
    srcs = [
        "lib/io/buffered_input_stream.cc",
        "lib/io/file_input_stream.cc",
        "lib/io/file_system.cc",
    ] + select({
        ":windows": [
            "lib/io/windows_file_system.cc",
            "lib/io/windows_file_system.h",
        ],
        "//conditions:default": [
            "lib/io/posix_file_system.cc",
            "lib/io/posix_file_system.h",
        ],
    }),
    hdrs = [
        "include/tfrt/io/buffered_input_stream.h",
        "include/tfrt/io/file_input_stream.h",
        "include/tfrt/io/file_system.h",
        "include/tfrt/io/input_stream.h",
    ],
    alwayslink_static_registration_src = "lib/io/static_registration.cc",
    visibility = ["//visibility:public"],
    deps = [
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

td_library(
    name = "compiler_td_files",
    srcs = [
        "include/tfrt/compiler/opdefs/tfrt_op_interfaces.td",
        "include/tfrt/compiler/opdefs/tfrt_traits.td",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "compiler_tfrt_op_interfaces_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "include/tfrt/compiler/opdefs/tfrt_op_interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "include/tfrt/compiler/opdefs/tfrt_op_interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/compiler/opdefs/tfrt_op_interfaces.td",
    deps = [":compiler_td_files"],
)

tfrt_cc_library(
    name = "compiler_tfrt_op_interfaces",
    srcs = [
        "include/tfrt/compiler/opdefs/tfrt_op_interfaces.cc.inc",
        "include/tfrt/compiler/opdefs/tfrt_op_interfaces.h.inc",
        "lib/compiler/opdefs/tfrt_op_interfaces.cc",
    ],
    hdrs = ["include/tfrt/compiler/opdefs/tfrt_op_interfaces.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_tfrt_op_interfaces_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "compiler_tfrt_traits_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/compiler/opdefs/tf_traits.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/compiler/opdefs/tfrt_traits.td",
    deps = [":compiler_td_files"],
)

tfrt_cc_library(
    name = "compiler_tfrt_traits",
    srcs = ["lib/compiler/opdefs/tfrt_traits.cc"],
    hdrs = ["include/tfrt/compiler/opdefs/tfrt_traits.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_tfrt_traits_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "compiler_pass",
    srcs = [
        "lib/compiler/compiler_pass.cc",
    ],
    hdrs = [
        "include/tfrt/compiler/compiler_pass.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "stream_analysis",
    srcs = ["lib/compiler/stream_analysis.cc"],
    hdrs = ["include/tfrt/compiler/stream_analysis.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs",
        ":compiler_tfrt_op_interfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "print_stream_pass",
    srcs = ["lib/compiler/print_stream_pass.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":stream_analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:private"],
)

tfrt_cc_library(
    name = "init_tfrt_dialects",
    srcs = [
        "lib/init_tfrt_dialects.cc",
    ],
    hdrs = [
        "include/tfrt/init_tfrt_dialects.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
    visibility = ["//visibility:public"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs",
        ":core_runtime_sync_opdefs",
        ":tensor_opdefs",
        ":test_kernels_opdefs",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:VectorDialect",
        # copybara:uncomment "//third_party/tensorflow/compiler/xla/mlir/runtime/ir:rt",
        # copybara:uncomment "@tf_runtime//backends/jitrt:jitrt_opdefs",
    ],
)

tfrt_cc_library(
    name = "mlir_src_to_bef",
    srcs = [
        "lib/bef_converter/mlir_to_bef/mlir_src_to_bef.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/mlir_src_to_bef.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":init_tfrt_dialects",
        ":mlirtobef",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
)

tfrt_cc_library(
    name = "bef_attr_emitter",
    srcs = [
        "lib/bef_converter/mlir_to_bef/bef_attr_emitter.cc",
        "lib/bef_converter/mlir_to_bef/bef_attr_emitter.h",
        "lib/bef_converter/mlir_to_bef/bef_compilation_units.cc",
        "lib/bef_converter/mlir_to_bef/bef_compilation_units.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_attr_encoder",
        ":bef_emitter",
        ":core_runtime_opdefs",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "bef_attr_reader",
    srcs = [
        "lib/bef_converter/bef_to_mlir/bef_attr_reader.cc",
        "lib/bef_converter/bef_to_mlir/bef_attr_reader.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":core_runtime_opdefs",
        ":dtype",
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "bef_location_emitter",
    srcs = [
        "lib/bef_converter/mlir_to_bef/bef_location_emitter.cc",
        "lib/bef_converter/mlir_to_bef/bef_location_emitter.h",
        "lib/bef_converter/mlir_to_bef/bef_string_emitter.cc",
        "lib/bef_converter/mlir_to_bef/bef_string_emitter.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_emitter",
        ":bef_location",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "bef_string_emitter",
    srcs = [
        "lib/bef_converter/mlir_to_bef/bef_string_emitter.cc",
        "lib/bef_converter/mlir_to_bef/bef_string_emitter.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_emitter",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "bef_location",
    srcs = [
        "lib/bef/bef_location.cc",
    ],
    hdrs = [
        "include/tfrt/bef/bef_location.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "bef_location_reader",
    srcs = [
        "lib/bef_converter/bef_to_mlir/bef_location_reader.cc",
        "lib/bef_converter/bef_to_mlir/bef_location_reader.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bef",
        ":bef_location",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlir_runner_util",
    testonly = 1,
    srcs = ["lib/utils/mlir_runner_util.cc"],
    hdrs = ["include/tfrt/utils/mlir_runner_util.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
    ],
)
