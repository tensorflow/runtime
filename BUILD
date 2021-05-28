load(":build_defs.bzl", "tfrt_cc_library")

# copybara:uncomment load("//configlang/ncl/build_defs:ncl.bzl", "ncl_test")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "string_setting")
load("@tf_runtime//third_party/mlir:tblgen.bzl", "gentbl_cc_library")
# copybara:uncomment load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

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

# Setting to distinguish google-internal builds from open-source builds.
string_setting(
    name = "build_env",
    # copybara:uncomment_begin
    # build_setting_default = "google",
    # copybara:uncomment_end_and_comment_begin
    build_setting_default = "oss",
    # copybara:comment_end
    values = [
        "google",
        "oss",
    ],
    visibility = ["//visibility:private"],
)

# Setting whether to build in google environment. Use this in `select()`
# statements to conditionally provide different attributes for google and
# open-source builds.
config_setting(
    name = "is_build_env_google",
    flag_values = {":build_env": "google"},
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
    flag_values = {"rtti_and_exceptions": "False"},
    visibility = ["//visibility:public"],
)

# Config setting to conditionally link GPU targets.
alias(
    name = "gpu_enabled",
    # copybara:uncomment_begin
    # actual = "//tools/cc_target_os:linux-google",
    # copybara:uncomment_end_and_comment_begin
    actual = "@rules_cuda//cuda:is_cuda_enabled",
    # copybara:comment_end
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
# )
# copybara:uncomment_end

tfrt_cc_library(
    name = "profiled_allocator",
    srcs = ["lib/host_context/profiled_allocator.cc"],
    hdrs = ["include/tfrt/host_context/profiled_allocator.h"],
    visibility = [":friends"],
    deps = [":hostcontext"],
)

tfrt_cc_library(
    name = "hostcontext",
    srcs = [
        "lib/host_context/async_dispatch.cc",
        "lib/host_context/async_value.cc",
        "lib/host_context/async_value_ref.cc",
        "lib/host_context/chain.cc",
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
        "include/tfrt/host_context/debug_info.h",
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
    visibility = [":friends"],
    deps = [
        ":support",
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
        "include/tfrt/dtype/i1.h",
        "include/tfrt/dtype/quantized_types.h",
        "include/tfrt/support/bf16.h",
        "include/tfrt/support/forward_decls.h",
        "include/tfrt/support/fp16.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
    deps = [
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
        "lib/support/ref_count.cc",
        "lib/support/stack_trace.cc",
        "lib/support/string_util.cc",
    ],
    hdrs = [
        "include/tfrt/bef/bef_buffer.h",
        "include/tfrt/bef/bef_encoding.h",
        "include/tfrt/bef/bef_reader.h",
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
    visibility = [":friends"],
    deps = [
        ":dtype",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:unique_any",
    ] + select({
        ":use_std_thread": [],
        "//conditions:default": [
            # copybara:uncomment_begin(internal targets, remove for bazel query)
            # "//third_party/absl/synchronization",
            # "//third_party/absl/time",
            # "//thread",
            # copybara:uncomment_end
        ],
    }),
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
    visibility = [":friends"],
    deps = [
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "simple_tracing_sink",
    srcs = [
        "lib/tracing/simple_tracing_sink/simple_tracing_sink.cc",
    ],
    hdrs = [
        "include/tfrt/tracing/simple_tracing_sink/simple_tracing_sink.h",
    ],
    alwayslink_static_registration_src =
        "lib/tracing/simple_tracing_sink/static_registration.cc",
    visibility = [":friends"],
    deps = [
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "debug_tracing_sink",
    srcs = [
        "lib/tracing/debug_tracing_sink.cc",
    ],
    visibility = [":friends"],
    deps = [
        ":support",
        ":tracing",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
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
    visibility = [":friends"],
    deps = [
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
    visibility = [":friends"],
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
    visibility = [":friends"],
    deps = [
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
    visibility = [":friends"],
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
    visibility = [":friends"],
    deps = [
        ":dtype",
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "mlirtobef",
    srcs = [
        "include/tfrt/host_context/debug_info.h",
        "lib/bef_converter/mlir_to_bef/bef_attr_emitter.h",
        "lib/bef_converter/mlir_to_bef/bef_compilation_units.h",
        "lib/bef_converter/mlir_to_bef/mlir_to_bef.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/mlir_to_bef.h",
        "include/tfrt/host_context/debug_info.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
    deps = [
        ":bef_attr_emitter",
        ":bef_attr_encoder",
        ":bef_emitter",
        ":core_runtime_opdefs",
        ":dtype",
        ":stream_analysis",
        ":support",
        "@llvm-project//llvm:Support",
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
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
    deps = [
        ":init_tfrt_dialects",
        ":mlirtobef",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Translation",
    ],
)

tfrt_cc_library(
    name = "beftomlir",
    srcs = [
        "lib/bef_converter/bef_to_mlir/bef_attr_reader.h",
        "lib/bef_converter/bef_to_mlir/bef_to_mlir.cc",
    ],
    hdrs = [
        "include/tfrt/bef_converter/bef_to_mlir.h",
        "include/tfrt/host_context/attribute_utils.h",
    ],
    visibility = [":friends"],
    deps = [
        ":bef_attr_reader",
        ":core_runtime_opdefs",
        ":support",
        "@llvm-project//llvm:Support",
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
    visibility = [":friends"],
    deps = [
        ":beftomlir",
        ":init_tfrt_dialects",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
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
    visibility = [":friends"],
    deps = [
        ":bef_emitter",
        ":dtype",
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "OpBaseTdFiles",
    srcs = [
        "include/tfrt/basic_kernels/opdefs/tfrt_base.td",
        "include/tfrt/tensor/opdefs/host_tensor.td",
        "include/tfrt/tensor/opdefs/tensor.td",
        "include/tfrt/tensor/opdefs/tensor_shape_base.td",
        "include/tfrt/tfrt_op_base.td",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
)

exports_files(
    [
        "include/tfrt/tensor/opdefs/host_tensor.td",
        "include/tfrt/tensor/opdefs/tensor.td",
        "include/tfrt/tfrt_op_base.td",
        "include/tfrt/basic_kernels/opdefs/tfrt_base.td",
        "include/tfrt/core_runtime/opdefs/corert_traits.td",
        "include/tfrt/core_runtime/opdefs/corert_base.td",
    ],
    visibility = [":friends"],
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
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

tfrt_cc_library(
    name = "basic_kernels_opdefs",
    srcs = [
        "lib/basic_kernels/opdefs/basic_kernels.cc",
        "lib/basic_kernels/opdefs/tfrt_base.cc",
        "lib/basic_kernels/opdefs/tfrt_traits.cc",
    ],
    hdrs = [
        "include/tfrt/basic_kernels/opdefs/basic_kernels.h",
        "include/tfrt/basic_kernels/opdefs/tfrt_base.h",
        "include/tfrt/basic_kernels/opdefs/tfrt_traits.h",
        "include/tfrt/basic_kernels/opdefs/types.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
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
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tensor/opdefs/tensor_shape.td",
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    td_srcs = [
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
    visibility = [":friends"],
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
        "@llvm-project//mlir:SideEffects",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "include/tfrt/core_runtime/opdefs/corert_base.td",
        "include/tfrt/core_runtime/opdefs/corert_traits.td",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "include/tfrt/core_runtime/opdefs/corert_traits.td",
        "include/tfrt/core_runtime/opdefs/corert_base.td",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs",
        ":core_runtime_sync_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "bef_executor_driver",
    testonly = True,
    srcs = [
        "lib/bef_executor_driver/bef_executor_driver.cc",
    ],
    hdrs = [
        "include/tfrt/bef_executor_driver/bef_executor_driver.h",
    ],
    visibility = [":friends"],
    deps = [
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
    visibility = [":friends"],
    deps = [
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
#     visibility = [":friends"],
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
    visibility = [":friends"],
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
    td_srcs = [
        ":OpBaseTdFiles",
        "include/tfrt/core_runtime/opdefs/corert_traits.td",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs",
        ":tensor_opdefs",
        ":test_kernels_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
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
    td_srcs = [
        ":OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
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
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":tensor_opdefs",
        ":test_kernels_sync_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "io",
    srcs = [
        "lib/io/buffered_input_stream.cc",
        "lib/io/file_input_stream.cc",
        "lib/io/file_system.cc",
        "lib/io/posix_file_system.cc",
        "lib/io/posix_file_system.h",
    ],
    hdrs = [
        "include/tfrt/io/buffered_input_stream.h",
        "include/tfrt/io/file_input_stream.h",
        "include/tfrt/io/file_system.h",
        "include/tfrt/io/input_stream.h",
    ],
    alwayslink_static_registration_src = "lib/io/static_registration.cc",
    visibility = [":friends"],
    deps = [
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_library(
    name = "data",
    srcs = [
        "lib/data/batch_dataset.h",
        "lib/data/data_kernels.cc",
        "lib/data/dataset.cc",
        "lib/data/filter_dataset.cc",
        "lib/data/filter_dataset.h",
        "lib/data/interleave_dataset.cc",
        "lib/data/interleave_dataset.h",
        "lib/data/io.cc",
        "lib/data/io.h",
        "lib/data/log_dataset.h",
        "lib/data/map_dataset.cc",
        "lib/data/map_dataset.h",
        "lib/data/memory_dataset.h",
        "lib/data/prefetch_dataset.cc",
        "lib/data/prefetch_dataset.h",
        "lib/data/range_dataset.cc",
        "lib/data/range_dataset.h",
        "lib/data/repeat_dataset.cc",
        "lib/data/repeat_dataset.h",
        "lib/data/shuffle_dataset.cc",
        "lib/data/shuffle_dataset.h",
        "lib/data/skip_dataset.cc",
        "lib/data/skip_dataset.h",
        "lib/data/slice_dataset.h",
        "lib/data/tf_record_dataset.cc",
        "lib/data/tf_record_dataset.h",
    ],
    hdrs = [
        "include/tfrt/data/dataset.h",
    ],
    alwayslink_static_registration_src = "lib/data/static_registration.cc",
    visibility = [":friends"],
    deps = [
        ":dtype",
        ":hostcontext",
        ":io",
        ":support",
        ":tensor",
        ":tracing",
        "@llvm-project//llvm:Support",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

gentbl_cc_library(
    name = "data_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/data/opdefs/data_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/data/opdefs/data_ops_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/data/opdefs/data_ops.td",
    td_srcs = [
        ":OpBaseTdFiles",
    ],
)

tfrt_cc_library(
    name = "data_opdefs",
    srcs = [
        "lib/data/opdefs/data_ops.cc",
    ],
    hdrs = [
        "include/tfrt/data/opdefs/data_ops.h",
        "include/tfrt/data/opdefs/types.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":data_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tfrt_cc_library(
    name = "distributed_runtime",
    srcs = [
        "lib/distributed_runtime/callback_registry.cc",
        "lib/distributed_runtime/cluster_info.cc",
        "lib/distributed_runtime/distributed_context.cc",
        "lib/distributed_runtime/distributed_init_helper.cc",
        "lib/distributed_runtime/function_cache.cc",
        "lib/distributed_runtime/op_handler_kernels.cc",
        "lib/distributed_runtime/remote_chain_manager.cc",
        "lib/distributed_runtime/remote_device.cc",
        "lib/distributed_runtime/remote_object_manager.cc",
        "lib/distributed_runtime/remote_op_handler.cc",
        "lib/distributed_runtime/remote_tensor.cc",
        "lib/distributed_runtime/request_handler_impl.cc",
        "lib/distributed_runtime/server_context.cc",
        "lib/distributed_runtime/task_handle.cc",
        "lib/distributed_runtime/task_name_util.cc",
    ],
    hdrs = [
        "include/tfrt/distributed_runtime/callback_registry.h",
        "include/tfrt/distributed_runtime/cluster_info.h",
        "include/tfrt/distributed_runtime/distributed_context.h",
        "include/tfrt/distributed_runtime/distributed_init_helper.h",
        "include/tfrt/distributed_runtime/fabric_communicator.h",
        "include/tfrt/distributed_runtime/function_cache.h",
        "include/tfrt/distributed_runtime/payload.h",
        "include/tfrt/distributed_runtime/remote_chain_manager.h",
        "include/tfrt/distributed_runtime/remote_client.h",
        "include/tfrt/distributed_runtime/remote_device.h",
        "include/tfrt/distributed_runtime/remote_execute.h",
        "include/tfrt/distributed_runtime/remote_object.h",
        "include/tfrt/distributed_runtime/remote_object_manager.h",
        "include/tfrt/distributed_runtime/remote_op_handler.h",
        "include/tfrt/distributed_runtime/remote_tensor.h",
        "include/tfrt/distributed_runtime/request_handler.h",
        "include/tfrt/distributed_runtime/request_handler_impl.h",
        "include/tfrt/distributed_runtime/server_context.h",
        "include/tfrt/distributed_runtime/task_handle.h",
        "include/tfrt/distributed_runtime/task_name_util.h",
        "lib/distributed_runtime/op_handler_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/distributed_runtime/static_registration.cc",
    visibility = [":friends"],
    deps = [
        ":befexecutor",
        ":cluster_config_cc_proto",
        ":compiler_pass",
        ":core_runtime",
        ":hostcontext",
        ":mlir_src_to_bef",
        ":mlirtobef",
        ":remote_message_cc_proto",
        ":support",
        ":tensor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
)

proto_library(
    name = "cluster_config_proto",
    srcs = ["include/tfrt/distributed_runtime/proto/cluster_config.proto"],
    # copybara:uncomment cc_api_version = 2,
    visibility = [":friends"],
)

cc_proto_library(
    name = "cluster_config_cc_proto",
    visibility = [":friends"],
    deps = [":cluster_config_proto"],
)

proto_library(
    name = "remote_message_proto",
    srcs = ["include/tfrt/distributed_runtime/proto/remote_message.proto"],
    # copybara:uncomment cc_api_version = 2,
    visibility = [":friends"],
    deps = [":cluster_config_proto"],
)

cc_proto_library(
    name = "remote_message_cc_proto",
    visibility = [":friends"],
    deps = [":remote_message_proto"],
)

tfrt_cc_library(
    name = "distributed_kernels",
    srcs = [
        "lib/distributed_runtime/kernels.cc",
        "lib/distributed_runtime/test_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/distributed_runtime/distributed_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/distributed_runtime/kernels_static_registration.cc",
    visibility = [":friends"],
    deps = [
        ":compiler_pass",
        ":core_runtime",
        ":distributed_runtime",
        ":hostcontext",
        ":init_tfrt_dialects",
        ":remote_message_cc_proto",
        ":support",
        ":tensor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

gentbl_cc_library(
    name = "distributed_kernels_opdefs_inc_gen",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/distributed_runtime/opdefs/kernels.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/distributed_runtime/opdefs/kernels_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/distributed_runtime/opdefs/kernels.td",
    td_srcs = [
        ":OpBaseTdFiles",
        "@tf_runtime//:include/tfrt/core_runtime/opdefs/corert_base.td",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

tfrt_cc_library(
    name = "distributed_kernels_opdefs",
    srcs = [
        "lib/distributed_runtime/opdefs/kernels.cc",
    ],
    hdrs = [
        "include/tfrt/distributed_runtime/opdefs/kernels.h",
        "include/tfrt/distributed_runtime/opdefs/types.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs",
        ":distributed_kernels_opdefs_inc_gen",
        ":tensor_opdefs",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
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
    visibility = [":friends"],
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
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        "@llvm-project//mlir:IR",
    ],
)

tfrt_cc_library(
    name = "print_stream_pass",
    srcs = ["lib/compiler/print_stream_pass.cc"],
    visibility = [":friends"],
    deps = [
        ":stream_analysis",
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
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs",
        ":core_runtime_sync_opdefs",
        ":data_opdefs",
        ":distributed_kernels_opdefs",
        ":tensor_opdefs",
        ":test_kernels_opdefs",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:StandardOps",
        "@tf_runtime//backends/cpu:cpurt_opdefs",
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
    visibility = [":friends"],
    deps = [
        ":init_tfrt_dialects",
        ":mlirtobef",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
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
    visibility = [":friends"],
    deps = [
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
    visibility = [":friends"],
    deps = [
        ":core_runtime_opdefs",
        ":dtype",
        ":hostcontext",
        ":support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "tpu_opdefs_inc_gen",
    includes = ["include"],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/tpu/opdefs/tpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/tpu/opdefs/tpu_ops_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/tpu/opdefs/tpu_ops.td",
    td_srcs = [
        ":OpBaseTdFiles",
        "@tf_runtime//:include/tfrt/core_runtime/opdefs/corert_base.td",
    ],
)

tfrt_cc_library(
    name = "tpu_opdefs",
    srcs = [
        "lib/tpu/opdefs/tpu_ops.cc",
    ],
    hdrs = [
        "include/tfrt/tpu/opdefs/tpu_ops.h",
    ],
    visibility = [":friends"],
    deps = [
        ":basic_kernels_opdefs",
        ":core_runtime_opdefs",
        ":tensor_opdefs",
        ":tpu_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
