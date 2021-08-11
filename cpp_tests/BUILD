load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library", "tfrt_cc_test")

licenses(["notice"])

package(
    # Don't enable because the internal gtest headers use c++17.
    # default_copts = ["-Wc++14-compat"],
    default_visibility = [":__subpackages__"],
)

tfrt_cc_library(
    name = "common",
    testonly = True,
    hdrs = [
        "include/tfrt/cpp_tests/error_util.h",
        "include/tfrt/cpp_tests/test_util.h",
    ],
    visibility = [
        # copybara:uncomment_begin
        # ":__subpackages__",
        # "//learning/brain/experimental/tfrt/cpp_tests:__subpackages__",
        # "//third_party/tensorflow/core/runtime_fallback:__subpackages__",
        # "//third_party/tensorflow/core/tfrt:__subpackages__",
        # "@tf_runtime//backends/gpu/cpp_tests:__subpackages__",
        # "//third_party/tf_runtime_google/cpp_tests:__subpackages__",
        # copybara:uncomment_end_and_comment_begin
        "//visibility:public",
        # copybara:comment_end
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "host_context/async_value_ref_test",
    srcs = ["host_context/async_value_ref_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/async_value_ptr_test",
    srcs = ["host_context/async_value_ptr_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/async_value_test",
    srcs = ["host_context/async_value_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/host_allocator_test",
    srcs = [
        "host_context/host_allocator_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/host_buffer_test",
    srcs = [
        "host_context/host_buffer_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "dtype/dtype_test",
    srcs = ["dtype/dtype_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/tensor_type_registration_test",
    srcs = ["support/tensor_type_registration_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "host_context/host_context_test",
    srcs = [
        "host_context/host_context_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/location_test",
    srcs = [
        "host_context/location_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/request_context_test",
    srcs = [
        "host_context/request_context_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/timer_queue_test",
    srcs = [
        "host_context/timer_queue_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/request_deadline_tracker_test",
    srcs = [
        "host_context/request_deadline_tracker_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/parallel_for_test",
    srcs = [
        "host_context/parallel_for_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/resource_context_test",
    srcs = [
        "host_context/resource_context_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "host_context/value_test",
    srcs = [
        "host_context/value_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "tensor/tensor_test",
    srcs = [
        "tensor/tensor_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "tensor/btf_test",
    srcs = [
        "tensor/btf_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "tensor/tensor_serialize_utils_test",
    srcs = [
        "tensor/tensor_serialize_utils_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "tensor/tensor_shape_test",
    srcs = [
        "tensor/tensor_shape_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "host_context/sync_kernel_test",
    srcs = [
        "host_context/sync_kernel_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/concurrent_vector_test",
    srcs = [
        "support/concurrent_vector_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/crc32c_test",
    srcs = [
        "support/crc32c_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/dense_host_tensor_test",
    srcs = [
        "support/dense_host_tensor_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "support/error_test",
    srcs = [
        "support/error_test.cc",
    ],
    deps = [
        ":common",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/hash_util_test",
    srcs = [
        "support/hash_util_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/random_util_test",
    srcs = [
        "support/random_util_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/ranges_test",
    srcs = [
        "support/ranges_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/ref_count_test",
    srcs = [
        "support/ref_count_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/string_util_test",
    srcs = [
        "support/string_util_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/variant_test",
    srcs = [
        "support/variant_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/map_by_type_test",
    srcs = [
        "support/map_by_type_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/latch_test",
    srcs = [
        "support/latch_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/philox_random_test",
    srcs = [
        "support/philox_random_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/thread_local_test",
    srcs = [
        "support/thread_local_test.cc",
    ],
    tags = ["no_oss"],  # TODO(b/178142173)
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_library(
    name = "driver_test_lib",
    testonly = True,
    srcs = [
        "core_runtime/driver.cc",
    ],
    hdrs = [
        "core_runtime/driver.h",
    ],
    visibility = [
        ":__subpackages__",
        # copybara:uncomment "//third_party/tf_runtime_google/cpp_tests:__subpackages__",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
    ],
)

tfrt_cc_test(
    name = "core_runtime/driver_test",
    srcs = [
        "core_runtime/driver_test.cc",
    ],
    deps = [
        ":driver_test_lib",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:test_ops_alwayslink",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_test(
    name = "core_runtime/op_attrs_test",
    srcs = [
        "core_runtime/op_attrs_test.cc",
    ],
    deps = [
        ":common",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "core_runtime/op_handler_test",
    srcs = ["core_runtime/op_handler_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:test_ops_alwayslink",
    ],
)

tfrt_cc_test(
    name = "support/aligned_buffer_test",
    srcs = ["support/aligned_buffer_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/bef_encoding_test",
    srcs = ["support/bef_encoding_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "support/bef_reader_test",
    srcs = ["support/bef_reader_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "utils/kernel_runner_test",
    srcs = ["utils/kernel_runner_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:kernel_runner",
        "@tf_runtime//:test_kernels_alwayslink",
    ],
)

tfrt_cc_test(
    name = "host_runtime/concurrent_work_queue_test",
    srcs = ["host_runtime/concurrent_work_queue_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

# Options to pass to 'bazel test' that affect what's measured:
# --copt=-DTFRT_DISABLE_TRACING:            strip tracing code.
# --copt=-DTFRT_BM_DISABLE_TRACING_REQUEST: do not request tracing.
tfrt_cc_test(
    name = "tracing",
    srcs = [
        "tracing/tracing_benchmark.cc",
        "tracing/tracing_test.cc",
    ],
    deps = [
        ":common",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
        "@tf_runtime//:tracing",
    ],
)

tfrt_cc_test(
    name = "bef_converter/bef_attr_encoder_test",
    srcs = ["bef_converter/bef_attr_encoder_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "bef_converter/bef_attr_emitter_test",
    srcs = ["bef_converter/bef_attr_emitter_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
        "@tf_runtime//:bef_attr_emitter",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "bef_converter/bef_attr_reader_test",
    srcs = ["bef_converter/bef_attr_reader_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
        "@tf_runtime//:bef_attr_emitter",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:bef_attr_reader",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_test(
    name = "bef_converter/bef_location_emitter_test",
    srcs = [
        "bef_converter/bef_location_emitter_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:bef_location",
        "@tf_runtime//:bef_location_emitter",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "bef_converter/bef_location_reader_test",
    srcs = ["bef_converter/bef_location_reader_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:bef_location_emitter",
        "@tf_runtime//:bef_location_reader",
    ],
)

tfrt_cc_test(
    name = "bef_converter/bef_string_emitter_test",
    srcs = [
        "bef_converter/bef_string_emitter_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:bef_string_emitter",
        "@tf_runtime//:support",
    ],
)
