load("@tf_runtime//third_party/mlir:tblgen.bzl", "gentbl")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")
load(
    "@rules_cuda//cuda:defs.bzl",
    "cuda_library",
)

package(
    default_visibility = [":__subpackages__"],
)

licenses(["notice"])

package_group(
    name = "tests_and_tools",
    packages = [
        "//backends/gpu/...",
        # Authorized users go here.
        "//tools/...",
        "//cpp_tests/...",
    ],
)

# Generated API headers and implementations for HIP libraries.
# Note: Only gpu_stream_wrapper should depend on hip_stubs.
tfrt_cc_library(
    name = "hip_stubs",
    srcs = ["lib/stream/hip_stub.cc"],
    hdrs = [
        "include/tfrt/gpu/stream/hip_forwards.h",
        "include/tfrt/gpu/stream/hip_stub.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        # No HIP shared object dependencies, they are dynamically loaded.
        "@tf_runtime//:support",
        "@tf_runtime//third_party/hip:stub_inc",
    ],
)

# Generated API implementations for CUDA libraries.
# Note: Only gpu_stream_wrapper should depend on cuda_stubs, and
# --@rules_cuda//cuda:cuda_runtime should be set to this target.
tfrt_cc_library(
    name = "cuda_stubs",
    srcs = [
        "lib/stream/cublas_stub.cc",
        "lib/stream/cuda_stub.cc",
        "lib/stream/cudart_stub.cc",
        "lib/stream/cudnn_stub.cc",
        "lib/stream/cusolver_stub.cc",
    ],
    visibility = ["//visibility:private"],
    deps = [
        # No CUDA shared object dependencies, they are dynamically loaded.
        # TODO(csigg): change to @cuda_headers once crt is included.
        "@local_cuda//:cuda_headers",
        "@cudnn_headers",
        "@tf_runtime//:support",
        "@tf_runtime//third_party/cuda:stub_inc",
    ],
)

tfrt_cc_library(
    name = "gpu_stream_wrapper",
    srcs = [
        "include/tfrt/gpu/stream/cuda_forwards.h",
        "lib/stream/blas_wrapper.cc",
        "lib/stream/cublas_wrapper.cc",
        "lib/stream/cuda_wrapper.cc",
        "lib/stream/cudart_wrapper.cc",
        "lib/stream/cudnn_wrapper.cc",
        "lib/stream/cusolver_wrapper.cc",
        "lib/stream/dnn_wrapper.cc",
        "lib/stream/hash_utils.cc",
        "lib/stream/hip_wrapper.cc",
        "lib/stream/solver_wrapper.cc",
        "lib/stream/stream_wrapper.cc",
        "lib/stream/wrapper_detail.h",
    ],
    hdrs = [
        "include/tfrt/gpu/stream/blas_wrapper.h",
        "include/tfrt/gpu/stream/cublas_wrapper.h",
        "include/tfrt/gpu/stream/cuda_wrapper.h",
        "include/tfrt/gpu/stream/cudart_wrapper.h",
        "include/tfrt/gpu/stream/cudnn_wrapper.h",
        "include/tfrt/gpu/stream/cusolver_wrapper.h",
        "include/tfrt/gpu/stream/dense_map_utils.h",
        "include/tfrt/gpu/stream/dnn_wrapper.h",
        "include/tfrt/gpu/stream/hash_utils.h",
        "include/tfrt/gpu/stream/hip_wrapper.h",
        "include/tfrt/gpu/stream/solver_wrapper.h",
        "include/tfrt/gpu/stream/stream_wrapper.h",
    ],
    visibility = [
        ":tests_and_tools",
    ],
    deps = [
        ":cuda_stubs",
        ":hip_stubs",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_library(
    name = "gpu_memory",
    srcs = [
        "lib/memory/bfc_gpu_allocator.cc",
        "lib/memory/block_allocator.cc",
        "lib/memory/caching_gpu_allocator.cc",
        "lib/memory/gpu_allocator.cc",
        "lib/memory/gpu_buffer.cc",
        "lib/memory/sub_allocator.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/memory/bfc_gpu_allocator.h",
        "include/tfrt/gpu/memory/block_allocator.h",
        "include/tfrt/gpu/memory/caching_gpu_allocator.h",
        "include/tfrt/gpu/memory/gpu_allocator.h",
        "include/tfrt/gpu/memory/gpu_buffer.h",
        "include/tfrt/gpu/memory/sub_allocator.h",
    ],
    deps = [
        ":gpu_stream_wrapper",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tracing",
    ],
)

tfrt_cc_library(
    name = "gpu_config",
    srcs = [
        "lib/device/gpu_config.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/device/gpu_config.h",
    ],
    visibility = [
        "@tf_runtime//:friends",
    ],
    deps = [
        ":gpu_memory",
        ":gpu_stream_wrapper",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_library(
    name = "gpu_tensor",
    srcs = [
        "lib/tensor/dense_gpu_tensor.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/tensor/dense_gpu_tensor.h",
    ],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":gpu_memory",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "gpu_event_manager",
    srcs = [
        "lib/event_manager.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/event_manager.h",
    ],
    deps = [
        ":gpu_stream_wrapper",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_library(
    name = "gpu_op_handler",
    srcs = [
        "lib/core_runtime/gpu_op_handler.cc",
        "lib/core_runtime/gpu_op_registry.cc",
        "lib/core_runtime/op_handler_kernels.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/core_runtime/gpu_dispatch_context.h",
        "include/tfrt/gpu/core_runtime/gpu_op_handler.h",
        "include/tfrt/gpu/core_runtime/gpu_op_registry.h",
        "include/tfrt/gpu/core_runtime/gpu_op_utils.h",
        "lib/core_runtime/gpu_op_registry_impl.h",
        "lib/core_runtime/op_handler_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/core_runtime/static_registration.cc",
    visibility = [
        ":tests_and_tools",
        "@tf_runtime//:friends",
    ],
    deps = [
        ":gpu_config",
        ":gpu_device_alwayslink",
        ":gpu_memory",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "gpu_device",
    srcs = [
        "lib/device/conversion_function.cc",
        "lib/device/device.cc",
        "lib/device/device_util.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/device/conversion_function.h",
        "include/tfrt/gpu/device/device.h",
        "include/tfrt/gpu/device/device_util.h",
    ],
    alwayslink_static_registration_src = "lib/device/static_registration.cc",
    visibility = [
        ":tests_and_tools",
        "@tf_runtime//:friends",
    ],
    deps = [
        ":gpu_config",
        ":gpu_device_eigen_support",
        ":gpu_memory",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "gpu_device_eigen_support",
    srcs = ["lib/device/eigen_support.cc"],
    hdrs = ["lib/device/eigen_support.h"],
    deps = [
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@cuda_headers",
        "@eigen_archive//:eigen3",
        "@tf_runtime//:dtype",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
    ],
)

cuda_library(
    name = "test_cuda_kernels",
    srcs = ["lib/ops/test/test_cuda_kernels.cu.cc"],
    hdrs = ["lib/ops/test/test_cuda_kernels.h"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
    ],
)

tfrt_cc_library(
    name = "gpu_test_ops",
    srcs = [
        "lib/ops/test/test_ops.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/ops/test/gpu_ops_and_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/ops/test/static_registration.cc",
    visibility = [
        "@tf_runtime//:friends",
    ],
    deps = [
        ":gpu_device",
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":test_cuda_kernels",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:test_metadata_functions",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_library(
    name = "blas_support",
    hdrs = [
        "include/tfrt/gpu/blas_support.h",
    ],
    deps = [
        ":gpu_stream_wrapper",
        "@llvm-project//llvm:Support",
    ],
)

tfrt_cc_library(
    name = "gpu_tf_ops",
    srcs = [
        "lib/ops/tf/gpu_ops.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/ops/tf/gpu_ops.h",
    ],
    alwayslink_static_registration_src = "lib/ops/tf/static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        ":tf_gpu_binary_ops",
        ":tf_gpu_dnn_ops",
        ":tf_gpu_matmul_op",
        ":tf_gpu_mlir_ops",
        ":tf_gpu_nullary_ops",
        ":tf_gpu_pad_op",
        ":tf_gpu_reduction_ops",
        ":tf_gpu_transpose_op",
        ":tf_gpu_unary_ops",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigen_kernels",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_metadata_functions",
    ],
)

tfrt_cc_library(
    name = "tf_gpu_nullary_ops",
    srcs = ["lib/ops/tf/nullary_ops.cc"],
    hdrs = ["lib/ops/tf/nullary_ops.h"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

cuda_library(
    name = "tf_gpu_transpose_op",
    srcs = ["lib/ops/tf/transpose_op.cu.cc"],
    hdrs = ["lib/ops/tf/transpose_op.h"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
    ],
)

cuda_library(
    name = "tf_gpu_reduction_ops",
    srcs = [
        "lib/ops/tf/eigen_helper.cu.h",
        "lib/ops/tf/reduction_ops.cu.cc",
    ],
    hdrs = ["lib/ops/tf/reduction_ops.h"],
    deps = [
        "@tf_runtime//:core_runtime",
        ":gpu_op_handler",
        ":gpu_memory",
        ":gpu_tensor",
        ":gpu_stream_wrapper",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//:dtype",
        "@tf_runtime//backends/common:tf_dnn_ops_util",
        "@eigen_archive//:eigen3",
        # TODO(ezhulenev): Figure out how to use hipcub.
        "@cub_archive//:cub",
        "@llvm-project//llvm:Support",
    ],
)

cuda_library(
    name = "tf_gpu_unary_ops",
    srcs = [
        "lib/ops/tf/eigen_helper.cu.h",
        "lib/ops/tf/unary_ops.cu.cc",
    ],
    hdrs = ["lib/ops/tf/unary_ops.h"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
    ],
)

tfrt_cc_library(
    name = "generated_kernel_images",
    hdrs = [
        "include/tfrt/gpu/ops/tf/bias_add_f16_kernel.h",
        "include/tfrt/gpu/ops/tf/bias_add_f32_kernel.h",
        "include/tfrt/gpu/ops/tf/bias_add_f64_kernel.h",
        "include/tfrt/gpu/ops/tf/relu_f16_kernel.h",
        "include/tfrt/gpu/ops/tf/relu_f32_kernel.h",
        "include/tfrt/gpu/ops/tf/relu_f64_kernel.h",
    ],
    includes = ["backends/gpu/include"],
)

tfrt_cc_library(
    name = "tf_gpu_mlir_ops",
    srcs = ["lib/ops/tf/mlir_ops.cc"],
    hdrs = ["lib/ops/tf/mlir_ops.h"],
    deps = [
        ":generated_kernel_images",
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tfrt_cc_library(
    name = "tf_gpu_dnn_ops",
    srcs = [
        "lib/ops/tf/dnn_ops.cc",
    ],
    hdrs = ["lib/ops/tf/dnn_ops.h"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        ":pad_op_noncuda",
        ":tf_gpu_dnn_ops_cu",
        ":tf_gpu_matmul_op",  # TODO(iga): For GEMM-calling utility only.
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigen_kernels",  # TODO(csigg): Only for shape_functions.h, fix.
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_dnn_ops_util",
    ],
)

cuda_library(
    name = "tf_gpu_dnn_ops_cu",
    srcs = [
        "lib/ops/tf/dnn_ops.cu.cc",
    ],
    hdrs = ["lib/ops/tf/dnn_ops_cu.h"],
    features = ["-parse_headers"],  # See b/152520254
    deps = [
        ":gpu_memory",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_dnn_ops_util",
    ],
)

cuda_library(
    name = "tf_gpu_binary_ops",
    srcs = [
        "lib/ops/tf/binary_ops.cu.cc",
        "lib/ops/tf/eigen_helper.cu.h",
    ],
    hdrs = ["lib/ops/tf/binary_ops.h"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
    ],
)

tfrt_cc_library(
    name = "tf_gpu_matmul_op",
    srcs = ["lib/ops/tf/matmul_op.cc"],
    hdrs = ["lib/ops/tf/matmul_op.h"],
    copts = ["-mf16c"],
    deps = [
        ":blas_support",
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//:tracing",
    ],
)

cuda_library(
    name = "tf_gpu_pad_op",
    srcs = [
        "lib/ops/tf/pad_op.cc",
        "lib/ops/tf/pad_op.cu.cc",
    ],
    hdrs = ["lib/ops/tf/pad_op.h"],
    defines = [
        "EIGEN_MUTEX=std::mutex",
        "EIGEN_MUTEX_LOCK=std::unique_lock<std::mutex>",
        "EIGEN_CONDVAR=std::condition_variable",
        "EIGEN_AVOID_STL_ARRAY",
    ],
    deps = [
        ":blas_support",
        ":gpu_device_eigen_support",
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_dnn_ops_util",
    ],
)

tfrt_cc_library(
    name = "pad_op_noncuda",
    srcs = [
        "lib/ops/tf/pad_op_noncuda.cc",
    ],
    hdrs = [
        "lib/ops/tf/pad_op_noncuda.h",
    ],
    deps = [
        ":gpu_tensor",
        ":tf_gpu_pad_op",
        "@tf_runtime//:support",
    ],
)

filegroup(
    name = "cuda_test_opdefs_ops",
    srcs = [
        "include/tfrt/gpu/kernels/cuda_opdefs/cuda_test_ops.h",
    ],
)

gentbl(
    name = "cuda_opdefs_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/tfrt/gpu/kernels/cuda_opdefs/cuda_opdefs.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/tfrt/gpu/kernels/cuda_opdefs/cuda_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/gpu/kernels/cuda_opdefs/cuda_ops.td",
    td_includes = [
        "include",
        "backends/gpu/include",
    ],
    td_srcs = [
        "@tf_runtime//:OpBaseTdFiles",
        "include/tfrt/gpu/kernels/cuda_opdefs/cuda_ops_base.td",
    ],
)

gentbl(
    name = "cuda_test_opdefs_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "include/tfrt/gpu/kernels/cuda_opdefs/cuda_test_opdefs.h.inc",
        ),
        (
            "-gen-op-defs",
            "include/tfrt/gpu/kernels/cuda_opdefs/cuda_test_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/gpu/kernels/cuda_opdefs/cuda_test_ops.td",
    td_includes = [
        "include",
        "backends/gpu/include",
    ],
    td_srcs = [
        "@tf_runtime//:OpBaseTdFiles",
        "include/tfrt/gpu/kernels/cuda_opdefs/cuda_ops_base.td",
    ],
)

tfrt_cc_library(
    name = "cuda_opdefs",
    srcs = [
        "lib/kernels/cuda_opdefs/cuda_ops.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/kernels/cuda_opdefs/cuda_ops.h",
    ],
    alwayslink_static_registration_src = "lib/kernels/cuda_opdefs/static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":cuda_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

tfrt_cc_library(
    name = "cuda_test_opdefs",
    srcs = [
        "lib/kernels/cuda_opdefs/cuda_test_ops.cc",
    ],
    hdrs = [
        ":cuda_test_opdefs_ops",
    ],
    alwayslink_static_registration_src = "lib/kernels/cuda_opdefs/test_static_registration.cc",
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":cuda_opdefs",
        ":cuda_opdefs_inc_gen",
        ":cuda_test_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

tfrt_cc_library(
    name = "cuda",
    srcs = [
        "lib/kernels/cuda/kernels.cc",
        "lib/kernels/cuda/kernels.h",
    ],
    alwayslink_static_registration_src = "lib/kernels/cuda/static_registration.cc",
    visibility = [":tests_and_tools"],
    deps = [
        ":cuda_opdefs_alwayslink",
        ":gpu_memory",
        ":gpu_stream_wrapper",
        ":gpu_tensor",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_library(
    name = "cuda_test",
    srcs = [
        "lib/kernels/cuda/test_kernels.cc",
        "lib/kernels/cuda/test_kernels.h",
    ],
    alwayslink_static_registration_src = "lib/kernels/cuda/test_static_registration.cc",
    visibility = [":tests_and_tools"],
    deps = [
        ":cuda",
        ":gpu_stream_wrapper",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)
