load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load(
    "@tf_runtime//:build_defs.bzl",
    "if_google",
    "if_oss",
    "tfrt_cc_binary",
    "tfrt_cc_library",
)
load(
    # copybara:uncomment_begin
    # "@cuda_headers//:build_defs.bzl",
    # copybara:uncomment_end_and_comment_begin
    "@rules_cuda//cuda:defs.bzl",
    # copybara:comment_end
    "cuda_library",
)

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

package_group(
    name = "tests_and_tools",
    packages = if_google(
        [
            "//learning/brain/experimental/tfrt/cpp_tests/...",
            "@tf_runtime//backends/gpu/...",
            "@tf_runtime//tools/...",
            "@tf_runtime//cpp_tests/...",
        ],
        [
            # Note: packages may not contain repos (e.g. @tf_runtime//...)
            "//backends/gpu/...",
            "//tools/...",
            "//cpp_tests/...",
        ],
    ),
)

package_group(
    name = "xla_friends",
    packages = [
    ] + if_google(
        [
            "//third_party/tensorflow/compiler/xla/service/gpu/...",
            "//third_party/tensorflow/compiler/mlir/tfrt/...",
            "@tf_runtime//backends/gpu/...",
            "@tf_runtime//cpp_tests/...",
            "//learning/brain/experimental/tfrt/cpp_tests/...",
        ],
        [
            # Note: packages may not contain repos (e.g. @tf_runtime//...)
            "//backends/gpu/...",
            "//cpp_tests/...",
        ],
    ),
)

# Config setting to conditionally build CUDA device code. That is, cuda_library
# targets should only be depended on if this setting is true. This setting does
# not affect CUDA host-only code.
alias(
    name = "cuda_enabled",
    actual = if_google(
        "//tools/cc_target_os:linux-google",
        "@rules_cuda//cuda:is_cuda_enabled",
    ),
)

tfrt_cc_library(
    name = "symbol_loader",
    hdrs = ["lib/wrapper/symbol_loader.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:support",
    ],
)

# Generated API headers and implementations for HIP libraries.
# Note: Only gpu_wrapper should depend on rocm_stubs.
tfrt_cc_library(
    name = "rocm_stubs",
    srcs = [
        "lib/wrapper/hip_stub.cc",
        "lib/wrapper/hipfft_stub.cc",
        "lib/wrapper/miopen_stub.cc",
        "lib/wrapper/rccl_stub.cc",
        "lib/wrapper/rocblas_stub.cc",
        "lib/wrapper/rocsolver_stub.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/wrapper/hip_forwards.h",
        "include/tfrt/gpu/wrapper/hip_stub.h",
        "include/tfrt/gpu/wrapper/hipfft_stub.h",
        "include/tfrt/gpu/wrapper/miopen_stub.h",
        "include/tfrt/gpu/wrapper/rccl_stub.h",
        "include/tfrt/gpu/wrapper/rocblas_stub.h",
        "include/tfrt/gpu/wrapper/rocsolver_stub.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        # No HIP shared object dependencies, they are dynamically loaded.
        ":symbol_loader",
        "@cuda_headers",
        "@tf_runtime//third_party/hip:stub_inc",
    ] + if_google(
        # rccl_stub.h includes NCCL headers for its enums and types.
        ["@nccl_headers//:src_hdrs"],
        ["@nccl_headers"],
    ),
)

# Headers and generated API implementations for CUDA libraries.
#
# No target should depend on this explicitly.
# It should rather be used (through .bazelrc) as
# '--@rules_cuda//cuda:cuda_runtime=@tf_runtime//backends/gpu:cuda_stubs'
# which makes all cuda_library() targets implicitly depend on it.
#
# Not used internally, but we want to include it in CI builds.
tfrt_cc_library(
    name = "cuda_stubs",
    srcs = [
        "lib/wrapper/cublas_stub.cc",
        "lib/wrapper/cuda_stub.cc",
        "lib/wrapper/cudart_stub.cc",
        "lib/wrapper/cudnn_stub.cc",
        "lib/wrapper/cufft_stub.cc",
        "lib/wrapper/cusolver_stub.cc",
        "lib/wrapper/nccl_stub.cc",
    ],
    visibility = ["//visibility:private"],
    # No CUDA shared object dependencies, they are dynamically loaded.
    deps = [
        ":symbol_loader",
        "@cuda_headers",
        "@tf_runtime//third_party/cuda:stub_inc",
    ] + if_google(
        ["@cudnn_headers//:cudnn_header"],
        ["@cudnn_headers"],
    ) + if_google(
        ["@nccl_headers//:src_hdrs"],
        ["@nccl_headers"],
    ),
)

tfrt_cc_library(
    name = "gpu_wrapper",
    srcs = [
        "include/tfrt/gpu/wrapper/cuda_forwards.h",
        "include/tfrt/gpu/wrapper/cuda_type_traits.h",
        "lib/wrapper/blas_wrapper.cc",
        "lib/wrapper/ccl_enums.cc",
        "lib/wrapper/ccl_wrapper.cc",
        "lib/wrapper/cublas_enums.cc",
        "lib/wrapper/cublas_wrapper.cc",
        "lib/wrapper/cuda_wrapper.cc",
        "lib/wrapper/cudart_wrapper.cc",
        "lib/wrapper/cudnn_enums.cc",
        "lib/wrapper/cudnn_wrapper.cc",
        "lib/wrapper/cufft_wrapper.cc",
        "lib/wrapper/cusolver_wrapper.cc",
        "lib/wrapper/dnn_wrapper.cc",
        "lib/wrapper/driver_wrapper.cc",
        "lib/wrapper/fft_wrapper.cc",
        "lib/wrapper/hip_wrapper.cc",
        "lib/wrapper/hipfft_wrapper.cc",
        "lib/wrapper/miopen_enums.cc",
        "lib/wrapper/miopen_wrapper.cc",
        "lib/wrapper/nccl_wrapper.cc",
        "lib/wrapper/rccl_wrapper.cc",
        "lib/wrapper/rocblas_enums.cc",
        "lib/wrapper/rocblas_wrapper.cc",
        "lib/wrapper/rocsolver_wrapper.cc",
        "lib/wrapper/runtime_wrapper.cc",
        "lib/wrapper/solver_wrapper.cc",
        "lib/wrapper/wrapper_detail.h",
    ],
    hdrs = [
        "include/tfrt/gpu/wrapper/blas_wrapper.h",
        "include/tfrt/gpu/wrapper/ccl_types.h",
        "include/tfrt/gpu/wrapper/ccl_wrapper.h",
        "include/tfrt/gpu/wrapper/cublas_wrapper.h",
        "include/tfrt/gpu/wrapper/cuda_wrapper.h",
        "include/tfrt/gpu/wrapper/cudart_wrapper.h",
        "include/tfrt/gpu/wrapper/cudnn_wrapper.h",
        "include/tfrt/gpu/wrapper/cufft_wrapper.h",
        "include/tfrt/gpu/wrapper/cusolver_wrapper.h",
        "include/tfrt/gpu/wrapper/dense_map_utils.h",
        "include/tfrt/gpu/wrapper/dnn_wrapper.h",
        "include/tfrt/gpu/wrapper/driver_wrapper.h",
        "include/tfrt/gpu/wrapper/fft_wrapper.h",
        "include/tfrt/gpu/wrapper/hip_wrapper.h",
        "include/tfrt/gpu/wrapper/hipfft_wrapper.h",
        "include/tfrt/gpu/wrapper/miopen_wrapper.h",
        "include/tfrt/gpu/wrapper/nccl_wrapper.h",
        "include/tfrt/gpu/wrapper/rccl_wrapper.h",
        "include/tfrt/gpu/wrapper/rocblas_wrapper.h",
        "include/tfrt/gpu/wrapper/rocsolver_wrapper.h",
        "include/tfrt/gpu/wrapper/runtime_wrapper.h",
        "include/tfrt/gpu/wrapper/solver_wrapper.h",
        "include/tfrt/gpu/wrapper/wrapper.h",
    ],
    visibility = if_google(
        [
            ":tests_and_tools",
            ":xla_friends",
        ],
        ["//visibility:public"],
    ),
    deps = [
        ":rocm_stubs",
        "@cuda_headers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:support",
    ] + if_google(
        [
            # CUDA libraries are statically linked.
            "//platforms/gpus/cuda/dynamic_libcuda",
            "@cudnn_frontend",
            "@cuda_headers//:cublas_static",
            "@cuda_headers//:cudart_static",
            "@cuda_headers//:cufft_static",
            "@cuda_headers//:cusolver_static",
            "@cudnn_headers//:cudnn",
            "@nccl_headers//:nccl",
        ],
        [
            "@cudnn_frontend",
            "@cudnn_headers",
            "@nccl_headers",
            # This target is set to :cuda_stubs in TFRT's .bazelrc file.
            # For XLIR we use a different target based on stream_executor.
            "@rules_cuda//cuda:cuda_runtime",
        ],
    ),
)

tfrt_cc_library(
    name = "gpu_memory",
    srcs = [
        "lib/memory/bfc_gpu_allocator.cc",
    ],
    hdrs = [
        "include/tfrt/gpu/memory/bfc_gpu_allocator.h",
    ],
    visibility = [":tests_and_tools"],
    deps = [
        ":gpu_types",
        ":gpu_wrapper",
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
        ":gpu_types",
        ":gpu_wrapper",
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
        ":gpu_types",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
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
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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
        ":gpu_tensor",
        ":gpu_wrapper",
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
    local_defines = select({
        ":cuda_enabled": ["TFRT_GPU_CUDA_ENABLED"],
        "//conditions:default": [],
    }),
    visibility = [
        "@tf_runtime//:friends",
    ],
    deps = [
        ":gpu_device",
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_types",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:test_metadata_functions",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ] + select({
        ":cuda_enabled": [":test_cuda_kernels"],
        "//conditions:default": [],
    }),
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
    local_defines = select({
        ":cuda_enabled": ["TFRT_GPU_CUDA_ENABLED"],
        "//conditions:default": [],
    }),
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_wrapper",
        ":tf_gpu_matmul_op",
        ":tf_gpu_mlir_ops",
        ":tf_gpu_nullary_ops",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:tf_metadata_functions",
    ] + select({
        ":cuda_enabled": [
            ":tf_gpu_binary_ops",
            ":tf_gpu_dnn_ops",
            ":tf_gpu_pad_op",
            ":tf_gpu_reduction_ops",
            ":tf_gpu_transpose_op",
            ":tf_gpu_unary_ops",
        ],
        "//conditions:default": [],
    }),
)

tfrt_cc_library(
    name = "tf_gpu_nullary_ops",
    srcs = ["lib/ops/tf/nullary_ops.cc"],
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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
    deps = [
        "@tf_runtime//:core_runtime",
        ":gpu_op_handler",
        ":gpu_memory",
        ":gpu_tensor",
        ":gpu_wrapper",
        ":gpu_types",
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
)

tfrt_cc_library(
    name = "tf_gpu_mlir_ops",
    srcs = ["lib/ops/tf/mlir_ops.cc"],
    deps = [
        ":generated_kernel_images",
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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
    deps = [
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
        ":pad_op_noncuda",
        ":tf_gpu_dnn_ops_cu",
        ":tf_gpu_matmul_op",  # TODO(iga): For GEMM-calling utility only.
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//backends/common:shape_functions",
        "@tf_runtime//backends/common:tf_dnn_ops_util",
    ],
)

cuda_library(
    name = "tf_gpu_dnn_ops_cu",
    srcs = [
        "lib/ops/tf/dnn_ops.cu.cc",
    ],
    hdrs = ["lib/ops/tf/dnn_ops_cu.h"],
    deps = [
        ":gpu_memory",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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
    defines = if_oss([
        # TODO(b/161569340): Short-term fix. Remove
        "EIGEN_MUTEX=std::mutex",
        "EIGEN_MUTEX_LOCK=std::unique_lock<std::mutex>",
        "EIGEN_CONDVAR=std::condition_variable",
        "EIGEN_AVOID_STL_ARRAY",
    ]),
    deps = [
        ":gpu_device_eigen_support",
        ":gpu_memory",
        ":gpu_op_handler",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
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

td_library(
    name = "GpuOpBaseTdFile",
    srcs = [
        "include/tfrt/gpu/kernels/gpu_ops_base.td",
    ],
    includes = ["include"],
    visibility = if_google(
        [":xla_friends"],
        ["//visibility:public"],
    ),
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "gpu_opdefs_inc_gen",
    includes = [
        "../../include",
        "include",
    ],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/gpu/kernels/gpu_opdefs.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/gpu/kernels/gpu_opdefs.cpp.inc",
        ),
        (
            ["--gen-typedef-decls"],
            "include/tfrt/gpu/kernels/gpu_typedefs.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "include/tfrt/gpu/kernels/gpu_typedefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/gpu/kernels/gpu_ops.td",
    td_srcs = [
        "include/tfrt/gpu/kernels/gpu_blas_ops.td",
        "include/tfrt/gpu/kernels/gpu_ccl_ops.td",
        "include/tfrt/gpu/kernels/gpu_dnn_ops.td",
        "include/tfrt/gpu/kernels/gpu_driver_ops.td",
        "include/tfrt/gpu/kernels/gpu_fft_ops.td",
        "include/tfrt/gpu/kernels/gpu_solver_ops.td",
    ],
    deps = [
        ":GpuOpBaseTdFile",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "gpu_conversion_helper_opdefs_inc_gen",
    includes = [
        "../../include",
        "include",
    ],
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/tfrt/gpu/kernels/gpu_conversion_helper_opdefs.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/tfrt/gpu/kernels/gpu_conversion_helper_opdefs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/tfrt/gpu/kernels/gpu_conversion_helper_ops.td",
    deps = [
        "@llvm-project//mlir:GPUOpsTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

tfrt_cc_library(
    name = "gpu_opdefs",
    srcs = ["lib/kernels/gpu_ops.cc"],
    hdrs = ["include/tfrt/gpu/kernels/gpu_ops.h"],
    visibility = if_google(
        ["@tf_runtime//:friends"],
        ["//visibility:public"],
    ),
    deps = [
        ":gpu_conversion_helper_opdefs_inc_gen",
        ":gpu_opdefs_inc_gen",
        ":gpu_wrapper",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

tfrt_cc_library(
    name = "gpu_passes",
    srcs = [
        "lib/passes/gpu_async_patterns.cc",
        "lib/passes/gpu_to_tfrt_passes.cc",
        "lib/passes/set_entry_point.cc",
    ],
    hdrs = ["include/tfrt/gpu/passes/passes.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        ":gpu_entry_point",
        ":gpu_opdefs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
        "@tf_runtime//:test_kernels_opdefs",
    ],
)

tfrt_cc_library(
    name = "gpu_kernels_detail",
    hdrs = ["include/tfrt/gpu/kernels/kernels_detail.h"],
    visibility = if_google(
        [
            ":tests_and_tools",
            ":xla_friends",
        ],
        ["//visibility:public"],
    ),
    deps = [
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_library(
    name = "gpu_kernels",
    srcs = [
        "lib/kernels/blas_kernels.cc",
        "lib/kernels/ccl_kernels.cc",
        "lib/kernels/dnn_kernels.cc",
        "lib/kernels/driver_kernels.cc",
        "lib/kernels/fft_kernels.cc",
        "lib/kernels/solver_kernels.cc",
    ],
    alwayslink_static_registration_src = "lib/kernels/static_registration.cc",
    visibility = if_google(
        [
            ":tests_and_tools",
            ":xla_friends",
        ],
        ["//visibility:public"],
    ),
    deps = [
        ":gpu_kernels_detail",
        ":gpu_memory",
        ":gpu_tensor",
        ":gpu_types",
        ":gpu_wrapper",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//third_party/llvm_derived:raw_ostream",
    ],
)

tfrt_cc_binary(
    name = "tfrt_gpu_opt",
    srcs = ["tools/tfrt_gpu_opt/tfrt_gpu_opt.cc"],
    visibility = [":tests_and_tools"],
    deps = [
        ":gpu_opdefs",
        ":gpu_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:support",
        "@tf_runtime//:test_kernels_opdefs",
    ],
)

tfrt_cc_binary(
    name = "tfrt_gpu_translate",
    srcs = ["tools/tfrt_gpu_translate/static_registration.cc"],
    visibility = [":tests_and_tools"],
    deps = [
        ":gpu_opdefs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Translation",
        "@tf_runtime//:beftomlir_translate",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:mlirtobef_translate",
        "@tf_runtime//third_party/llvm_derived:tfrt_translate_main",
    ],
)

tfrt_cc_binary(
    name = "tfrt_gpu_executor",
    srcs = ["tools/tfrt_gpu_executor/main.cc"],
    visibility = [":tests_and_tools"],
    deps = [
        ":gpu_executor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//tools:bef_executor_lightweight_kernels",
    ],
)

tfrt_cc_library(
    name = "gpu_entry_point",
    hdrs = ["lib/gpu_entry_point.h"],
)

tfrt_cc_library(
    name = "gpu_executor",
    srcs = ["lib/gpu_executor.cc"],
    hdrs = ["include/tfrt/gpu/gpu_executor.h"],
    visibility = if_google(
        [":xla_friends"],
        ["//visibility:public"],
    ),
    deps = [
        ":gpu_entry_point",
        ":gpu_types",
        ":gpu_wrapper",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_library(
    name = "gpu_types",
    srcs = ["lib/gpu_types.cc"],
    hdrs = ["include/tfrt/gpu/gpu_types.h"],
    visibility = if_google(
        [":xla_friends"],
        ["//visibility:public"],
    ),
    deps = [
        ":gpu_wrapper",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)
