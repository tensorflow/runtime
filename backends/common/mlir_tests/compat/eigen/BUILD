load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

package(default_visibility = ["@tf_runtime//:__subpackages__"])

exports_files([
    "conv2d.bias.benchmarks.mlir",
    "matmul.benchmarks.mlir",
    "max_pooling.benchmarks.mlir",
])

glob_tfrt_lit_tests(
    data = [
        "test_data/batch_norm_f32.btf",
        "test_data/batch_norm_grad_f32.btf",
        "test_data/conv2d_batch_norm_f32.btf",
        "test_data/conv2d_bias_f32.btf",
        "test_data/conv2d_grad_filter_f32.btf",
        "test_data/conv2d_grad_input_f32.btf",
        "test_data/matmul_f32.btf",
        "test_data/matmul_i32.btf",
        "test_data/max_pooling_f32.btf",
        ":test_utilities",
    ],
    exclude = glob(["*.benchmarks.mlir"]),
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "@llvm-project//llvm:FileCheck",
        "@tf_runtime//tools:bef_executor",
    ],
)
