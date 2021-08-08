load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

package(default_visibility = [
    "@tf_runtime//:__subpackages__",
    # copybara:uncomment "//third_party/tf_runtime_google:__subpackages__",
])

exports_files([
    "matmul.mlir",
])

glob_tfrt_lit_tests(
    data = [
        ":test_utilities",
        "@tf_runtime//backends/cpu/mlir_tests/mnist/test_data:matmul_test_f32.btf",
        "@tf_runtime//backends/cpu/mlir_tests/mnist/test_data:matmul_test_i32.btf",
        "@tf_runtime//backends/cpu/mlir_tests/mnist/test_data:test_tensor.btf",
    ],
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
