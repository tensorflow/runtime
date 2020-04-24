load("@tf_runtime//mlir_tests:lit.bzl", "glob_lit_tests")

licenses(["notice"])

package(default_visibility = [
    "@tf_runtime//:__subpackages__",
])

exports_files([
    "matmul.mlir",
])

glob_lit_tests(
    data = [
        ":test_utilities",
        "@tf_runtime//backends/cpu/mlir_tests/mnist/test_data:matmul_test_f32.btf",
        "@tf_runtime//backends/cpu/mlir_tests/mnist/test_data:matmul_test_i32.btf",
        "@tf_runtime//backends/cpu/mlir_tests/mnist/test_data:test_tensor.btf",
    ],
    #=== GOOGLE_PIPER: tf_runtime/mlir_tests:run_lit.sh ===#
    test_file_exts = [
        "mlir",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
        #=== GOOGLE_PIPER: llvm-project/mlir:run_lit.sh ===#
        "@tf_runtime//tools:bef_executor",
        "@tf_runtime//tools:tfrt_translate",
    ],
)
