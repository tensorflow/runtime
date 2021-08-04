load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

glob_tfrt_lit_tests(
    data = [":test_utilities"],
    no_bef_translation = ["kernel_errors.mlir"],
    size_override = {"parallel.mlir": "medium"},
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "@llvm-project//llvm:FileCheck",
        "@tf_runtime//tools:bef_executor",
        "@tf_runtime//tools:tfrt_opt",
    ],
)
