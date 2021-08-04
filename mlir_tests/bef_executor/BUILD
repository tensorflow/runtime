load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

glob_tfrt_lit_tests(data = [":test_utilities"])

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@tf_runtime//tools:bef_executor_lite",
        "@tf_runtime//tools:tfrt_opt",
    ],
)
