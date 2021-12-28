load("@tf_runtime//mlir_tests:lit.bzl", "glob_lit_tests")

licenses(["notice"])

# copybara:uncomment_begin(custom run_lit.sh driver is not supported in OSS)
# glob_lit_tests(
#     data = [":test_utilities"],
#     driver = "@tf_runtime//backends/cpu:tools/run_lit.sh",
#     test_file_exts = ["mlir"],
# )
#
# # Bundle together all of the test utilities that are used by tests.
# filegroup(
#     name = "test_utilities",
#     testonly = True,
#     srcs = [
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//mlir:run_lit.sh",
#         "@tf_runtime//backends/cpu:cpurt_opt",
#     ],
# )
# copybara:uncomment_end
