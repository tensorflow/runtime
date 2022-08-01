load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

# TODO(b/240450920): We are in the process of migrating JitRt backend to XLA
# and while we are doing this we can't keep it buildable/testable in OSS.

# copybara:uncomment_begin
#
# # CPU tests
# glob_tfrt_lit_tests(
#     data = [":test_utilities"],
# )
#
# # Bundle together all of the test utilities that are used by tests.
# filegroup(
#     name = "test_utilities",
#     testonly = True,
#     srcs = [
#         "@llvm-project//llvm:FileCheck",
#         "@tf_runtime//tools:bef_executor",
#     ],
# )
#
# copybara:uncomment_end
