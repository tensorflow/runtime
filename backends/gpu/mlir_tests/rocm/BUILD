load("@tf_runtime//:build_defs.bzl", "if_google")
load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

glob_tfrt_lit_tests(
    data = [":test_utilities"],
    default_tags = [if_google("notap", "requires-gpu-amd")],
    # copybara:uncomment driver = "@tf_runtime//backends/gpu/mlir_tests:run_lit.sh",
    tfrt_translate = "@tf_runtime//backends/gpu:tfrt_gpu_translate",
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "@llvm-project//llvm:FileCheck",
        "@tf_runtime//backends/gpu:tfrt_gpu_executor",
        "@tf_runtime//backends/gpu:tfrt_gpu_opt",
        "@tf_runtime//backends/gpu:tfrt_gpu_translate",
        "@tf_runtime//tools:bef_executor_lite",
    ],
)
