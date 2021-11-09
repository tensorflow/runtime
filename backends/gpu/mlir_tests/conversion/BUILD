load("@tf_runtime//:build_defs.bzl", "if_google")
load("@tf_runtime//mlir_tests:lit.bzl", "glob_lit_tests")

licenses(["notice"])

glob_lit_tests(
    cfgs = "@tf_runtime//mlir_tests:litcfgs",  # copybara:comment
    data = [":test_utilities"],
    # copybara:uncomment driver = "@tf_runtime//backends/gpu/mlir_tests:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "@llvm-project//llvm:FileCheck",
        "@tf_runtime//backends/gpu:tfrt_gpu_opt",
    ] + if_google([
        "@llvm-project//mlir:run_lit.sh",
    ]),
)
