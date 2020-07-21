load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

# GPU tests
glob_tfrt_lit_tests(
    data = [":test_utilities"],
    default_tags = ["requires-gpu-nvidia"],
    tags_override = {
        "tf_biasadd.benchmark.mlir": ["manual"],
        "tf_conv2d.benchmark.mlir": ["manual"],
        "tf_fusedbatchnorm.benchmark.mlir": ["manual"],
        "tf_reduction.benchmark.mlir": ["manual"],
        "tf_relu.benchmark.mlir": ["manual"],
        "multi_gpu.mlir": [
            "manual",
            "notap",
        ],
    },
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
        "@tf_runtime//tools:bef_executor",
        "@tf_runtime//tools:bef_name",
        "@tf_runtime//tools:tfrt_opt",
    ],
)
