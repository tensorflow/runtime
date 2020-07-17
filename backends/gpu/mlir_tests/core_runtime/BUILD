load("@tf_runtime//mlir_tests:lit.bzl", "glob_lit_tests")

licenses(["notice"])

# GPU tests
glob_lit_tests(
    data = [":test_utilities"],
    default_tags = ["requires-gpu-nvidia"],
    #=== GOOGLE_PIPER: tf_runtime/mlir_tests:run_lit.sh ===#
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
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
        #=== GOOGLE_PIPER: llvm-project/mlir:run_lit.sh ===#
        "@tf_runtime//tools:bef_executor",
        "@tf_runtime//tools:tfrt_opt",
        "@tf_runtime//tools:tfrt_translate",
    ],
)
