load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

glob_tfrt_lit_tests(
    data = [
        # Un-comment this line and download the BTF file to the specified location
        # in order to run resnet50_graph_inference.mlir.
        # "test_data/resnet50_graph_inference_tensors.btf",
        ":test_utilities",
    ],
    # Use size_override if needed.
    default_size = "large",
    default_tags = [
        "integration_test",
        "do_not_disable_rtti",
    ],
    exclude = [
        # Remove this line before running resnet50_graph_inference.mlir.
        "resnet50_graph_inference.mlir",
        "*.benchmarks.mlir",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "@llvm-project//llvm:FileCheck",
        "@tf_runtime//tools:bef_executor",
        "@tf_runtime//tools:bef_name",
    ],
)
