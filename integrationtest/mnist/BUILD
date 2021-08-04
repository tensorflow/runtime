load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

licenses(["notice"])

glob_tfrt_lit_tests(
    data = [
        "test_data/mnist_metadata.btf",
        "test_data/mnist_tensors.btf",
        ":test_utilities",
    ],
    # Use size_override if needed.
    default_size = "large",
    default_tags = ["integration_test"],
)

# copybara:uncomment_begin
# # pickle to btf conversion is internal.
# genrule(
#     name = "format_mnist_btf",
#     srcs = ["test_data/mnist_tensors.pickle"],
#     outs = ["mnist_tensors.btf"],
#     cmd = """
#     $(location //third_party/tf_runtime/utils/mnist:convert_from_pickle_to_btf.par) \
#     --output_file $@ --input_files $(SRCS)""",
#     exec_tools = ["@tf_runtime//utils/mnist:convert_from_pickle_to_btf.par"],
# )
# copybara:uncomment_end

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "@llvm-project//llvm:FileCheck",
        "@tf_runtime//tools:bef_executor",
    ],
)
