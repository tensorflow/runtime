package(default_visibility = ["//third_party/cuda:__pkg__"])

exports_files(srcs = glob(["*.json"]))

# Clang libTool to generate HIP stub headers.
py_binary(
    name = "header_codegen",
    srcs = ["header_codegen.py"],
    python_version = "PY3",
)

# Clang libTool to generate CUDA/HIP stub libraries which dynamically load the
# genuine libraries and forward the call.
py_binary(
    name = "impl_codegen",
    srcs = ["impl_codegen.py"],
    python_version = "PY3",
)
