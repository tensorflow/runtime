package(default_visibility = ["//third_party/cuda:__pkg__"])

exports_files(srcs = glob(["*.json"]))

# Clang libTool to generate CUDA stub libraries which dynamically load the
# genuine libraries and forward the call.
py_binary(
    name = "stub_codegen",
    srcs = ["stub_codegen.py"],
    python_version = "PY3",
)
