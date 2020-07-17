load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu.cc"],
    hdrs = ["kernel.h"],
    visibility = ["//:__subpackages__"],
)

cc_library(
    name = "no_cuda",
    srcs = ["no_cuda.cc"],
    hdrs = ["kernel.h"],
)

# Run with e.g. 'bazel run --config=cuda //examples:hello_cuda'
cc_binary(
    name = "hello_cuda",
    srcs = ["main.cc"],
    deps = select({
        "@rules_cuda//cuda:cuda_enabled": [":kernel"],
        "//conditions:default": ["no_cuda"],
    }),
)
