package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "header_files",
    srcs = glob(["cuda/**"]),
)

cc_library(
    name = "cuda_headers",
    hdrs = [":header_files"],
    includes = ["cuda"],
)
