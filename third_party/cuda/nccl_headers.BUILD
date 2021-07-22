package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "header_files",
    srcs = glob(["nccl/src/*.h"]),
)

cc_library(
    name = "nccl_headers",
    hdrs = [":header_files"],
    includes = ["nccl"],
)
