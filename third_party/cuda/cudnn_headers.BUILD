package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "header_files",
    srcs = glob(["cudnn/*.h"]),
)

cc_library(
    name = "cudnn_headers",
    hdrs = [":header_files"],
    includes = ["cudnn"],
)
