package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

filegroup(
    name = "header_files",
    srcs = glob(["*.h"]),
)

cc_library(
    name = "cudnn_headers",
    hdrs = [":header_files"],
    includes = ["."],  # Allow <angled> include.
)
