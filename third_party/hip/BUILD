load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

licenses(["notice"])

tfrt_cc_library(
    name = "stub_inc",
    includes = ["."],
    textual_hdrs = glob(["*.inc"]),
    visibility = ["//visibility:public"],
)
