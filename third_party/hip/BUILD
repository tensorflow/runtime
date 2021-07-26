load("@tf_runtime//:build_defs.bzl", "if_google", "tfrt_cc_library")

licenses(["notice"])

tfrt_cc_library(
    name = "stub_inc",
    compatible_with = if_google(["//buildenv/target:non_prod"]),
    includes = ["."],
    textual_hdrs = glob(["*.inc"]),
    visibility = ["//visibility:public"],
)
