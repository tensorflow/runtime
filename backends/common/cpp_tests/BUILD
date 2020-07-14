load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "bcast_test",
    srcs = ["bcast_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/common:tf_bcast",
    ],
)
