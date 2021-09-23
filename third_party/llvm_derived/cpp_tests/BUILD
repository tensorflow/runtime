load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "Support/unique_any_test",
    srcs = [
        "Support/unique_any_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//third_party/llvm_derived:unique_any",
    ],
)
