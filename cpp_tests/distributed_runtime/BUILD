load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

tfrt_cc_test(
    name = "distributed_context_test",
    srcs = ["distributed_context_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:distributed_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:remote_message_cc_proto",
        "@tf_runtime//:support",
        "@tf_runtime//cpp_tests:common",
    ],
)

tfrt_cc_test(
    name = "cluster_info_test",
    srcs = ["cluster_info_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:distributed_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//cpp_tests:common",
    ],
)

tfrt_cc_test(
    name = "callback_registry_test",
    srcs = ["callback_registry_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:distributed_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//cpp_tests:common",
    ],
)
