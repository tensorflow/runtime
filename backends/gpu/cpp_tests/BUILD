load(
    "@tf_runtime//:build_defs.bzl",
    "if_google",
    "tfrt_cc_library",
    "tfrt_cc_test",
)

licenses(["notice"])

tfrt_cc_library(
    name = "common",
    testonly = True,
    hdrs = ["include/common.h"],
    deps = [
        "@tf_runtime//backends/gpu:gpu_wrapper",
        "@tf_runtime//cpp_tests:common",
    ],
)

[
    tfrt_cc_test(
        name = "wrapper/%s_wrapper_test" % name,
        srcs = [
            "instantiate_suite.cc",
            "wrapper/%s_wrapper_test.cc" % name,
        ],
        # Skip ROCm tests by default for now. TODO(csigg): make configurable.
        args = ["--%s_filter=*CUDA" % if_google("gunit", "gtest")],
        tags = [
            "noasan",
            "nomsan",
            "requires-gpu-nvidia",
        ],
        deps = [
            ":common",
            "@com_google_googletest//:gtest_main",
            "@tf_runtime//backends/gpu:gpu_wrapper",
        ] + if_google(["@com_google_absl//absl/debugging:leak_check"]),
    )
    for name in [
        "blas",
        "ccl",
        "driver",
        "fft",
        "runtime",
        "solver",
    ]
]

tfrt_cc_test(
    name = "wrapper/dnn_wrapper_test",
    srcs = [
        "instantiate_suite.cc",
        "wrapper/dnn_wrapper_test.cc",
    ],
    # Skip ROCm tests by default for now. TODO(csigg): make configurable.
    args = ["--%s_filter=*CUDA" % if_google("gunit", "gtest")],
    env = {"CUDNN_LOGDEST_DBG": "tfrt"},
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-sm70",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ] + if_google(["@com_google_absl//absl/debugging:leak_check"]),
)

tfrt_cc_test(
    name = "gpu_types_test",
    srcs = [
        "gpu_types_test.cc",
        "instantiate_suite.cc",
    ],
    # Skip ROCm tests by default for now. TODO(csigg): make configurable.
    args = ["--%s_filter=*CUDA" % if_google("gunit", "gtest")],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//backends/gpu:gpu_executor",
        "@tf_runtime//backends/gpu:gpu_types",
        "@tf_runtime//backends/gpu:gpu_wrapper",
    ],
)

tfrt_cc_test(
    name = "work_queue_test",
    srcs = [
        "work_queue_test.cc",
    ],
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//backends/gpu:gpu_executor",
    ],
)
