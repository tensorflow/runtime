load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["notice"])

[
    tfrt_cc_test(
        name = "wrapper/%s_wrapper_test" % name,
        srcs = [
            "wrapper/%s_wrapper_test.cc" % name,
            "wrapper/common.h",
        ] + (
            [] if name == "runtime" or name == "cufft" else [
                "wrapper/instantiate_suite.cc",
            ]
        ),
        env = {
            "CUDNN_LOGDEST_DBG": "tfrt",
            "CUDNN_LOGINFO_DBG": "1",
        } if name == "dnn" else {},
        tags = [
            "noasan",
            "nomsan",
            "requires-gpu-nvidia",
        ],
        deps = [
            "@llvm-project//llvm:Support",
            "@tf_runtime//backends/gpu:gpu_wrapper",
            "@tf_runtime//cpp_tests:common",
            "@tf_runtime//:support",
            "@com_google_googletest//:gtest_main",
        ] + select({
            "@tf_runtime//:is_build_env_google": ["//third_party/absl/debugging:leak_check"],
            "//conditions:default": [],
        }),
    )
    for name in [
        "blas",
        "dnn",
        "driver",
        "cufft",
        "runtime",
        "solver",
    ]
]

tfrt_cc_test(
    name = "memory/gpu_buffer_test",
    srcs = [
        "memory/gpu_buffer_test.cc",
    ],
    tags = [
        "noasan",
        "nomsan",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//backends/gpu:gpu_types",
        "@tf_runtime//backends/gpu:gpu_wrapper",
        "@tf_runtime//cpp_tests:common",
    ],
)
