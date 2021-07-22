load("//:build_defs.bzl", "tfrt_cc_library")

licenses(["notice"])

[
    genrule(
        name = "%s_stub_inc" % api,
        srcs = [
            "//backends/gpu/tools/stub_codegen:%s.json" % api,
            "@cuda_headers//:header_files",
            "@cudnn_headers//:header_files",
            "@nccl_headers//:header_files",
        ],
        outs = ["include/%s_stub.cc.inc" % api],
        cmd = "$(location //backends/gpu/tools/stub_codegen:impl_codegen) " +
              "$(location //backends/gpu/tools/stub_codegen:%s.json) " % api +
              "$(OUTS)",
        tools = ["//backends/gpu/tools/stub_codegen:impl_codegen"],
    )
    for api in [
        "cublas",
        "cuda",
        "cudart",
        "cudnn",
        "cufft",
        "cufftxt",
        "cusolverdn",
        "nccl",
    ]
]

tfrt_cc_library(
    name = "stub_inc",
    textual_hdrs = [
        ":cublas_stub_inc",
        ":cuda_stub_inc",
        ":cudart_stub_inc",
        ":cudnn_stub_inc",
        ":cufft_stub_inc",
        ":cufftxt_stub_inc",
        ":cusolverdn_stub_inc",
        ":nccl_stub_inc",
    ],
    visibility = ["//visibility:public"],
)
