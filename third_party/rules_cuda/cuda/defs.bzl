"""CUDA rules for Bazel."""

load("@local_cuda//:defs.bzl", "cuda_path")

cuda_targets = [
    "sm_30",
    "sm_32",
    "sm_35",
    "sm_37",
    "sm_50",
    "sm_52",
    "sm_53",
    "sm_60",
    "sm_61",
    "sm_62",
    "sm_70",
    "sm_72",
    "sm_75",
]

CudaInfo = provider(fields = ["value"])

def cuda_library(name, copts = [], deps = [], features = [], exec_compatible_with = [], **kwargs):
    """Macro wrapping a cc_library which can contain CUDA device code.

    Args:
      name: forwarded to cc_library.
      copts: forwarded to cc_library.
      deps: forwarded to cc_library.
      features: forwarded to cc_library.
      exec_compatible_with: forwarded to cc_library.
      **kwargs: forwarded to cc_library.
    """

    cuda_copts = ["-x", "cuda", "--cuda-path=%s" % cuda_path]
    for cuda_target in cuda_targets:
        cuda_copts += select({
            "@rules_cuda//cuda:cuda_target_%s_enabled" % cuda_target: [
                "--cuda-gpu-arch=%s " % cuda_target,
            ],
            "//conditions:default": [],
        })

    cuda_features = ["-use_header_modules"]
    if kwargs.get("textual_hdrs", None):
        features.extend(["-layering_check", "-parse_headers"])

    native.cc_library(
        name = name,
        copts = copts + cuda_copts,
        deps = deps + ["@rules_cuda//cuda:cuda_runtime"],
        features = features + cuda_features,
        exec_compatible_with = exec_compatible_with + [
            "@rules_cuda//cuda:requires_local_cuda",
            "@rules_cuda//cuda:requires_cuda_targets",
            # A clang compiler is required to interpret the copts above.
            # The auto-detected local toolchain does not set this constraint
            # value though, and this repository doesn't want to get into the
            # business of defining and registring toolchains.
            # "@bazel_tools//tools/cpp:clang",
        ],
        **kwargs
    )
