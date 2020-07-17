load("//cuda:defs.bzl", "cuda_library")
load(":defs.bzl", "cuda_library_test", "cuda_runtime_test", "cuda_targets_test")

cuda_targets_test(
    name = "cuda_targets_test",
    target_under_test = "//cuda:cuda_targets",
)

cc_library(
    name = "cuda_test_runtime",
    hdrs = ["cuda.h"],
)

cuda_runtime_test(
    name = "cuda_runtime_test",
    target_under_test = "//cuda:cuda_runtime",
)

cuda_library(
    name = "cuda_test_library",
    srcs = ["test.cu.cc"],
    tags = ["manual"],
)

cuda_library_test(
    name = "cuda_library_test",
    target_under_test = ":cuda_test_library",
)

test_suite(
    name = "rules_cuda_test_suite",
    tests = [
        ":cuda_library_test",
        ":cuda_runtime_test",
        ":cuda_targets_test",
    ],
)
