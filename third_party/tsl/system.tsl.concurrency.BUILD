load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "concurrency_async_value",
    srcs = [
        "tsl/concurrency/async_value.cc",
        "tsl/concurrency/async_value_ref.cc",
    ],
    hdrs = [
        "tsl/concurrency/async_value.h",
        "tsl/concurrency/async_value_ref.h",
        "tsl/concurrency/chain.h",
    ],
    deps = [
        ":concurrency_concurrent_vector",
        ":concurrency_ref_count",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "concurrency_concurrent_vector",
    hdrs = ["tsl/concurrency/concurrent_vector.h"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "concurrency_ref_count",
    hdrs = ["tsl/concurrency/ref_count.h"],
)
