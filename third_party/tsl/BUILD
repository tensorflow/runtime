package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "concurrency/async_value",
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
        ":concurrency/concurrent_vector",
        ":concurrency/ref_count",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "concurrency/concurrent_vector",
    hdrs = ["tsl/concurrency/concurrent_vector.h"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "concurrency/ref_count",
    hdrs = ["tsl/concurrency/ref_count.h"],
)
