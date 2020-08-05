load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library", "tfrt_cc_test")

licenses(["reciprocal"])

package(
    default_visibility = ["@tf_runtime//:friends"],
)

exports_files(["LICENSE"])

filegroup(
    name = "concurrent_work_queue_srcs",
    srcs = [
        "lib/blocking_work_queue.h",
        "lib/environment.h",
        "lib/event_count.h",
        "lib/multi_threaded_work_queue.cc",
        "lib/non_blocking_work_queue.h",
        "lib/task_deque.h",
        "lib/task_priority_deque.h",
        "lib/task_queue.h",
        "lib/work_queue_base.h",
    ] + select({
        "//conditions:default": ["lib/environment_std.h"],
    }),
)

tfrt_cc_library(
    name = "concurrent_work_queue",
    hdrs = [":concurrent_work_queue_srcs"],
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//conditions:default": [],
    }),
)

tfrt_cc_test(
    name = "cpp_tests/blocking_work_queue_test",
    srcs = [
        "cpp_tests/blocking_work_queue_test.cc",
    ],
    includes = ["lib"],
    deps = [
        ":concurrent_work_queue",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/multi_threaded_work_queue_test",
    srcs = [
        "cpp_tests/multi_threaded_work_queue_test.cc",
    ],
    includes = ["lib"],
    deps = [
        ":concurrent_work_queue",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/non_blocking_work_queue_test",
    srcs = ["cpp_tests/non_blocking_work_queue_test.cc"],
    includes = ["lib"],
    deps = [
        ":concurrent_work_queue",
        "//testing/base/public:gunit_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/task_deque_test",
    srcs = ["cpp_tests/task_deque_test.cc"],
    includes = ["lib"],
    deps = [
        ":concurrent_work_queue",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/task_priority_deque_test",
    srcs = ["cpp_tests/task_priority_deque_test.cc"],
    includes = ["lib"],
    deps = [
        ":concurrent_work_queue",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/task_queue_test",
    srcs = ["cpp_tests/task_queue_test.cc"],
    includes = ["lib"],
    deps = [
        ":concurrent_work_queue",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
    ],
)
