load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

licenses(["reciprocal"])

package(
    default_visibility = ["@tf_runtime//:friends"],
)

exports_files(["LICENSE"])

# These headers are intertwined with tf_runtime:hostcontext. Export them to
# be built as part of the hostcontext.
#
# Ideally we would split the hostcontext into a headers target and a library
# target. However, there is not really a good solution to re-export the headers
# from the library target, and downstream targets would need to depend on both.
filegroup(
    name = "concurrent_work_queue_hdrs",
    srcs = [
        "lib/blocking_work_queue.h",
        "lib/event_count.h",
        "lib/non_blocking_work_queue.h",
        "lib/task_deque.h",
        "lib/task_priority_deque.h",
        "lib/task_queue.h",
        "lib/work_queue_base.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

filegroup(
    name = "concurrent_work_queue_srcs",
    srcs = [
        "lib/multi_threaded_work_queue.cc",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

tfrt_cc_test(
    name = "cpp_tests/blocking_work_queue_test",
    srcs = [
        "cpp_tests/blocking_work_queue_test.cc",
        ":concurrent_work_queue_hdrs",
    ],
    includes = ["lib"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/multi_threaded_work_queue_test",
    srcs = [
        "cpp_tests/multi_threaded_work_queue_test.cc",
        ":concurrent_work_queue_hdrs",
    ],
    includes = ["lib"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/non_blocking_work_queue_test",
    srcs = [
        "cpp_tests/non_blocking_work_queue_test.cc",
        ":concurrent_work_queue_hdrs",
    ],
    includes = ["lib"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/task_deque_test",
    srcs = [
        "cpp_tests/task_deque_test.cc",
        ":concurrent_work_queue_hdrs",
    ],
    includes = ["lib"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/task_priority_deque_test",
    srcs = [
        "cpp_tests/task_priority_deque_test.cc",
        ":concurrent_work_queue_hdrs",
    ],
    includes = ["lib"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tfrt_cc_test(
    name = "cpp_tests/task_queue_test",
    srcs = [
        "cpp_tests/task_queue_test.cc",
        ":concurrent_work_queue_hdrs",
    ],
    includes = ["lib"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)
