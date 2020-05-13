load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

licenses(["notice"])

package(
    default_visibility = ["@tf_runtime//:__subpackages__"],
)

exports_files([
    "LICENSE",
])

tfrt_cc_library(
    name = "ostream",
    srcs = ["lib/ostream.cc"],
    hdrs = ["include/llvm_derived/ostream.h"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//llvm:support",
    ],
)

tfrt_cc_library(
    name = "tfrt_translate_main",
    srcs = ["tools/tfrt_translate/main.cc"],
    visibility = ["@tf_runtime//:friends"],
    deps = [
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)
