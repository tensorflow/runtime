load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

licenses(["notice"])

package(
    default_visibility = [
        "@tf_runtime//:friends",
        "@tf_runtime//third_party/llvm_derived/cpp_tests:__subpackages__",
    ],
)

exports_files(["LICENSE"])

tfrt_cc_library(
    name = "raw_ostream",
    srcs = ["lib/Support/raw_ostream.cpp"],
    hdrs = ["include/llvm_derived/Support/raw_ostream.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:config",
    ],
)

tfrt_cc_library(
    name = "in_place",
    hdrs = ["include/llvm_derived/Support/in_place.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

tfrt_cc_library(
    name = "unique_any",
    hdrs = ["include/llvm_derived/Support/unique_any.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":in_place",
    ],
)

tfrt_cc_library(
    name = "tfrt_translate_main",
    srcs = ["tools/mlir-translate/mlir-translate.cpp"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
