load(":expand_template.bzl", "expand_template")

exports_files(["LICENSE"])

_DNNL_RUNTIME_SEQ = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_SEQ",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_SEQ",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

expand_template(
    name = "dnnl_config_h",
    src = "include/dnnl_config.h.in",
    out = "include/dnnl_config.h",
    substitutions = _DNNL_RUNTIME_SEQ,
)

# Create the file dnnl_version.h with DNNL version numbers.
# Currently, the version numbers are hard coded here. If DNNL is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1) Automatically get the version numbers from CMakeLists.txt.

expand_template(
    name = "dnnl_version_h",
    src = "include/dnnl_version.h.in",
    out = "include/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "1",
        "@DNNL_VERSION_MINOR@": "6",
        "@DNNL_VERSION_PATCH@": "4",
        "@DNNL_VERSION_HASH@": "N/A",
    },
)

cc_library(
    name = "dnnl_single_threaded",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/*.cpp",
        "src/cpu/**/*.c",
        "src/cpu/**/*.cpp",
    ]) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
    ],
    copts = ["-fexceptions"],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/gemm/bf16",
        "src/cpu/gemm/f32",
        "src/cpu/gemm/s8x8s32",
        "src/cpu/rnn",
        "src/cpu/x64/jit_utils",
        "src/cpu/xbyak",
    ],
    textual_hdrs = glob([
        "include/*",
        "src/common/*.hpp",
        "src/cpu/*.hpp",
        "src/cpu/**/*.h",
        "src/cpu/**/*.hpp",
        "src/cpu/xbyak/*.h",
    ]),
    visibility = ["//visibility:public"],
)
