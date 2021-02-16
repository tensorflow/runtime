"""Rules to add a 'cuda' feature to clang and nvcc toolchains."""

load(":defs.bzl", "CudaTargetsInfo")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "feature",
    "flag_group",
    "flag_set",
    "tool",
    "with_feature_set",
)
load("@local_cuda//:defs.bzl", "if_local_cuda")

_compile_actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile]

def _cuda_toolchain_config(features, action_configs):
    return struct(features = features, action_configs = action_configs)

def _clang_cuda_toolchain_config(cuda_targets):
    flags = ["--language=cuda", "--cuda-path=external/local_cuda/cuda"]
    for cuda_target in cuda_targets:
        flags.append("--cuda-gpu-arch=%s" % cuda_target)

    return _cuda_toolchain_config(
        features = [feature(
            name = "cuda",
            flag_sets = [flag_set(
                actions = _compile_actions,
                flag_groups = [flag_group(flags = flags)],
            )],
        )],
        action_configs = [],
    )

def _nvcc_cuda_toolchain_config(cuda_targets, nvcc, compiler_path):
    flags = [
        "--x=cu",
        "--compiler-bindir=" + compiler_path,
        "--forward-unknown-to-host-compiler",
    ]
    gencode_format = "--generate-code=arch=compute_{0},code={1}_{0}"
    for cuda_target in cuda_targets:
        flags.append(gencode_format.format(cuda_target[3:], "compute"))
        flags.append(gencode_format.format(cuda_target[3:], "sm"))

    return _cuda_toolchain_config(
        features = [feature(
            name = "cuda",
            flag_sets = [flag_set(
                actions = _compile_actions,
                flag_groups = [flag_group(flags = flags)],
            )],
        )],
        action_configs = [
            action_config(
                action_name = ACTION_NAMES.cpp_compile,
                tools = [
                    tool(
                        tool = nvcc,
                        with_features = [with_feature_set(features = ["cuda"])],
                    ),
                    tool(path = compiler_path),
                ],
            ),
        ],
    )

def cuda_toolchain_config(cuda_toolchain_info, compiler_path):
    """Returns features and action configs to include in cc_toolchain_config.

    Args:
      cuda_toolchain_info: instance of @rules_cuda//cuda:cuda_toolchain_info.
      compiler_path: path to host compiler.

    Returns:
      Struct with features and action_configs to add to cc_toolchain_config.
    """
    if if_local_cuda(False, True):
        return _cuda_toolchain_config(features = [], action_configs = [])
    cuda_targets = cuda_toolchain_info[CudaTargetsInfo].cuda_targets
    compiler = cuda_toolchain_info[BuildSettingInfo].value
    if compiler == "clang":
        return _clang_cuda_toolchain_config(cuda_targets)
    nvcc = cuda_toolchain_info.files.to_list()[0]
    return _nvcc_cuda_toolchain_config(cuda_targets, nvcc, compiler_path)

def cuda_compiler_deps():
    return if_local_cuda(["@local_cuda//:compiler_deps"])
