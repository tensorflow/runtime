load(":defs.bzl", "cuda_targets")
load("//private:build.bzl", "bool_setting", "cuda_targets_flag", "local_cuda_path")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@local_cuda//:defs.bzl", "cuda_path")

package(default_visibility = ["//visibility:public"])

# Config setting that is intended to be used in select() statements to
# configure targets based on whether a it should be built with CUDA support.
#
# This config setting is not used in this repository, but is rather intended
# as a central switch for different unrelated users.
#
# Set with --define=enable_cuda
config_setting(
    name = "cuda_enabled",
    define_values = {"enable_cuda": "1"},
)

# Command line flag to specify the list of CUDA compute architectures to
# compile for. For details, please consult the '--cuda-gpu-arch' clang flag.
#
# Provides CudaInfo of the list of cuda_targets to build.
#
# Example usage: --@rules_cuda//cuda:cuda_targets=sm_60,sm_70
cuda_targets_flag(
    name = "cuda_targets",
    build_setting_default = ["sm_52"],
)

# Command line flag to specify the CUDA runtime. Use this target as CUDA
# runtime dependency.
#
# This target is implicitly added as a dependency to cuda_library() targets.
#
# Example usage: --@rules_cuda//cuda:cuda_runtime=@local_cuda//:cuda_runtime
label_flag(
    name = "cuda_runtime",
    build_setting_default = "@local_cuda//:cuda_runtime_static",
)

# Config setting to select() targets from @local_cuda//.
#
# Do not use as a condition whether your target should support CUDA, use
# :cuda_enabled instead. The intended use of this config setting is to verify
# that the configuration found the local CUDA toolkit. See :requires_local_cuda
# for an example.
config_setting(
    name = "local_cuda_exists",
    flag_values = {":_local_cuda_exists": "True"},
)

# Config setting whether building cuda_target (sm_xy) has been requested.
[
    config_setting(
        name = "cuda_target_%s_enabled" % cuda_target,
        flag_values = {":cuda_targets": cuda_target},
    )
    for cuda_target in cuda_targets
]

# Config setting whether building any cuda_targets has been requested.
#
# Do not use as a condition whether your target should support CUDA, use
# :cuda_enabled instead. The intended use of this config setting is to verify
# that at least one target has been requested. See :requires_cuda_targets for
# an example.
selects.config_setting_group(
    name = "cuda_target_any_enabled",
    match_any = [
        "cuda_target_%s_enabled" % cuda_target
        for cuda_target in cuda_targets
    ],
)

# Contraint value verifying that @local_cuda exists.
alias(
    name = "requires_local_cuda",
    actual = select(
        {":local_cuda_exists": ":_valid"},
        no_match_error = "CUDA toolkit not found, use '--repo_env=CUDA_PATH=<path>'.",
    ),
)

# Contraint value verifying that builing any cuda_targets has been requested.
alias(
    name = "requires_cuda_targets",
    actual = select(
        {":cuda_target_any_enabled": ":_valid"},
        no_match_error = "No CUDA targets specified, use '--client_env=CUDA_TARGETS=sm_xy,...'.",
    ),
)

# Provides CudaInfo of the path to @local_cuda//cuda.
local_cuda_path(
    name = "local_cuda_path",
)

# Helper for :local_cuda_exists
bool_setting(
    name = "_local_cuda_exists",
    build_setting_default = bool(cuda_path),
    visibility = ["//visibility:private"],
)

# Helper for requires_* targets above to check configuration.
constraint_setting(
    name = "_constraint",
    default_constraint_value = ":_valid",
    visibility = ["//visibility:private"],
)

constraint_value(
    name = "_valid",
    constraint_setting = ":_constraint",
)
