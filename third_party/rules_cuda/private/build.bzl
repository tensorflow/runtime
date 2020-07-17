"""Private code for CUDA rules."""

load("//cuda:defs.bzl", "CudaInfo", "cuda_targets")
load("@local_cuda//:defs.bzl", "cuda_path")

def _cuda_targets_flag_impl(ctx):
    for cuda_target in ctx.build_setting_value:
        if cuda_target not in cuda_targets:
            fail("%s is not a supported %s value." % (cuda_target, ctx.label))
    return CudaInfo(value = ctx.build_setting_value)

cuda_targets_flag = rule(
    implementation = _cuda_targets_flag_impl,
    build_setting = config.string_list(flag = True),
)

def _local_cuda_path_impl(ctx):
    return CudaInfo(value = cuda_path)

local_cuda_path = rule(implementation = _local_cuda_path_impl)

def _bool_setting_impl(ctx):
    pass  # Never change the default value.

bool_setting = rule(
    implementation = _bool_setting_impl,
    build_setting = config.bool(),
)
