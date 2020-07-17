# [CUDA](http://nvidia.com/cuda) rules for [Bazel](https://github.com/bazelbuild/bazel)

The `@rules_cuda` repository primarily contains a `cuda_library` macro which
allows compiling a C++ bazel target containing CUDA device code using clang.

A secondary `@local_cuda` repository contains bazel targets for the CUDA toolkit
installed on the execution machine.

## Setup

Add the following snippet to your `WORKSPACE` file:

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_cuda",
    urls = ["https://github.com/bazelbuild/rules_cuda/archive/????.zip"],
    sha256 = "????",
)
load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")
rules_cuda_dependencies()
```

## Using `cuda_library`

Then, in the `BUILD` and/or `*.bzl` files in your own workspace, you can create
C++ targets containing CUDA device code with the `cuda_library` macro:

```python
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu"],
)
```

## Building

The normal `bazel build` command allows configuring a few properties of the
`@rules_cuda` repository.

*   `--define=enable_cuda=1`: triggers the `cuda_enabled` config setting. This
    config setting is not used in this repository, but is rather intended as a
    central switch to let users know whether building with CUDA support has been
    requested.
*   `--@rules_cuda//cuda:cuda_targets=sm_xy,...`: configures the list of CUDA
    compute architectures to compile for as a comma-separated list. The default
    is `"sm_52"`. For details, please consult the
    [--cuda-gpu-arch](https://llvm.org/docs/CompileCudaWithLLVM.html#invoking-clang)
    clang flag.
*   `--@rules_cuda//cuda:cuda_runtime=<label>`: configures the CUDA runtime
    target. The default is `"@local_cuda//:cuda_runtime_static"`. This target is
    implicitly added as a dependency to cuda_library() targets.
    `@rules_cuda//cuda:cuda_runtime` should be used as the CUDA runtime target
    everywhere so that the actual runtime can be configured in a central place.
*   `--repo_env=CUDA_PATH=<path>`: Specifies the path to the locally installed
    CUDA toolkit. The default is `"/usr/local/cuda"`.
*   `--client_env=CC=clang`: Triggers bazel's auto-configured local toolchain to
    use clang.
