# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides a workspace macro to load dependent repositories."""

load("@tf_runtime//third_party:repo.bzl", "tfrt_http_archive")
load("@tf_runtime//third_party/eigen:workspace.bzl", eigen = "repo")
load("@tf_runtime//third_party/llvm:workspace.bzl", llvm = "repo")
load("@tf_runtime//third_party/cuda:dependencies.bzl", "cuda_dependencies")

def _rules_cuda_impl(repository_ctx):
    workspace = Label("@tf_runtime//third_party:rules_cuda/WORKSPACE")
    path = repository_ctx.path(workspace).dirname
    repository_ctx.symlink(path, "")

_rules_cuda = repository_rule(
    implementation = _rules_cuda_impl,
    local = True,
)

def tfrt_dependencies():
    """Loads TFRT external dependencies into WORKSPACE."""

    _rules_cuda(name = "rules_cuda")

    cuda_dependencies()

    llvm(name = "llvm-project")

    tfrt_http_archive(
        name = "bazel_skylib",
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        ],
    )

    eigen(name = "eigen_archive")

    tfrt_http_archive(
        name = "dnnl",
        build_file = "@tf_runtime//third_party/dnnl:BUILD",
        sha256 = "5369f7b2f0b52b40890da50c0632c3a5d1082d98325d0f2bff125d19d0dcaa1d",
        strip_prefix = "oneDNN-1.6.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v1.6.4.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v1.6.4.tar.gz",
        ],
    )

    tfrt_http_archive(
        name = "com_google_googletest",
        sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
        strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
            "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        ],
    )

    tfrt_http_archive(
        name = "com_github_google_benchmark",
        strip_prefix = "benchmark-16703ff83c1ae6d53e5155df3bb3ab0bc96083be",
        sha256 = "59f918c8ccd4d74b6ac43484467b500f1d64b40cc1010daa055375b322a43ba3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/benchmark/archive/16703ff83c1ae6d53e5155df3bb3ab0bc96083be.zip",
            "https://github.com/google/benchmark/archive/16703ff83c1ae6d53e5155df3bb3ab0bc96083be.zip",
        ],
    )

    tfrt_http_archive(
        name = "com_google_protobuf",
        sha256 = "bf0e5070b4b99240183b29df78155eee335885e53a8af8683964579c214ad301",
        strip_prefix = "protobuf-3.14.0",
        system_build_file = "@tf_runtime//third_party/systemlibs:protobuf.BUILD",
        system_link_files = {
            "@tf_runtime//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.14.0.zip",
            "https://github.com/protocolbuffers/protobuf/archive/v3.14.0.zip",
        ],
    )

    tfrt_http_archive(
        name = "cub_archive",
        build_file = "@tf_runtime//third_party:cub/BUILD",
        patch_file = "@tf_runtime//third_party:cub/pr170.patch",
        sha256 = "6bfa06ab52a650ae7ee6963143a0bbc667d6504822cbd9670369b598f18c58c3",
        strip_prefix = "cub-1.8.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.8.0.zip",
            "https://github.com/NVlabs/cub/archive/1.8.0.zip",
        ],
    )

    tfrt_http_archive(
        name = "zlib",
        build_file = "@tf_runtime//third_party:zlib.BUILD",
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = "@tf_runtime//third_party/systemlibs:zlib.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    tfrt_http_archive(
        name = "py-cpuinfo",
        strip_prefix = "py-cpuinfo-0.2.3",
        sha256 = "f6a016fdbc4e7fadf2d519090fcb4fa9d0831bad4e85245d938e5c2fe7623ca6",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/p/py-cpuinfo/py-cpuinfo-0.2.3.tar.gz",
            "https://pypi.python.org/packages/source/p/py-cpuinfo/py-cpuinfo-0.2.3.tar.gz",
        ],
        build_file = "@tf_runtime//third_party:py-cpuinfo.BUILD",
    )
