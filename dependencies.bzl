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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@tf_runtime//third_party:repo.bzl", "tfrt_http_archive")

def tfrt_dependencies():
    """Loads TFRT external dependencies into WORKSPACE."""
    http_archive(
        name = "rules_cc",
        sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6",
        strip_prefix = "rules_cc-0.1.5",
        url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz",
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
