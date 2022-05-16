"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "6f5f847a36af2f2e9dfce4ed99f1a243161c7d5d"
    LLVM_SHA256 = "72cf1df6f0049350035a90aaf2c18e884ee42cd03e8b94d2ead41428d2aa2da3"

    tfrt_http_archive(
        name = name,
        build_file = "//third_party/llvm:BUILD",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
    )
