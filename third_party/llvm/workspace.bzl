"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "89786c2b992c3cb4c4a230542d2af34ec2915a08"
    LLVM_SHA256 = "d6a6c9b67f6557dcdfd0e565d047b676a7cf8aeb70fa2bf8c0fa92ddc1a05cd5"

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
