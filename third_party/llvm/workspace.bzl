"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "3c58e53041fcdeae36ef5ca1e0683e0f1f16bf69"
    LLVM_SHA256 = "c00ec33ff66b424d0ab42a4e67bf7a91695ea0555b6207845016735309183376"

    tfrt_http_archive(
        name = name,
        build_file = "//third_party/llvm:BUILD",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        patch_file = "//third_party/llvm:zstd.patch",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
    )
