"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "b8a062061571b7868013f1fefb891bdaa2da1adc"
    LLVM_SHA256 = "b5cafaa5d5f80f25e701a5c5e37898c2b6e0f925db298190c2b694f6e328275d"

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
