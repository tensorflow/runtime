"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "7db0ac977acf276fb0817cfb89e490cdbae0ab56"
    EIGEN_SHA256 = "f4697dcc5f0545cf9ab70be8f6b313bdc31c0405f0ed30f8967bc02e7d7aa12a"

    tfrt_http_archive(
        name = name,
        build_file = "//third_party/eigen:BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
    )
