"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "7b35638ddb99a0298c5d3450de506a8e8e0203d3"
    EIGEN_SHA256 = "2f25d7d0279c57ce7c533bc71ba78af9c24a0a0aac4102bfeb28c2b5737499d1"

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
