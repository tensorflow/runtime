"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "5ed7a86ae96d411c450fb190f5a725f38f2aea9d"
    EIGEN_SHA256 = "99a208f314391f3ef930ff66ae15b2e332b7370444c9f96562c3b4eb0da889fe"

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
