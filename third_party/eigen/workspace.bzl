"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "b3bf8d6a13585ff248c079402654647d298de60b"
    EIGEN_SHA256 = "2ec7ccfe975d2e05768172b7b98975852e8f739f776ea3ae7ee9fc981dcc38de"

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
