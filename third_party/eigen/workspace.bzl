"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "085c2fc5d53f391afcccce21c45e15f61c827ab1"
    EIGEN_SHA256 = "cd72f0a56a95d85cb8a0160f4adc7fea72da49fbb7351ebb31c4e67e1a5fc8bd"

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
