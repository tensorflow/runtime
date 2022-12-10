"""Provides the repository macro to import TSL."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports TSL."""

    # Attention: tools parse and update these lines.
    TSL_COMMIT = "c0be4ec60e391615e8bb78de3a6b78068ed6d67e"
    TSL_SHA256 = "f96c508e2587c4d4979a77ea9d47e79465d746db798db0f689f0988465dc3689"

    tfrt_http_archive(
        name = name,
        build_file = "//third_party/tsl:BUILD",
        sha256 = TSL_SHA256,
        strip_prefix = "tsl-{commit}".format(commit = TSL_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/tsl/archive/{commit}.tar.gz".format(commit = TSL_COMMIT),
            "https://github.com/google/tsl/archive/{commit}.tar.gz".format(commit = TSL_COMMIT),
        ],
    )
