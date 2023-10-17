"""Provides the repository macro to import TSL."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports TSL."""

    # Attention: tools parse and update these lines.
    TSL_COMMIT = "4427d2138fb68a39174310c7362092bd0eea6596"
    TSL_SHA256 = "229b73685e2c650b325378100da10d892578c96ee82b6f06071ce55c328db0ca"

    SYS_DIRS = [
        "concurrency",
    ]
    SYS_LINKS = {
        "//third_party/tsl:system.tsl.{name}.BUILD".format(name = n): "tsl/{name}/BUILD.bazel".format(name = n)
        for n in SYS_DIRS
    }

    tfrt_http_archive(
        name = name,
        sha256 = TSL_SHA256,
        build_file = "//third_party/tsl:local_tsl.BUILD",
        system_build_file = "//third_party/tsl:system.BUILD",
        system_link_files = SYS_LINKS,
        strip_prefix = "tsl-{commit}".format(commit = TSL_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/tsl/archive/{commit}.tar.gz".format(commit = TSL_COMMIT),
            "https://github.com/google/tsl/archive/{commit}.tar.gz".format(commit = TSL_COMMIT),
        ],
    )
