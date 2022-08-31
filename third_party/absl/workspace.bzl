"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports absl."""

    # Attention: tools parse and update these lines.
    ABSL_COMMIT = "273292d1cfc0a94a65082ee350509af1d113344d"
    ABSL_SHA256 = "94aef187f688665dc299d09286bfa0d22c4ecb86a80b156dff6aabadc5a5c26d"

    SYS_DIRS = [
        "algorithm",
        "base",
        "cleanup",
        "container",
        "debugging",
        "flags",
        "functional",
        "hash",
        "memory",
        "meta",
        "numeric",
        "random",
        "status",
        "strings",
        "synchronization",
        "time",
        "types",
        "utility",
    ]
    SYS_LINKS = {
        "//third_party/absl:system.absl.{name}.BUILD".format(name = n): "absl/{name}/BUILD.bazel".format(name = n)
        for n in SYS_DIRS
    }

    tfrt_http_archive(
        name = name,
        sha256 = ABSL_SHA256,
        build_file = "//third_party/absl:absl.BUILD",
        system_build_file = "//third_party/absl:system.BUILD",
        system_link_files = SYS_LINKS,
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT),
            "https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT),
        ],
    )
