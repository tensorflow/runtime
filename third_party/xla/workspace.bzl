"""Provides the repository macro to import TSL."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports XLA."""

    # Attention: tools parse and update these lines.
    XLA_COMMIT = "441aad6966f40917da1116fd25ec6c7839be0ef6"
    XLA_SHA256 = "178637d895096b073efcbee8b2d36cc663bcdd55d63f273c6be86609b2916572"

    tfrt_http_archive(
        name = name,
        sha256 = XLA_SHA256,
        strip_prefix = "xla-{commit}".format(commit = XLA_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT),
            "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT),
        ],
    )
