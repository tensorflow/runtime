"""Dependencies for CUDA rules."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _local_cuda_impl(repository_ctx):
    # Path to CUDA Toolkit is
    # - taken from CUDA_PATH environment variable or
    # - determined through 'which ptxas' or
    # - defaults to '/usr/local/cuda'
    cuda_path = "/usr/local/cuda"
    ptxas_path = repository_ctx.which("ptxas")
    if ptxas_path:
        cuda_path = ptxas_path.dirname.dirname
    cuda_path = repository_ctx.os.environ.get("CUDA_PATH", cuda_path)

    if repository_ctx.path(cuda_path).exists:
        repository_ctx.symlink(cuda_path, "cuda")
        cuda_symlink = str(repository_ctx.path("cuda"))
        build_file = "//private:BUILD.local_cuda"
    else:
        cuda_symlink = None
        build_file = "//private:BUILD"  # Empty file

    repository_ctx.symlink(Label(build_file), "BUILD")
    repository_ctx.file("defs.bzl", "cuda_path = %s\n" % repr(cuda_symlink))

_local_cuda = repository_rule(
    implementation = _local_cuda_impl,
    environ = ["CUDA_PATH", "PATH"],
    # remotable = True,
)

def rules_cuda_dependencies():
    maybe(
        name = "bazel_skylib",
        repo_rule = http_archive,
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        ],
    )

    _local_cuda(name = "local_cuda")
