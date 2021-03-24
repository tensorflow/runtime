"""CUDA headers repository."""

def _download_nvidia_headers(repository_ctx, output, url, sha256, strip_prefix):
    # Keep the mirror up-to-date manually (see b/154869892) with:
    # /google/bin/releases/tensorflow-devinfra-team/cli_tools/tf_mirror <url>
    repository_ctx.download_and_extract(
        url = [
            "http://gitlab.com/nvidia/headers/" + url,
            "http://mirror.tensorflow.org/gitlab.com/nvidia/headers/" + url,
        ],
        output = output,
        sha256 = sha256,
        stripPrefix = strip_prefix,
    )

def _cuda_headers_impl(repository_ctx):
    tag = "cuda-10-2"
    for name, sha256 in [
        ("cublas", "9537c3e89a85ea0082217e326cd8e03420f7723e05c98d730d80bda8b230c81b"),
        ("cudart", "8a203bd87a2fde37608e8bc3c0c9347b40586906c613b6bef0bfc3995ff40099"),
        ("cufft", "bac1602183022c7a9c3e13078fcac59e4eee0390afe99c3c7348c894a97e19dd"),
        ("cusolver", "68e049c1d27ad3558cddd9ad82cf885b6789f1f01934f9b60340c391fa8e6279"),
        ("misc", "5e208a8e0f25c9df41121f0502eadae903fa64f808437516198004bdbf6af04b"),
    ]:
        url = "cuda-individual/{name}/-/archive/{tag}/{name}-{tag}.tar.gz".format(name = name, tag = tag)
        strip_prefix = "{name}-{tag}".format(name = name, tag = tag)
        _download_nvidia_headers(repository_ctx, "cuda", url, sha256, strip_prefix)

    repository_ctx.symlink(Label("//third_party/cuda:cuda_headers.BUILD"), "BUILD")

def _cudnn_headers_impl(repository_ctx):
    tag = "v7.6.5"
    url = "cudnn/-/archive/{tag}/cudnn-{tag}.tar.gz".format(tag = tag)
    strip_prefix = "cudnn-{tag}".format(tag = tag)
    sha256 = "ef45f4649328da678285b8ce589a8296cedcc93819ffdbb5eea5346a0619a766"
    _download_nvidia_headers(repository_ctx, "cudnn", url, sha256, strip_prefix)

    repository_ctx.symlink(Label("//third_party/cuda:cudnn_headers.BUILD"), "BUILD")

_cuda_headers = repository_rule(
    implementation = _cuda_headers_impl,
    # remotable = True,
)

_cudnn_headers = repository_rule(
    implementation = _cudnn_headers_impl,
    # remotable = True,
)

def cuda_dependencies():
    print("The following command will download NVIDIA proprietary " +
          "software. By using the software you agree to comply with the " +
          "terms of the license agreement that accompanies the software. " +
          "If you do not agree to the terms of the license agreement, do " +
          "not use the software.")

    _cuda_headers(name = "cuda_headers")
    _cudnn_headers(name = "cudnn_headers")
