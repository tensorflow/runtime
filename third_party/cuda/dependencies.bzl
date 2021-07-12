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
    build_file = Label("//third_party/cuda:cuda_headers.BUILD")
    patch_file = Label("//third_party/cuda:cuda_headers.patch")

    print("\n\033[22;33mNOTICE:\033[0m The following command will download " +
          "NVIDIA proprietary software. By using the software you agree to " +
          "comply with the terms of the license agreement that accompanies " +
          "the software. If you do not agree to the terms of the license " +
          "agreement, do not use the software.")

    tag = "cuda-11.4.0"
    for name, sha256 in [
        ("cublas", "ed5dfe78ec300d86ea724c512c0707456ce6240ee746c705192158f09691f108"),
        ("cudart", "62fa6cfd28e09a043d326801b5d8d3b08c4e44278dbb4e8e6b2f8fd721baa5a0"),
        ("cufft", "1bae8e1375b6bfedb3f3cb07c9480b2898ec21bce3d4285b7b2764ebddca4551"),
        ("cupti", "4cffaa710b18d58cf21fdc93994de7b86d28c0655915cfc5e5d2483482f89a40"),
        ("curand", "06d28c1205e1648d60d61ff009bc5a094629e3c3188ce21859778b07d307dc25"),
        ("cusolver", "15c5d7c15f462c76bad5a3bbb99c776111264b1255e166399673bd167354718c"),
        ("cusparse", "a8e8302ba702fc64262c750fc7af57c9eea51bafe1344074a00959c20865b875"),
        ("npp", "7130787ac4a97baa3056b6dcd9d12357b0a4b3ca3649f027f416a4bf88817b57"),
        ("nvjpeg", "7d3d74e5774f80536760012b6f54ba0247e9abd748f5e0cc237853495ae8affc"),
        ("nvrtc", "8ca8524821eae8a37e1ae314d2d598d0949f0aad4034cbe725c55ac8c8afad3a"),
    ]:
        url = "cuda-individual/{name}/-/archive/{tag}/{name}-{tag}.tar.gz".format(name = name, tag = tag)
        strip_prefix = "{name}-{tag}".format(name = name, tag = tag)
        _download_nvidia_headers(repository_ctx, "cuda", url, sha256, strip_prefix)

    repository_ctx.symlink(build_file, "BUILD")
    repository_ctx.patch(patch_file)

def _cudnn_headers_impl(repository_ctx):
    build_file = Label("//third_party/cuda:cudnn_headers.BUILD")
    patch_file = Label("//third_party/cuda:cudnn_headers.patch")

    tag = "v8.2.1.32"
    url = "cudnn/-/archive/{tag}/cudnn-{tag}.tar.gz".format(tag = tag)
    strip_prefix = "cudnn-{tag}".format(tag = tag)
    sha256 = "40a3b87ae7d258881a1ff18ecd19e19752f7b1c463ae9dceb575dfdaad8085d9"
    _download_nvidia_headers(repository_ctx, "cudnn", url, sha256, strip_prefix)

    repository_ctx.symlink(build_file, "BUILD")
    repository_ctx.patch(patch_file)

_cuda_headers = repository_rule(
    implementation = _cuda_headers_impl,
    # remotable = True,
)

_cudnn_headers = repository_rule(
    implementation = _cudnn_headers_impl,
    # remotable = True,
)

def cuda_dependencies():
    _cuda_headers(name = "cuda_headers")
    _cudnn_headers(name = "cudnn_headers")
