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

    tag = "cuda-11.4.3"
    for name, sha256 in [
        ("cccl", "27aeb3f58ff9f549879dbf812715ac56f9cf706546bed80ba76848467cb34810"),
        ("cublas", "dba442b4532b42dd355eec4a3e9c1c8dfe54825a92782e98ee5ced4c1700f0e0"),
        ("cudart", "ba56549b23ebd887f75d4523a5e33f4d0e96bc93f199ca2ed3d6ec2c83a2cef2"),
        ("cufft", "66a6591e20e3fe61a77bf84a8286d2de0c98a831f3c95bf2725961d7ca7d53a8"),
        ("cupti", "d6ae8a73b8f3cc2c61efd8d620c483cf3fb08220b20f0c593a00a089713e7cc9"),
        ("curand", "cd2702cff984f82f22d5a12de899224ee75e8696dcf2e621a4c726cfddbfc448"),
        ("cusolver", "3cd40aa0c003ed10c22b3af63f64767d0a0d0567fa2c27f446b318566ed5f709"),
        ("cusparse", "0472a77c60859d679b3c8643e65ad895e25c362961b718f4125bcec4e4991ec7"),
        ("npp", "c06c3f9ea86d3c5f7ef962765866f30870b85bcb195837c6572a1fafc900e908"),
        ("nvcc", "8abce627612660a285e8698e2a43348202f1e193f8f70208b8edf315a8591505"),
        ("nvjpeg", "70a756a6ad813adab5eb77a6ae92154280d3847d005ca145485e04178893cc94"),
        ("nvrtc", "b38d791cdd0d90eb941b765ee592644a358790536d27508b48e21c029e960aa5"),
    ]:
        url = "cuda-individual/{name}/-/archive/{tag}/{name}-{tag}.tar.gz".format(name = name, tag = tag)
        strip_prefix = "{name}-{tag}".format(name = name, tag = tag)
        _download_nvidia_headers(repository_ctx, "cuda", url, sha256, strip_prefix)

    repository_ctx.symlink(build_file, "BUILD")
    repository_ctx.patch(patch_file)

def _cudnn_headers_impl(repository_ctx):
    build_file = Label("//third_party/cuda:cudnn_headers.BUILD")
    patch_file = Label("//third_party/cuda:cudnn_headers.patch")

    tag = "v8.2.4.15"
    url = "cudnn/-/archive/{tag}/cudnn-{tag}.tar.gz".format(tag = tag)
    strip_prefix = "cudnn-{tag}".format(tag = tag)
    sha256 = "a5a2749cee42dd0a175d6dfcfbab7e64acee55210febe2f32d4605eef32591af"
    _download_nvidia_headers(repository_ctx, "cudnn", url, sha256, strip_prefix)

    repository_ctx.symlink(build_file, "BUILD")
    repository_ctx.patch(patch_file)

def _cudnn_frontend_impl(repository_ctx):
    build_file = Label("//third_party/cuda:cudnn_frontend.BUILD")

    version = "0.4.1"
    url = "archive/refs/tags/v{version}.tar.gz".format(version = version)
    strip_prefix = "cudnn-frontend-{version}".format(version = version)
    sha256 = "e0cef5e4440d24115c770160ba4a08821ae24c357a1623a7e9ca736ed685131c"
    repository_ctx.download_and_extract(
        url = [
            "http://github.com/NVIDIA/cudnn-frontend/" + url,
            "http://mirror.tensorflow.org/github.com/NVIDIA/cudnn-frontend/" + url,
        ],
        output = "cudnn_frontend",
        sha256 = sha256,
        stripPrefix = strip_prefix,
    )

    repository_ctx.symlink(build_file, "BUILD")

def _nccl_headers_impl(repository_ctx):
    build_file = Label("//third_party/cuda:nccl_headers.BUILD")
    patch_file = Label("//third_party/cuda:nccl_headers.patch")

    tag = "2.8.3-1"
    url = "nccl/archive/v{tag}.tar.gz".format(tag = tag)
    strip_prefix = "nccl-{tag}".format(tag = tag)
    sha256 = "3ae89ddb2956fff081e406a94ff54ae5e52359f5d645ce977c7eba09b3b782e6"
    repository_ctx.download_and_extract(
        url = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nvidia/" + url,
            "https://github.com/nvidia/" + url,
        ],
        output = "nccl",
        sha256 = sha256,
        stripPrefix = strip_prefix,
    )

    repository_ctx.symlink(build_file, "BUILD")
    repository_ctx.patch(patch_file)
    repository_ctx.symlink("nccl/src/nccl.h.in", "nccl/src/nccl.h")

_cuda_headers = repository_rule(
    implementation = _cuda_headers_impl,
    # remotable = True,
)

_cudnn_headers = repository_rule(
    implementation = _cudnn_headers_impl,
    # remotable = True,
)

_cudnn_frontend = repository_rule(
    implementation = _cudnn_frontend_impl,
    # remotable = True,
)

_nccl_headers = repository_rule(
    implementation = _nccl_headers_impl,
    # remotable = True,
)

def cuda_dependencies():
    _cuda_headers(name = "cuda_headers")
    _cudnn_headers(name = "cudnn_headers")
    _cudnn_frontend(name = "cudnn_frontend")
    _nccl_headers(name = "nccl_headers")
